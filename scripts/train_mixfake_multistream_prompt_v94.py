#!/usr/bin/env python3
"""Fine-tune a task-specific MixFake multi-stream prompt specialist.

This follows the released MixFake design (base/frequency/texture prompts) on
the original mixture, without source separation.  The XLS-R backbone stays
frozen; the prompt modules and Spectra-AASIST backend are trainable.  Clean and
content-matched fast telephone views share the exact same crop.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import random
import sys
import time

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_curve
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from data_guard import assert_no_locked_eval_leakage  # noqa: E402
from multistream_prompt_spectra import (  # noqa: E402
    MultiStreamSpectraMultitask, warm_start_shared_backend,
)
from pipeline import load_audio  # noqa: E402
from telephone_channel import apply_channel  # noqa: E402
from train_wpt_spectra_multitask import load_spectra, preemphasis, worker_seed  # noqa: E402
from train_mixfake_component_readout_v93 import sample_weights  # noqa: E402


TASK_INDEX = {"voice": 0, "music": 1, "file": 2}
FAST_CHANNELS = ("resample8k", "mulaw_numpy", "fft_narrowband", "lowpass_5k")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def finite_eer(labels, scores) -> float:
    labels = np.asarray(labels, np.int64)
    scores = np.asarray(scores, np.float64)
    if len(np.unique(labels)) != 2:
        return float("nan")
    fpr, tpr, _ = roc_curve(labels, scores, pos_label=1, drop_intermediate=False)
    fnr = 1 - tpr
    index = int(np.argmin(np.abs(fpr - fnr)))
    return float((fpr[index] + fnr[index]) / 2)


def read_manifest(path: Path, audio_root: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, dtype={"ID": str}, low_memory=False)
    required = {
        "ID", "VOICE_FAKE", "MUSIC_FAKE", "FILE_FAKE",
        "VOICE_PRESENT", "MUSIC_PRESENT",
    }
    if missing := required.difference(frame.columns):
        raise ValueError(f"manifest misses {sorted(missing)}")
    frame = frame.copy()
    if "PATH" in frame and frame.PATH.notna().all():
        frame["PATH"] = frame.PATH.astype(str)
    elif "ARCHIVE_MEMBER" in frame:
        frame["PATH"] = frame.ARCHIVE_MEMBER.map(
            lambda value: str(audio_root / str(value))
        )
    else:
        raise ValueError("manifest requires PATH or ARCHIVE_MEMBER")
    if "COMPONENT_CASE" not in frame:
        frame["COMPONENT_CASE"] = "not_mixed"
    if "CELL_V57" in frame:
        frame["COMPONENT_CASE"] = frame.COMPONENT_CASE.fillna(frame.CELL_V57)
    frame["COMPONENT_CASE"] = frame.COMPONENT_CASE.fillna("not_mixed").astype(str)
    if "MUSIC_GENERATOR" not in frame:
        frame["MUSIC_GENERATOR"] = "unspecified"
    if "MUSIC_GENERATOR_V57" in frame:
        frame["MUSIC_GENERATOR"] = frame.MUSIC_GENERATOR.fillna(
            frame.MUSIC_GENERATOR_V57
        )
    frame["MUSIC_GENERATOR"] = frame.MUSIC_GENERATOR.fillna("unspecified").astype(str)
    # Full extraction integrity (all 11,365 files and hashes) is established by
    # prepare_mixfake_alltype_v93.py. Avoid thousands of serial NAS stat calls
    # here; training will still fail immediately if any selected file is absent.
    probe = frame.PATH.iloc[[0, -1]] if len(frame) > 1 else frame.PATH
    missing = [path for path in probe if not Path(path).is_file()]
    if missing:
        raise FileNotFoundError(missing[0])
    return frame


def windows_at(audio: np.ndarray, length: int, starts: np.ndarray) -> np.ndarray:
    if len(audio) < length:
        repeated = np.tile(audio, math.ceil(length / max(1, len(audio))))[:length]
        return np.repeat(repeated[None], len(starts), axis=0).astype(np.float32)
    maximum = len(audio) - length
    starts = np.clip(starts, 0, maximum)
    return np.stack([audio[start:start + length] for start in starts]).astype(np.float32)


class PairedMixtures(Dataset):
    def __init__(self, frame: pd.DataFrame, window: int, views: int,
                 training: bool, channel: bool) -> None:
        self.frame = frame.reset_index(drop=True)
        self.window = int(window)
        self.views = int(views)
        self.training = bool(training)
        self.channel = bool(channel)

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int):
        row = self.frame.iloc[index]
        audio = load_audio(Path(row.PATH))
        if len(audio) < self.window:
            starts = np.zeros(self.views, dtype=np.int64)
        else:
            maximum = len(audio) - self.window
            if self.training:
                starts = np.random.randint(maximum + 1, size=self.views)
            else:
                starts = np.rint(np.linspace(0, maximum, self.views)).astype(np.int64)
        clean = windows_at(audio, self.window, starts)
        if self.channel:
            if "PHONE_PATH" in self.frame and isinstance(row.PHONE_PATH, str):
                changed_audio = load_audio(Path(row.PHONE_PATH))
            else:
                channel_name = FAST_CHANNELS[
                    int(hashlib.sha256(str(row.ID).encode()).hexdigest()[:8], 16)
                    % len(FAST_CHANNELS)
                ]
                changed_audio = apply_channel(audio, channel_name, key=index)
            changed = windows_at(changed_audio, self.window, starts)
        else:
            changed = clean.copy()
        if self.training:
            gain = np.float32(np.random.uniform(.65, 1.35))
            clean *= gain; changed *= gain
        target = np.asarray(
            [row.VOICE_FAKE, row.MUSIC_FAKE, row.FILE_FAKE], dtype=np.float32
        )
        return clean, changed, target, index


def task_rank(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    positive, negative = logits[target.eq(1)], logits[target.eq(0)]
    if not len(positive) or not len(negative):
        return logits.sum() * 0
    return F.softplus(-(positive[:, None] - negative[None])).mean()


@torch.inference_mode()
def evaluate(
    model, loader, device, task: int, *, use_changed: bool = False,
) -> tuple[float, np.ndarray]:
    model.eval()
    scores, indices = [], []
    for clean, changed, _target, index in loader:
        waveform = changed if use_changed else clean
        waveform = preemphasis(waveform.to(device, non_blocking=True))
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                            enabled=device.type == "cuda"):
            output = model(waveform)[:, task]
        scores.extend(output.float().sigmoid().cpu().tolist())
        indices.extend(index.tolist())
    frame = loader.dataset.frame.iloc[indices]
    label = frame[["VOICE_FAKE", "MUSIC_FAKE", "FILE_FAKE"]].iloc[:, task]
    return finite_eer(label, scores), np.asarray(scores, np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-manifest", type=Path, required=True)
    parser.add_argument("--development-manifest", type=Path, required=True)
    parser.add_argument("--audio-root", type=Path, required=True)
    parser.add_argument(
        "--paired-audio-root", type=Path, default=None,
        help="Optional exact-codec directory containing <ID>.wav partners.",
    )
    parser.add_argument("--model-dir", type=Path,
                        default=ROOT / "models/external/spectra_aasist")
    parser.add_argument("--warm-start", type=Path, default=(
        ROOT / "v50_v57m_challenger_v2/model/spectra-aasist/"
        "wpt_spectra_multitask.pt"
    ))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--task", choices=tuple(TASK_INDEX), default="music")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=20260928)
    parser.add_argument("--window", type=int, default=64600)
    parser.add_argument("--views", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--eval-batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--samples-per-epoch", type=int, default=4096)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--prompt-lr", type=float, default=2e-3)
    parser.add_argument("--head-lr", type=float, default=5e-4)
    parser.add_argument("--backend-lr", type=float, default=5e-5)
    parser.add_argument("--ranking-weight", type=float, default=.10)
    parser.add_argument("--consistency-weight", type=float, default=.10)
    parser.add_argument(
        "--artifactbench-case-weight", type=float, default=1.,
        help=("Relative sampler mass for ArtifactBench rows after the usual "
              "component-case/generator balancing."),
    )
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if "locked" in " ".join(map(str, [args.train_manifest, args.development_manifest, args.output])).lower():
        raise ValueError("locked data are forbidden during training")
    assert_no_locked_eval_leakage(args.train_manifest, ROOT / "configs/data_partitions.yaml")
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(args.seed)
    train = read_manifest(args.train_manifest, args.audio_root)
    development = read_manifest(args.development_manifest, args.audio_root)
    # Component EER is conditional on presence.  Absent-component truth is
    # intentionally NaN in the broad catalog and must never be coerced to real.
    if args.task in ("voice", "music"):
        present = f"{args.task.upper()}_PRESENT"
        train = train.loc[train[present].eq(1)].reset_index(drop=True)
        development = development.loc[development[present].eq(1)].reset_index(drop=True)
    if args.paired_audio_root is not None:
        for frame, split in ((train, "train"), (development, "development")):
            root = args.paired_audio_root / split
            paths = frame.ID.map(lambda value: root / f"{value}.wav")
            missing = [str(path) for path in paths if not path.is_file()]
            if missing:
                raise FileNotFoundError(missing[0])
            frame["PHONE_PATH"] = paths.map(str)
    if set(train.ID) & set(development.ID):
        raise ValueError("train/development ID overlap")
    if args.smoke:
        # Manifests evolved from compact RR/RF/FR/FF names to descriptive
        # component cases.  Select from the cases that are actually present so
        # smoke tests cannot silently construct an empty sampler.
        def smoke_rows(frame: pd.DataFrame) -> pd.DataFrame:
            pieces = [group.head(8) for _, group in frame.groupby(
                "COMPONENT_CASE", sort=True
            )]
            selected = pd.concat(pieces, ignore_index=True)
            label = selected[["VOICE_FAKE", "MUSIC_FAKE", "FILE_FAKE"]].iloc[
                :, TASK_INDEX[args.task]
            ]
            if selected.empty or label.dropna().nunique() != 2:
                raise ValueError(
                    f"smoke subset for {args.task} must contain both labels"
                )
            return selected

        train = smoke_rows(train)
        development = smoke_rows(development)
        args.samples_per_epoch = 32
        args.epochs = 1
        args.patience = 1
        args.batch_size = min(args.batch_size, 4)
        args.eval_batch_size = min(args.eval_batch_size, 8)
    train_data = PairedMixtures(train, args.window, args.views, True, True)
    dev_clean = PairedMixtures(development, args.window, args.views, False, False)
    dev_phone = PairedMixtures(development, args.window, args.views, False, True)
    sampling_weights = sample_weights({
        "component_case": train.COMPONENT_CASE.to_numpy(),
        "music_generator": train.MUSIC_GENERATOR.to_numpy(),
    })
    if args.artifactbench_case_weight <= 0:
        raise ValueError("artifactbench-case-weight must be positive")
    if "DATASET" in train:
        external = train.DATASET.fillna("").eq("artifactbench_v1_train").to_numpy()
        sampling_weights[torch.from_numpy(external)] *= args.artifactbench_case_weight
    sampler = WeightedRandomSampler(
        sampling_weights,
        args.samples_per_epoch, replacement=True,
        generator=torch.Generator().manual_seed(args.seed),
    )
    common = dict(num_workers=args.workers, pin_memory=True,
                  worker_init_fn=worker_seed, persistent_workers=args.workers > 0)
    train_loader = DataLoader(train_data, batch_size=args.batch_size, sampler=sampler, **common)
    clean_loader = DataLoader(dev_clean, batch_size=args.eval_batch_size, shuffle=False, **common)
    phone_loader = DataLoader(dev_phone, batch_size=args.eval_batch_size, shuffle=False, **common)

    device = torch.device(args.device)
    model = MultiStreamSpectraMultitask(load_spectra(args.model_dir, device)).to(device)
    warm = torch.load(args.warm_start, map_location="cpu", weights_only=False)
    warm_type = warm.get("model_type")
    if warm_type == "wpt_spectra_multitask":
        # The original v94 experiment changes prompt semantics, so only the
        # architecture-compatible Spectra/AASIST backend and task head are
        # transferred from the WPT anchor.
        warm_report = warm_start_shared_backend(model, warm["state"])
        warm_report["source_model_type"] = warm_type
    elif warm_type == "mixfake_multistream_prompt_v94":
        # A second-stage codec adaptation must retain the already learned
        # base/frequency/texture prompts as well as the shared backend.  The
        # compact loader verifies that only the frozen XLS-R backbone is
        # absent; it will reject a silently incompatible checkpoint.
        config = warm.get("config", {})
        expected = {
            "base_tokens": model.prompt_encoder.prompts.base_tokens,
            "frequency_tokens": model.prompt_encoder.prompts.frequency_tokens,
            "texture_tokens": model.prompt_encoder.prompts.texture_tokens,
        }
        mismatch = {
            key: (config.get(key), value)
            for key, value in expected.items()
            if config.get(key) != value
        }
        if mismatch:
            raise ValueError(f"multi-stream warm-start config mismatch: {mismatch}")
        model.load_trainable_state_dict(warm["state"])
        warm_report = {
            "source_model_type": warm_type,
            "tensors_loaded": len(warm["state"]),
            "parameters_loaded": sum(value.numel() for value in warm["state"].values()),
            "source_task": config.get("task"),
        }
    else:
        raise ValueError(
            "warm-start must be a WPT or MixFake multi-stream checkpoint; "
            f"got {warm_type!r}"
        )
    prompts = list(model.prompt_encoder.prompts.parameters())
    head = list(model.task_head.parameters())
    excluded = {id(value) for value in prompts + head}
    backend = [value for value in model.parameters()
               if value.requires_grad and id(value) not in excluded]
    optimizer = torch.optim.AdamW([
        {"params": prompts, "lr": args.prompt_lr},
        {"params": head, "lr": args.head_lr},
        {"params": backend, "lr": args.backend_lr},
    ], weight_decay=5e-4)
    task = TASK_INDEX[args.task]
    best, best_state, stale, history = float("inf"), None, 0, []
    started = time.monotonic()
    for epoch in range(args.epochs):
        model.train(); losses = []
        for step, (clean, phone, target, _) in enumerate(train_loader):
            target = target[:, task].to(device, non_blocking=True)
            clean = preemphasis(clean.to(device, non_blocking=True))
            phone = preemphasis(phone.to(device, non_blocking=True))
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                                enabled=device.type == "cuda"):
                clean_logit = model(clean)[:, task]
                phone_logit = model(phone)[:, task]
                bce = .5 * (
                    F.binary_cross_entropy_with_logits(clean_logit, target)
                    + F.binary_cross_entropy_with_logits(phone_logit, target)
                )
                rank = .5 * (
                    task_rank(clean_logit, target) + task_rank(phone_logit, target)
                )
                consistency = F.smooth_l1_loss(phone_logit, clean_logit.detach())
                loss = bce + args.ranking_weight * rank + args.consistency_weight * consistency
            optimizer.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [value for value in model.parameters() if value.requires_grad], 3.,
                error_if_nonfinite=True,
            )
            optimizer.step(); losses.append(float(loss.detach()))
            if step and step % 100 == 0:
                print(json.dumps({"epoch": epoch, "step": step,
                                  "loss": float(np.mean(losses[-100:]))}), flush=True)
        clean_eer, clean_scores = evaluate(model, clean_loader, device, task)
        phone_eer, phone_scores = evaluate(
            model, phone_loader, device, task, use_changed=True
        )
        selection = .65 * (clean_eer + phone_eer) / 2 + .35 * max(clean_eer, phone_eer)
        record = {"epoch": epoch, "loss": float(np.mean(losses)),
                  "clean_eer": clean_eer, "phone_eer": phone_eer,
                  "selection_error": selection,
                  "seconds": time.monotonic() - started}
        history.append(record); print(json.dumps(record), flush=True)
        if selection < best - 1e-5:
            best, stale = selection, 0
            best_state = copy.deepcopy(model.trainable_state_dict())
            best_scores = {"clean": clean_scores, "phone": phone_scores}
            best_epoch = epoch
        else:
            stale += 1
            if stale >= args.patience: break
    if best_state is None:
        raise RuntimeError("no checkpoint selected")
    args.output.mkdir(parents=True)
    checkpoint = {
        "model_type": "mixfake_multistream_prompt_v94",
        "state": best_state,
        "config": {"temperature": model.temperature, "window": args.window,
                   "views": args.views, "task": args.task,
                   "base_tokens": model.prompt_encoder.prompts.base_tokens,
                   "frequency_tokens": model.prompt_encoder.prompts.frequency_tokens,
                   "texture_tokens": model.prompt_encoder.prompts.texture_tokens,
                   "prompt_dropout": model.prompt_encoder.prompts.dropout.p,
                   "artifactbench_case_weight": args.artifactbench_case_weight},
        "best_epoch": best_epoch, "selection_error": best,
        "seed": args.seed, "warm_start_report": warm_report,
        "provenance": {
            "train_manifest": str(args.train_manifest),
            "train_manifest_sha256": sha256_file(args.train_manifest),
            "development_manifest": str(args.development_manifest),
            "development_manifest_sha256": sha256_file(args.development_manifest),
            "warm_start_sha256": sha256_file(args.warm_start),
            "locked_scores_read": False,
        },
    }
    torch.save(checkpoint, args.output / "multistream_prompt.pt")
    pd.DataFrame(history).to_csv(args.output / "history.csv", index=False)
    np.savez_compressed(args.output / "development_predictions.npz",
                        ids=development.ID.to_numpy(dtype=str), **best_scores)
    development_label = development[
        ["VOICE_FAKE", "MUSIC_FAKE", "FILE_FAKE"]
    ].iloc[:, task]
    (args.output / "summary.json").write_text(json.dumps({
        "best_epoch": best_epoch, "selection_error": best,
        "clean_eer": finite_eer(development_label, best_scores["clean"]),
        "phone_eer": finite_eer(development_label, best_scores["phone"]),
        "elapsed_seconds": time.monotonic() - started,
    }, indent=2) + "\n")
    print(json.dumps({"status": "complete", "best_epoch": best_epoch,
                      "selection_error": best, "output": str(args.output)}), flush=True)


if __name__ == "__main__":
    main()
