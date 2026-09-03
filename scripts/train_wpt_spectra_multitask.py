#!/usr/bin/env python3
"""Co-train compact WPT-XLS-R+AASIST on approved all-type audio banks."""

from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import math
import random
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from safetensors.torch import load_file
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from data_guard import assert_no_locked_eval_leakage  # noqa: E402
from evaluate_diagnostic import official_eer  # noqa: E402
from pipeline import find_audio_files, load_audio  # noqa: E402
from wpt_spectra import (  # noqa: E402
    WPTSpectraMultitask, component_loss, component_ranking_loss,
)


TRAIN_DEFAULT = (
    "forensic_call_train_v1",
    "external_mixed_train_v1",
    "mixed_devvoice_train_v1",
    "mixed_fmc_music_train_v1",
    "mixfake_music_train_v1",
    "telephone_mixed_train_v1",
    "temporal_mixed_train_v2",
    "channel_invariant_factorial_train_v1",
)
DEV_DEFAULT = (
    "factorial_eval_1200_v2_dev",
    "mixfake_music_dev_v1",
    "external_mixed_v1",
    "source_disjoint_mixed_v1",
    "source_disjoint_mixed_equal_v1",
    "source_disjoint_music_v1",
    "telephone_mixed_dev_v1",
    "asvspoof_voice_wav_dev_v1",
)


def truth_path(name: str) -> Path:
    factorial_splits = {
        "factorial_eval_1200_v2_dev": "truth_dev.csv",
        "factorial_eval_1200_v2_holdout": "truth_holdout.csv",
        "factorial_eval_1200_v2_locked": "truth_locked.csv",
    }
    if name in factorial_splits:
        return ROOT / "data/eval/factorial_eval_1200_v2" / factorial_splits[name]
    return ROOT / "data/eval" / name / "truth.csv"


def data_name(name: str) -> str:
    return "factorial_eval_1200_v2" if name.startswith(
        "factorial_eval_1200_v2_"
    ) else name


def load_frame(name: str, role: str) -> pd.DataFrame:
    path = truth_path(name)
    frame = pd.read_csv(path, dtype={"ID": str})
    if role == "train" and name == "temporal_mixed_train_v2" and "SPLIT" in frame:
        frame = frame[frame.SPLIT.eq("train")]
    audio_dir = ROOT / "data/eval" / data_name(name) / "audio"
    paths = {path.stem: path for path in find_audio_files(audio_dir)}
    missing = sorted(set(frame.ID).difference(paths))
    if missing:
        raise FileNotFoundError(
            f"{name}: {len(missing)} IDs have no audio: {missing[:5]}"
        )
    frame = frame.copy()
    frame["DATASET"] = name
    frame["PATH"] = [str(paths[item]) for item in frame.ID]
    return frame.reset_index(drop=True)


def _repeat_or_crop(
    audio: np.ndarray, window: int, views: int, stochastic: bool,
) -> np.ndarray:
    if len(audio) < window:
        if not len(audio):
            audio = np.zeros(1, dtype=np.float32)
        repeated = np.tile(audio, math.ceil(window / len(audio)))[:window]
        return np.repeat(repeated[None], views, axis=0)
    maximum = len(audio) - window
    if views == 1:
        starts = np.asarray([maximum // 2])
    else:
        starts = np.rint(np.linspace(0, maximum, views)).astype(np.int64)
    if stochastic and maximum:
        radius = max(1, min(window // 3, maximum // max(1, views)))
        starts = np.clip(
            starts + np.random.randint(-radius, radius + 1, size=views),
            0, maximum,
        )
    return np.stack([audio[start:start + window] for start in starts]).astype(
        np.float32, copy=False
    )


class AudioBankDataset(Dataset):
    def __init__(
        self, frame: pd.DataFrame, window: int, views: int,
        stochastic: bool, augment: bool,
    ) -> None:
        self.frame = frame.reset_index(drop=True)
        self.window = int(window)
        self.views = int(views)
        self.stochastic = bool(stochastic)
        self.augment = bool(augment)

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int):
        row = self.frame.iloc[index]
        audio = load_audio(Path(row.PATH))
        windows = _repeat_or_crop(
            audio, self.window, self.views, self.stochastic
        )
        if self.augment:
            # Mild channel-independent augmentation. Dedicated telephone and
            # codec copies already provide the stronger domain perturbations.
            if np.random.random() < .35:
                snr = np.random.uniform(25, 45)
                rms = np.sqrt(np.mean(windows.astype(np.float64) ** 2, axis=1))
                noise = np.random.standard_normal(windows.shape).astype(np.float32)
                noise_rms = np.sqrt(np.mean(noise.astype(np.float64) ** 2, axis=1))
                scale = rms / (10 ** (snr / 20) * noise_rms + 1e-8)
                windows = windows + scale[:, None].astype(np.float32) * noise
            if np.random.random() < .5:
                windows *= np.float32(np.random.uniform(.5, 1.5))
        targets = np.asarray([
            0 if pd.isna(row.VOICE_FAKE) else row.VOICE_FAKE,
            0 if pd.isna(row.MUSIC_FAKE) else row.MUSIC_FAKE,
            row.FILE_FAKE,
        ], dtype=np.float32)
        presence = np.asarray(
            [row.VOICE_PRESENT, row.MUSIC_PRESENT], dtype=np.float32
        )
        return windows, targets, presence, index


def sampling_weights(frame: pd.DataFrame) -> np.ndarray:
    keys = []
    for row in frame.fillna(0).itertuples(index=False):
        keys.append((
            str(row.DATASET), int(row.VOICE_PRESENT), int(row.MUSIC_PRESENT),
            int(row.VOICE_FAKE), int(row.MUSIC_FAKE),
        ))
    counts = Counter(keys)
    strata_per_dataset = Counter(key[0] for key in counts)
    weights = np.asarray([
        1 / (counts[key] * strata_per_dataset[key[0]]) for key in keys
    ], dtype=np.float64)
    return weights / weights.mean()


def load_spectra(model_dir: Path, device: torch.device) -> torch.nn.Module:
    spec = importlib.util.spec_from_file_location(
        "wpt_vendored_spectra", model_dir / "model.py"
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot import Spectra-AASIST from {model_dir}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    model = module.SpectraAASIST()
    missing, unexpected = model.load_state_dict(
        load_file(model_dir / "model.safetensors"), strict=False
    )
    if missing or unexpected:
        raise RuntimeError(
            f"Spectra checkpoint mismatch: {missing[:5]} {unexpected[:5]}"
        )
    return model.to(device)


def preemphasis(waveforms: torch.Tensor, coefficient: float = .97) -> torch.Tensor:
    result = waveforms.clone()
    result[..., 1:] = waveforms[..., 1:] - coefficient * waveforms[..., :-1]
    return result


def finite_eer(labels, scores) -> float:
    labels = np.asarray(labels, dtype=np.int64)
    return (
        official_eer(labels, scores)
        if len(np.unique(labels)) == 2 else float("nan")
    )


@torch.inference_mode()
def evaluate(
    model: WPTSpectraMultitask,
    loaders: list[tuple[str, DataLoader]],
    device: torch.device,
) -> tuple[pd.DataFrame, pd.DataFrame, float]:
    model.eval()
    records, predictions = [], []
    for name, loader in loaders:
        logits, targets, presence, indices = [], [], [], []
        for windows, target, component_presence, index in loader:
            windows = preemphasis(windows.to(device, non_blocking=True))
            with torch.autocast(
                device_type=device.type, dtype=torch.bfloat16,
                enabled=device.type == "cuda",
            ):
                output = model(windows)
            logits.append(output.float().cpu())
            targets.append(target)
            presence.append(component_presence)
            indices.append(index)
        logits = torch.cat(logits).numpy()
        probability = 1 / (1 + np.exp(-np.clip(logits, -30, 30)))
        targets = torch.cat(targets).numpy()
        presence = torch.cat(presence).numpy()
        indices = torch.cat(indices).numpy()
        voice = presence[:, 0].astype(bool)
        music = presence[:, 1].astype(bool)
        eers = (
            finite_eer(targets[voice, 0], probability[voice, 0]),
            finite_eer(targets[music, 1], probability[music, 1]),
            finite_eer(targets[:, 2], probability[:, 2]),
        )
        weights = np.asarray([.2, .3, .5])
        available = np.isfinite(eers)
        normalized_ads = float(
            np.sum(weights[available] * (1 - np.asarray(eers)[available]))
            / weights[available].sum()
        )
        records.append({
            "DATASET": name, "N": len(targets),
            "VOICE_EER": eers[0], "MUSIC_EER": eers[1],
            "FILE_EER": eers[2], "NORMALIZED_ADS": normalized_ads,
        })
        frame = loader.dataset.frame.iloc[indices]
        predictions.append(pd.DataFrame({
            "DATASET": name, "ID": frame.ID.to_numpy(),
            "VOICE_FAKE_PROB": probability[:, 0],
            "MUSIC_FAKE_PROB": probability[:, 1],
            "FILE_FAKE_PROB": probability[:, 2],
        }))
    metrics = pd.DataFrame(records)
    score = .5 * metrics.NORMALIZED_ADS.mean() + .5 * metrics.NORMALIZED_ADS.min()
    return metrics, pd.concat(predictions, ignore_index=True), float(score)


def worker_seed(worker_id: int) -> None:
    seed = torch.initial_seed() % (2 ** 32)
    np.random.seed(seed + worker_id)
    random.seed(seed + worker_id)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-dir", type=Path,
        default=ROOT / "models/external/spectra_aasist",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--train-datasets", nargs="+", default=list(TRAIN_DEFAULT))
    parser.add_argument("--dev-datasets", nargs="+", default=list(DEV_DEFAULT))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--window", type=int, default=64_600)
    parser.add_argument("--views", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--eval-batch-size", type=int, default=12)
    parser.add_argument("--num-workers", type=int, default=6)
    parser.add_argument("--samples-per-epoch", type=int, default=12_000)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--prompt-lr", type=float, default=2e-3)
    parser.add_argument("--head-lr", type=float, default=1e-3)
    parser.add_argument("--backend-lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--voice-weight", type=float, default=.20)
    parser.add_argument("--music-weight", type=float, default=.35)
    parser.add_argument("--file-weight", type=float, default=.45)
    parser.add_argument("--ranking-weight", type=float, default=0.0)
    parser.add_argument(
        "--selection-axis", choices=("overall", "music"), default="overall",
        help="Early-stop on all ADS tasks or Music mean-plus-worst EER.",
    )
    parser.add_argument("--seed", type=int, default=20260905)
    args = parser.parse_args()

    for name in args.train_datasets:
        assert_no_locked_eval_leakage(
            truth_path(name), ROOT / "configs/data_partitions.yaml"
        )
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    train_frame = pd.concat([
        load_frame(name, "train") for name in args.train_datasets
    ], ignore_index=True)
    train_dataset = AudioBankDataset(
        train_frame, args.window, args.views, stochastic=True, augment=True
    )
    weights = sampling_weights(train_frame)
    sampler = WeightedRandomSampler(
        torch.from_numpy(weights), num_samples=args.samples_per_epoch,
        replacement=True, generator=torch.Generator().manual_seed(args.seed),
    )
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, sampler=sampler,
        num_workers=args.num_workers, pin_memory=True,
        persistent_workers=args.num_workers > 0, worker_init_fn=worker_seed,
    )
    dev_loaders = []
    for name in args.dev_datasets:
        dataset = AudioBankDataset(
            load_frame(name, "dev"), args.window, args.views,
            stochastic=False, augment=False,
        )
        loader = DataLoader(
            dataset, batch_size=args.eval_batch_size, shuffle=False,
            num_workers=args.num_workers, pin_memory=True,
            persistent_workers=False, worker_init_fn=worker_seed,
        )
        dev_loaders.append((name, loader))

    device = torch.device(args.device)
    base = load_spectra(args.model_dir, device)
    model = WPTSpectraMultitask(base).to(device)
    prompt_parameters = list(model.prompt_encoder.prompts.parameters())
    head_parameters = list(model.task_head.parameters())
    excluded = {id(value) for value in [*prompt_parameters, *head_parameters]}
    backend_parameters = [
        value for value in model.parameters()
        if value.requires_grad and id(value) not in excluded
    ]
    optimizer = torch.optim.AdamW([
        {"params": prompt_parameters, "lr": args.prompt_lr},
        {"params": head_parameters, "lr": args.head_lr},
        {"params": backend_parameters, "lr": args.backend_lr},
    ], weight_decay=args.weight_decay)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    history = []
    best_state, best_metrics, best_predictions = None, None, None
    best_score, best_epoch, stale = -float("inf"), -1, 0
    for epoch in range(args.epochs):
        model.train()
        running = []
        for step, (windows, target, presence, _index) in enumerate(train_loader):
            windows = preemphasis(windows.to(device, non_blocking=True))
            target = target.to(device, non_blocking=True)
            presence = presence.to(device, non_blocking=True)
            sample_weight = torch.ones(len(windows), device=device)
            with torch.autocast(
                device_type=device.type, dtype=torch.bfloat16,
                enabled=device.type == "cuda",
            ):
                logits = model(windows)
                loss, terms = component_loss(
                    logits, target, presence, sample_weight,
                    (args.voice_weight, args.music_weight, args.file_weight),
                )
            # Probability-space OR is the exact competition relationship, but
            # probability BCE is deliberately unsupported by CUDA autocast.
            # Compute this small auxiliary term in FP32 while leaving the main
            # forward and component BCEs in BF16.
            component_or = 1 - (
                1 - logits[:, 0].float().sigmoid()
            ) * (1 - logits[:, 1].float().sigmoid())
            consistency = torch.nn.functional.binary_cross_entropy(
                logits[:, 2].float().sigmoid(), component_or.detach()
            )
            ranking = component_ranking_loss(
                logits.float(), target.float(), presence
            )
            loss = (
                loss.float() + .05 * consistency
                + args.ranking_weight * ranking
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [value for value in model.parameters() if value.requires_grad], 3.0
            )
            optimizer.step()
            running.append(float(loss.detach()))
            if step and step % 100 == 0:
                print(json.dumps({
                    "epoch": epoch, "step": step,
                    "loss": float(np.mean(running[-100:])),
                }), flush=True)

        metrics, predictions, overall_score = evaluate(model, dev_loaders, device)
        if args.selection_axis == "music":
            music_eer = metrics.MUSIC_EER.dropna().to_numpy(np.float64)
            score = float(1 - .5 * (music_eer.mean() + music_eer.max()))
        else:
            score = overall_score
        record = {
            "EPOCH": epoch, "LOSS": float(np.mean(running)),
            "SELECTION": score,
            "OVERALL_SELECTION": overall_score,
            "MEAN_ADS": float(metrics.NORMALIZED_ADS.mean()),
            "WORST_ADS": float(metrics.NORMALIZED_ADS.min()),
        }
        history.append(record)
        print(json.dumps(record), flush=True)
        print(metrics.round(5).to_string(index=False), flush=True)
        if score > best_score + 1e-5:
            best_score, best_epoch = score, epoch
            best_state = copy.deepcopy(model.trainable_state_dict())
            best_metrics, best_predictions = metrics.copy(), predictions.copy()
            stale = 0
        else:
            stale += 1
            if stale >= args.patience:
                break

    if best_state is None:
        raise RuntimeError("training did not produce a checkpoint")
    checkpoint = {
        "model_type": "wpt_spectra_multitask",
        "state": best_state,
        "config": {
            "prompt_tokens": model.prompt_encoder.prompts.prompt_tokens,
            "wavelet_tokens": model.prompt_encoder.prompts.wavelet_tokens,
            "temperature": model.temperature,
            "window": args.window, "views": args.views,
            "task_weights": [
                args.voice_weight, args.music_weight, args.file_weight,
            ],
            "ranking_weight": args.ranking_weight,
            "selection_axis": args.selection_axis,
        },
        "train_datasets": list(args.train_datasets),
        "dev_datasets": list(args.dev_datasets),
        "best_epoch": best_epoch, "selection": best_score,
        "seed": args.seed,
    }
    torch.save(checkpoint, args.output_dir / "wpt_spectra_multitask.pt")
    pd.DataFrame(history).to_csv(args.output_dir / "history.csv", index=False)
    best_metrics.to_csv(args.output_dir / "dev_metrics.csv", index=False)
    best_predictions.to_csv(args.output_dir / "dev_predictions.csv", index=False)
    (args.output_dir / "summary.json").write_text(json.dumps({
        "best_epoch": best_epoch, "selection": best_score,
        "train_examples": len(train_frame),
        "trainable_parameters": sum(
            value.numel() for value in model.parameters() if value.requires_grad
        ),
    }, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
