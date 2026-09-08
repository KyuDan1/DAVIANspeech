#!/usr/bin/env python3
"""Train a raw-audio, separation-free EAT Music adapter (v55).

The training protocol is intentionally narrow: only authorised ``train``
partitions update parameters, and the source-disjoint codec-mixed v4 bank is
the sole checkpoint-selection bank.  Retrospective v5/v6/v7 banks are not
accepted by this script, preventing accidental repeated selection on them.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import transformers  # imported before the local timm shim is installed
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from data_guard import assert_no_locked_eval_leakage  # noqa: E402
from eat_detector import EatMusicDetector, _load_local_model  # noqa: E402
from eat_music_adapter import EatMusicAdapter, pairwise_ranking_loss  # noqa: E402
from evaluate_diagnostic import official_eer  # noqa: E402
from pipeline import AUDIO_EXTENSIONS, load_audio  # noqa: E402


TRAIN_DEFAULT = (
    "external_mixed_train_v1",
    "mixed_devvoice_train_v1",
    "mixed_fmc_music_train_v1",
    "mixfake_music_train_v1",
    "telephone_mixed_train_v1",
    "temporal_mixed_train_v2",
    "channel_invariant_factorial_train_v1",
)
DEV_NAME = "codec_mixed_dev_v4"


def truth_path(name: str) -> Path:
    return ROOT / "data/eval" / name / "truth.csv"


def guard_truth_path(name: str) -> Path:
    """Use a role-pure manifest when a corpus ships train+dev in truth.csv."""
    candidate = ROOT / "data/eval" / name / "truth_train.csv"
    return candidate if candidate.is_file() else truth_path(name)


def audio_index(name: str) -> dict[str, Path]:
    directory = ROOT / "data/eval" / name / "audio"
    if not directory.is_dir():
        raise FileNotFoundError(directory)
    result = {
        path.stem: path
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in AUDIO_EXTENSIONS
    }
    if not result:
        raise FileNotFoundError(f"no audio in {directory}")
    return result


def load_frame(name: str, role: str) -> pd.DataFrame:
    frame = pd.read_csv(truth_path(name), dtype={"ID": str})
    if role == "train" and "SPLIT" in frame:
        frame = frame[frame.SPLIT.eq("train")]
    frame = frame[
        frame.MUSIC_PRESENT.eq(1) & frame.MUSIC_FAKE.notna()
    ].copy()
    paths = audio_index(name)
    missing = set(frame.ID).difference(paths)
    if missing:
        raise FileNotFoundError(
            f"{name}: {len(missing)} truth IDs have no audio: {sorted(missing)[:5]}"
        )
    frame["PATH"] = frame.ID.map(paths)
    frame["DATASET"] = name
    return frame.reset_index(drop=True)


def _first(row: pd.Series, columns: tuple[str, ...]) -> str:
    for column in columns:
        value = row.get(column)
        if pd.notna(value) and str(value).strip() not in {"", "nan", "none", "0"}:
            return str(value).strip().lower()
    return "unknown"


def music_group(row: pd.Series) -> str:
    if int(row.MUSIC_FAKE):
        return "fake:" + _first(
            row, ("MUSIC_GENERATOR", "GENERATOR", "MUSIC_SOURCE_BANK", "SOURCE")
        )
    return "real:" + _first(
        row, ("MUSIC_SOURCE_BANK", "SOURCE", "MUSIC_GENERATOR")
    )


def balanced_weights(frame: pd.DataFrame) -> np.ndarray:
    """Equalise corpus, RR/RF/FR/FF, layout, channel and source/generator."""
    keys = []
    for _, row in frame.iterrows():
        keys.append((
            str(row.DATASET),
            int(row.get("VOICE_PRESENT", 0)),
            int(row.get("VOICE_FAKE", 0) if pd.notna(row.get("VOICE_FAKE")) else 0),
            int(row.MUSIC_FAKE),
            _first(row, ("MIX_MODE", "CONDITION", "AUDIO_TYPE")),
            _first(row, ("CHANNEL", "CODEC", "STRESS_VARIANT")),
            music_group(row),
        ))
    counts = Counter(keys)
    weights = np.asarray([1 / counts[key] for key in keys], dtype=np.float64)
    datasets = frame.DATASET.astype(str).to_numpy()
    for dataset in np.unique(datasets):
        selected = datasets == dataset
        weights[selected] /= weights[selected].sum()
    labels = frame.MUSIC_FAKE.astype(int).to_numpy()
    totals = np.bincount(labels, weights=weights, minlength=2)
    weights /= totals[labels].clip(1e-12)
    return weights / weights.mean()


def paired_indices(frame: pd.DataFrame) -> dict[int, tuple[int, ...]]:
    """Map channel-variant rows to same-mixture, same-label partners."""
    groups: dict[str, list[int]] = defaultdict(list)
    for index, row in frame.iterrows():
        mixture = row.get("MIXTURE_ID")
        if row.DATASET == "channel_invariant_factorial_train_v1" and pd.notna(mixture):
            groups[str(mixture)].append(index)
    result = {}
    for indices in groups.values():
        labels = {int(frame.iloc[index].MUSIC_FAKE) for index in indices}
        if len(indices) > 1 and len(labels) == 1:
            for index in indices:
                result[index] = tuple(other for other in indices if other != index)
    return result


def _crop(audio: np.ndarray, random_crop: bool) -> np.ndarray:
    samples = EatMusicDetector.SAMPLES
    if len(audio) <= samples:
        result = np.zeros(samples, dtype=np.float32)
        # Random padding shift makes silence location non-predictive.
        start = random.randrange(samples - len(audio) + 1) if random_crop else (
            samples - len(audio)
        ) // 2
        result[start : start + len(audio)] = audio
        return result
    if random_crop:
        start = random.randrange(len(audio) - samples + 1)
    else:
        start = (len(audio) - samples) // 2
    return np.asarray(audio[start : start + samples], dtype=np.float32)


def fbank(path: Path, random_crop: bool) -> torch.Tensor:
    return EatMusicDetector._fbank(_crop(load_audio(path), random_crop))


class RawMusicDataset(Dataset):
    def __init__(
        self, frame: pd.DataFrame, random_crop: bool, paired: dict[int, tuple[int, ...]]
    ) -> None:
        self.frame = frame
        self.random_crop = random_crop
        self.paired = paired

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int):
        row = self.frame.iloc[index]
        main = fbank(Path(row.PATH), self.random_crop)
        partners = self.paired.get(index)
        if partners:
            partner_index = partners[random.randrange(len(partners))]
            partner = fbank(Path(self.frame.iloc[partner_index].PATH), self.random_crop)
            paired = True
        else:
            partner = torch.zeros_like(main)
            paired = False
        return main, np.float32(row.MUSIC_FAKE), partner, paired


def deterministic_views(path: Path, count: int = 3) -> tuple[torch.Tensor, torch.Tensor]:
    audio = load_audio(path)
    samples = EatMusicDetector.SAMPLES
    if len(audio) <= samples:
        starts = [0]
    else:
        starts = sorted({int(value) for value in np.linspace(0, len(audio) - samples, count)})
    views = [
        EatMusicDetector._fbank(
            np.pad(audio[start : start + samples], (0, max(0, samples - len(audio[start : start + samples]))))
        )
        for start in starts
    ]
    result = torch.zeros(count, EatMusicDetector.FRAMES, 128)
    mask = torch.zeros(count, dtype=torch.bool)
    result[: len(views)] = torch.stack(views)
    mask[: len(views)] = True
    return result, mask


@torch.inference_mode()
def predict(
    model: EatMusicAdapter,
    frame: pd.DataFrame,
    device: torch.device,
    batch_size: int,
    workers: int,
) -> np.ndarray:
    class EvalDataset(Dataset):
        def __len__(self):
            return len(frame)

        def __getitem__(self, index):
            views, mask = deterministic_views(Path(frame.iloc[index].PATH))
            return views, mask

    loader = DataLoader(
        EvalDataset(), batch_size=batch_size, shuffle=False, num_workers=workers,
        pin_memory=device.type == "cuda", persistent_workers=workers > 0,
    )
    model.eval()
    scores = []
    for features, mask in loader:
        features = features[:, :, None].to(device, non_blocking=True)
        mask = mask.to(device, non_blocking=True)
        with torch.autocast(
            device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"
        ):
            logits = model(features, mask)
        scores.append(logits.float().sigmoid().cpu().numpy())
    return np.concatenate(scores)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--train-datasets", nargs="+", default=list(TRAIN_DEFAULT))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--adapter-blocks", nargs="+", type=int, default=[10, 11])
    parser.add_argument("--bottleneck", type=int, default=32)
    parser.add_argument("--hidden", type=int, default=192)
    parser.add_argument("--pooling-width", type=int, default=128)
    parser.add_argument("--pooling-heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=.15)
    parser.add_argument("--temperature", type=float, default=5.)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--samples-per-epoch", type=int, default=12000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--eval-batch-size", type=int, default=24)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--ranking-weight", type=float, default=.35)
    parser.add_argument("--consistency-weight", type=float, default=.03)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260904)
    args = parser.parse_args()

    if DEV_NAME in args.train_datasets:
        parser.error("codec_mixed_dev_v4 is selection-only")
    if any(name.startswith("codec_mixed_blind_v") for name in args.train_datasets):
        parser.error("retrospective blind banks are forbidden during training")
    for name in args.train_datasets:
        assert_no_locked_eval_leakage(
            guard_truth_path(name), ROOT / "configs/data_partitions.yaml"
        )
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    train = pd.concat(
        [load_frame(name, "train") for name in args.train_datasets], ignore_index=True
    )
    dev = load_frame(DEV_NAME, "development")
    # Concrete source-identity isolation, stronger than row-ID disjointness.
    source_overlap = {}
    for column in ("VOICE_SOURCE_ID", "MUSIC_SOURCE_ID"):
        if column in train and column in dev:
            overlap = set(train[column].dropna().astype(str)) & set(
                dev[column].dropna().astype(str)
            )
            source_overlap[column] = len(overlap)
            if overlap:
                raise ValueError(f"{column} leaks into {DEV_NAME}: {sorted(overlap)[:5]}")

    weights = balanced_weights(train)
    pairs = paired_indices(train)
    generator = torch.Generator().manual_seed(args.seed)
    sampler = WeightedRandomSampler(
        weights, num_samples=args.samples_per_epoch, replacement=True, generator=generator
    )
    dataset = RawMusicDataset(train, random_crop=True, paired=pairs)
    loader = DataLoader(
        dataset, batch_size=args.batch_size, sampler=sampler, num_workers=args.workers,
        pin_memory=True, persistent_workers=args.workers > 0, drop_last=True,
    )

    device = torch.device(args.device)
    eat = _load_local_model(ROOT / "models/eat-base-as2m", device)
    model = EatMusicAdapter(
        eat_model=eat,
        adapter_blocks=tuple(args.adapter_blocks),
        bottleneck=args.bottleneck,
        pooling_heads=args.pooling_heads,
        hidden=args.hidden,
        pooling_width=args.pooling_width,
        dropout=args.dropout,
        temperature=args.temperature,
    ).to(device)
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable, lr=args.learning_rate, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.learning_rate * .1
    )
    scaler = torch.amp.GradScaler(device.type, enabled=device.type == "cuda")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    best_state, best_eer, stale = None, float("inf"), 0
    history = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for features, targets, partner, has_partner in loader:
            features = features[:, None].to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            has_partner = has_partner.to(device, non_blocking=True).bool()
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"
            ):
                logits = model(features)
                bce = F.binary_cross_entropy_with_logits(logits.float(), targets)
                rank = pairwise_ranking_loss(logits.float(), targets)
                loss = (1 - args.ranking_weight) * bce + args.ranking_weight * rank
                if has_partner.any() and args.consistency_weight:
                    partner = partner.to(device, non_blocking=True)
                    pair_logits = model(
                        partner[has_partner, None]
                    )
                    consistency = F.smooth_l1_loss(
                        pair_logits.float(), logits[has_partner].float()
                    )
                    loss = loss + args.consistency_weight * consistency
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(trainable, 2.)
            scaler.step(optimizer)
            scaler.update()
            losses.append(float(loss.detach()))
        scheduler.step()

        scores = predict(model, dev, device, args.eval_batch_size, args.workers)
        eer = float(official_eer(dev.MUSIC_FAKE.astype(int).to_numpy(), scores))
        record = {"EPOCH": epoch, "LOSS": float(np.mean(losses)), "V4_MUSIC_EER": eer}
        history.append(record)
        print(json.dumps(record), flush=True)
        if eer < best_eer - 1e-6:
            best_eer = eer
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
                if not name.startswith("eat.")
            }
            pd.DataFrame({
                "ID": dev.ID, "MUSIC_FAKE": dev.MUSIC_FAKE.astype(int),
                "EAT_ADAPTER_MUSIC_PROB": scores,
            }).to_csv(args.output_dir / "dev_predictions.csv", index=False)
            stale = 0
        else:
            stale += 1
            if stale >= args.patience:
                break

    if best_state is None:
        raise RuntimeError("training produced no checkpoint")
    checkpoint = {
        "model_type": "eat_music_adapter_v55",
        "config": {
            "adapter_blocks": tuple(args.adapter_blocks),
            "bottleneck": args.bottleneck,
            "pooling_heads": args.pooling_heads,
            "hidden": args.hidden,
            "pooling_width": args.pooling_width,
            "dropout": args.dropout,
            "temperature": args.temperature,
        },
        "model": best_state,
        "train_datasets": list(args.train_datasets),
        "selection_dataset": DEV_NAME,
        "selection_music_eer": best_eer,
        "source_overlap": source_overlap,
        "seed": args.seed,
        "base_eat_sha256": sha256(ROOT / "models/eat-base-as2m/model.safetensors"),
    }
    torch.save(checkpoint, args.output_dir / "eat_music_adapter_v55.pt")
    pd.DataFrame(history).to_csv(args.output_dir / "history.csv", index=False)
    summary = {
        "best_v4_music_eer": best_eer,
        "best_epoch": int(pd.DataFrame(history).V4_MUSIC_EER.idxmin() + 1),
        "train_examples": len(train),
        "samples_per_epoch": args.samples_per_epoch,
        "trainable_parameters": sum(parameter.numel() for parameter in trainable),
        "paired_examples": len(pairs),
        "source_overlap": source_overlap,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
