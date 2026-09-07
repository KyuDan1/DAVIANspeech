#!/usr/bin/env python3
"""Train a source-balanced all-layer EAT music deepfake head."""

from __future__ import annotations

import argparse
import copy
import json
import random
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from data_guard import assert_no_locked_eval_leakage  # noqa: E402
from evaluate_diagnostic import official_eer  # noqa: E402
from hierarchical_eat_music import HierarchicalEatMusicHead  # noqa: E402


TRAIN_DEFAULT = (
    "external_mixed_train_v1",
    "mixed_devvoice_train_v1",
    "mixed_fmc_music_train_v1",
    "mixfake_music_train_v1",
    "telephone_mixed_train_v1",
    "temporal_mixed_train_v2",
    "channel_invariant_factorial_train_v1",
)
DEV_DEFAULT = (
    "mixfake_music_dev_v1",
    "external_mixed_v1",
    "source_disjoint_mixed_v1",
    "source_disjoint_mixed_equal_v1",
    "source_disjoint_music_v1",
    "factorial_eval_1200_v2_dev",
    "telephone_mixed_dev_v1",
)
TRUTH_OVERRIDES = {
    "factorial_eval_1200_v2_dev":
        ROOT / "data/eval/factorial_eval_1200_v2/truth_dev.csv",
    "factorial_eval_1200_v2_holdout":
        ROOT / "data/eval/factorial_eval_1200_v2/truth_holdout.csv",
}


def truth_path(name: str) -> Path:
    return TRUTH_OVERRIDES.get(name, ROOT / "data/eval" / name / "truth.csv")


@dataclass
class Bank:
    name: str
    ids: np.ndarray
    values: np.ndarray
    mask: np.ndarray
    targets: np.ndarray
    music_present: np.ndarray
    truth: pd.DataFrame


def load_bank(root: Path, name: str, music_only: bool = True) -> Bank:
    ids, values, masks = [], [], []
    projection = None
    paths = sorted((root / name).glob("shard_*.npz"))
    if not paths:
        raise FileNotFoundError(f"no hierarchical EAT shards for {name}")
    for path in paths:
        with np.load(path, allow_pickle=False) as shard:
            ids.append(shard["ids"].astype(str))
            values.append(shard["statistics"])
            masks.append(shard["view_mask"].astype(bool))
            current = shard["projection"].astype(np.float32)
            if projection is None:
                projection = current
            elif not np.array_equal(projection, current):
                raise ValueError(f"projection mismatch in {name}")
    ids = np.concatenate(ids)
    values = np.concatenate(values)
    masks = np.concatenate(masks)
    if len(set(ids)) != len(ids):
        raise ValueError(f"duplicate IDs in {name}")
    truth = pd.read_csv(truth_path(name), dtype={"ID": str}).set_index("ID")
    missing = set(ids).difference(truth.index)
    if missing:
        raise ValueError(f"missing truth rows for {name}: {sorted(missing)[:5]}")
    truth = truth.loc[ids].reset_index()
    present = truth.MUSIC_PRESENT.eq(1) & truth.MUSIC_FAKE.notna()
    selected = present if music_only else pd.Series(True, index=truth.index)
    indices = np.flatnonzero(selected.to_numpy())
    return Bank(
        name=name,
        ids=ids[indices],
        values=values[indices],
        mask=masks[indices],
        targets=truth.loc[selected, "MUSIC_FAKE"].fillna(0).to_numpy(np.float32),
        music_present=present[selected].to_numpy(bool),
        truth=truth.loc[selected].reset_index(drop=True),
    )


class BanksDataset(Dataset):
    def __init__(self, banks: list[Bank]) -> None:
        self.banks = banks
        self.offsets = np.cumsum([0] + [len(bank.ids) for bank in banks])

    def __len__(self) -> int:
        return int(self.offsets[-1])

    def __getitem__(self, index: int):
        bank_number = int(np.searchsorted(self.offsets, index, side="right") - 1)
        local = int(index - self.offsets[bank_number])
        bank = self.banks[bank_number]
        return bank.values[local], bank.mask[local], bank.targets[local]


def normalization(banks: list[Bank]) -> tuple[np.ndarray, np.ndarray]:
    total = total_square = None
    count = 0
    for bank in banks:
        for offset in range(0, len(bank.ids), 64):
            values = bank.values[offset:offset + 64].astype(np.float32)
            selected = values[bank.mask[offset:offset + 64]]
            current = selected.sum(axis=0, dtype=np.float64)
            square = np.square(selected).sum(axis=0, dtype=np.float64)
            total = current if total is None else total + current
            total_square = square if total_square is None else total_square + square
            count += len(selected)
    mean = total / count
    variance = np.maximum(total_square / count - np.square(mean), 1e-5)
    return mean.astype(np.float32), np.sqrt(variance).astype(np.float32)


def generator_group(row: pd.Series) -> str:
    if int(row.MUSIC_FAKE) == 1:
        for column in ("MUSIC_GENERATOR", "GENERATOR"):
            value = row.get(column)
            if pd.notna(value) and str(value).strip():
                return "fake:" + str(value).strip().lower()
        return "fake:" + str(row.get("SOURCE", row.get("MUSIC_SOURCE_BANK", "unknown")))
    for column in ("MUSIC_SOURCE_BANK", "SOURCE"):
        value = row.get(column)
        if pd.notna(value) and str(value).strip():
            return "real:" + str(value).strip().lower()
    return "real:unknown"


def sampling_weights(banks: list[Bank]) -> np.ndarray:
    frames = []
    for bank_index, bank in enumerate(banks):
        frame = bank.truth.copy()
        frame["_BANK"] = bank_index
        frame["_GROUP"] = frame.apply(generator_group, axis=1)
        frames.append(frame)
    frame = pd.concat(frames, ignore_index=True)
    keys = frame.apply(
        lambda row: (
            f"{row['_BANK']}|{int(row['MUSIC_FAKE'])}|{row['_GROUP']}"
        ),
        axis=1,
    )
    counts = keys.value_counts()
    weights = np.asarray([1 / counts[key] for key in keys], dtype=np.float64)
    bank_counts = frame._BANK.value_counts()
    weights *= np.asarray([1 / bank_counts[index] for index in frame._BANK])
    labels = frame.MUSIC_FAKE.astype(int).to_numpy()
    totals = np.bincount(labels, weights=weights, minlength=2)
    weights /= totals[labels].clip(1e-12)
    return (weights / weights.mean()).astype(np.float64)


def drop_views(mask: torch.Tensor, probability: float) -> torch.Tensor:
    if probability <= 0:
        return mask
    keep = mask & (torch.rand(mask.shape, device=mask.device) >= probability)
    empty = ~keep.any(dim=1)
    if empty.any():
        first = mask.float().argmax(dim=1)
        keep[empty, first[empty]] = True
    return keep


@torch.inference_mode()
def predict(
    model: HierarchicalEatMusicHead,
    banks: list[Bank],
    device: torch.device,
    batch_size: int,
) -> tuple[pd.DataFrame, pd.DataFrame, float]:
    model.eval()
    metric_rows, prediction_rows = [], []
    for bank in banks:
        scores = []
        for offset in range(0, len(bank.ids), batch_size):
            values = torch.from_numpy(bank.values[offset:offset + batch_size]).to(
                device=device, dtype=torch.float32
            )
            mask = torch.from_numpy(bank.mask[offset:offset + batch_size]).to(device)
            with torch.autocast(
                device_type=device.type, dtype=torch.bfloat16,
                enabled=device.type == "cuda",
            ):
                logit = model(values, mask)
            scores.append(logit.float().sigmoid().cpu().numpy())
        scores = np.concatenate(scores)
        selected = bank.music_present
        eer = float(official_eer(bank.targets[selected], scores[selected]))
        metric_rows.append({
            "DATASET": bank.name, "N": int(selected.sum()), "MUSIC_EER": eer
        })
        prediction_rows.append(pd.DataFrame({
            "DATASET": bank.name,
            "ID": bank.ids,
            "MUSIC_FAKE": bank.targets.astype(int),
            "MUSIC_PRESENT": bank.music_present.astype(int),
            "HIERARCHICAL_EAT_MUSIC_PROB": scores,
        }))
    metrics = pd.DataFrame(metric_rows)
    quality = 1 - metrics.MUSIC_EER
    selection = float(0.5 * quality.mean() + 0.5 * quality.min())
    return metrics, pd.concat(prediction_rows, ignore_index=True), selection


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stats-root", type=Path,
        default=ROOT / "output/eat_hierarchical_stats_v1",
    )
    parser.add_argument("--train-datasets", nargs="+", default=list(TRAIN_DEFAULT))
    parser.add_argument("--dev-datasets", nargs="+", default=list(DEV_DEFAULT))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=20260904)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--samples-per-epoch", type=int, default=16000)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--pool-heads", type=int, default=8)
    parser.add_argument("--depth", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.20)
    parser.add_argument("--view-dropout", type=float, default=0.15)
    parser.add_argument("--dsu-probability", type=float, default=0.5)
    parser.add_argument("--dsu-scale", type=float, default=1.0)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    args = parser.parse_args()
    if not 0 <= args.view_dropout < 1:
        parser.error("view dropout must lie in [0, 1)")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    for name in args.train_datasets:
        assert_no_locked_eval_leakage(
            truth_path(name), ROOT / "configs/data_partitions.yaml"
        )

    train = [load_bank(args.stats_root, name) for name in args.train_datasets]
    dev = [load_bank(args.stats_root, name) for name in args.dev_datasets]
    mean, std = normalization(train)
    device = torch.device(args.device)
    model = HierarchicalEatMusicHead(
        mean, std,
        max_views=train[0].values.shape[1],
        heads=args.heads,
        pool_heads=args.pool_heads,
        depth=args.depth,
        dropout=args.dropout,
        dsu_probability=args.dsu_probability,
        dsu_scale=args.dsu_scale,
    ).to(device)
    dataset = BanksDataset(train)
    weights = sampling_weights(train)
    generator = torch.Generator().manual_seed(args.seed)
    sampler = WeightedRandomSampler(
        weights, num_samples=args.samples_per_epoch,
        replacement=True, generator=generator,
    )
    loader = DataLoader(
        dataset, batch_size=args.batch_size, sampler=sampler,
        num_workers=4, pin_memory=device.type == "cuda", persistent_workers=True,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.learning_rate * 0.05
    )
    scaler = torch.amp.GradScaler(device.type, enabled=device.type == "cuda")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    best_state, best_selection, stale = None, -np.inf, 0
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for values, mask, target in loader:
            values = values.to(device=device, dtype=torch.float32, non_blocking=True)
            mask = drop_views(mask.to(device, non_blocking=True), args.view_dropout)
            target = target.to(device=device, dtype=torch.float32, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=device.type, dtype=torch.bfloat16,
                enabled=device.type == "cuda",
            ):
                logits = model(values, mask)
                loss = F.binary_cross_entropy_with_logits(logits.float(), target)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(optimizer)
            scaler.update()
            losses.append(float(loss.detach()))
        scheduler.step()
        metrics, predictions, selection = predict(
            model, dev, device, args.batch_size
        )
        history.append({
            "EPOCH": epoch,
            "LOSS": float(np.mean(losses)),
            "SELECTION": selection,
            "MEAN_MUSIC_EER": float(metrics.MUSIC_EER.mean()),
            "WORST_MUSIC_EER": float(metrics.MUSIC_EER.max()),
        })
        print(
            f"epoch={epoch:03d} loss={history[-1]['LOSS']:.5f} "
            f"selection={selection:.5f} mean_eer={metrics.MUSIC_EER.mean():.5f} "
            f"worst_eer={metrics.MUSIC_EER.max():.5f}",
            flush=True,
        )
        if selection > best_selection + 1e-6:
            best_selection = selection
            best_state = copy.deepcopy(model.state_dict())
            metrics.to_csv(args.output_dir / "dev_metrics.csv", index=False)
            predictions.to_csv(args.output_dir / "dev_predictions.csv", index=False)
            stale = 0
        else:
            stale += 1
            if stale >= args.patience:
                break
    if best_state is None:
        raise RuntimeError("training produced no checkpoint")
    model.load_state_dict(best_state)
    projection = None
    with np.load(
        args.stats_root / args.train_datasets[0] / "shard_0.npz",
        allow_pickle=False,
    ) as shard:
        projection = shard["projection"].astype(np.float32)
    checkpoint = {
        "model_type": "hierarchical_eat_music",
        "config": {
            "max_views": model.max_views,
            "heads": args.heads,
            "pool_heads": args.pool_heads,
            "depth": args.depth,
            "dropout": args.dropout,
            "dsu_probability": args.dsu_probability,
            "dsu_scale": args.dsu_scale,
        },
        "model": {name: value.cpu() for name, value in best_state.items()},
        "projection": projection,
        "train_datasets": list(args.train_datasets),
        "dev_datasets": list(args.dev_datasets),
        "selection": best_selection,
        "seed": args.seed,
    }
    torch.save(checkpoint, args.output_dir / "hierarchical_eat_music.pt")
    pd.DataFrame(history).to_csv(args.output_dir / "history.csv", index=False)
    summary = {
        "best_selection": best_selection,
        "best_epoch": int(pd.DataFrame(history).SELECTION.idxmax() + 1),
        "train_examples": len(dataset),
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
