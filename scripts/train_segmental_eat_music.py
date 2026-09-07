#!/usr/bin/env python3
"""Train an EAT segment/self-similarity music deepfake expert."""

from __future__ import annotations

import argparse
import copy
import json
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from data_guard import assert_no_locked_eval_leakage  # noqa: E402
from segmental_eat_music import SegmentalEatMusicHead  # noqa: E402
from train_hierarchical_eat_music import (  # noqa: E402
    BanksDataset, DEV_DEFAULT, TRAIN_DEFAULT, load_bank, normalization,
    predict, sampling_weights, truth_path,
)


def drop_views(mask: torch.Tensor, probability: float) -> torch.Tensor:
    """Randomly hide segments while always retaining the first valid one."""
    if probability <= 0:
        return mask
    keep = mask & (torch.rand(mask.shape, device=mask.device) >= probability)
    empty = ~keep.any(dim=1)
    if empty.any():
        first = mask.float().argmax(dim=1)
        keep[empty, first[empty]] = True
    return keep


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stats-root", type=Path,
        default=ROOT / "output/eat_segmental_stats_v1",
    )
    parser.add_argument("--train-datasets", nargs="+", default=list(TRAIN_DEFAULT))
    parser.add_argument("--dev-datasets", nargs="+", default=list(DEV_DEFAULT))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=20260904)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--samples-per-epoch", type=int, default=16000)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--layer-depth", type=int, default=1)
    parser.add_argument("--segment-depth", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=.20)
    parser.add_argument("--view-dropout", type=float, default=.20)
    parser.add_argument(
        "--branch-mode", choices=("content", "structure", "both"), default="both"
    )
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    args = parser.parse_args()
    if not 0 <= args.view_dropout < 1:
        parser.error("view dropout must lie in [0, 1)")
    if args.epochs <= 0 or args.patience <= 0 or args.samples_per_epoch <= 0:
        parser.error("training counts must be positive")
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
    if len({bank.values.shape[1:] for bank in train + dev}) != 1:
        raise ValueError("all segmental EAT banks must share one feature shape")
    mean, std = normalization(train)
    device = torch.device(args.device)
    model = SegmentalEatMusicHead(
        mean, std,
        max_views=train[0].values.shape[1],
        heads=args.heads,
        layer_depth=args.layer_depth,
        segment_depth=args.segment_depth,
        dropout=args.dropout,
        branch_mode=args.branch_mode,
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
        optimizer, T_max=args.epochs, eta_min=args.learning_rate * .05
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
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
            scaler.step(optimizer)
            scaler.update()
            losses.append(float(loss.detach()))
        scheduler.step()
        metrics, _, selection = predict(
            model, dev, device, max(args.batch_size, 128)
        )
        record = {
            "EPOCH": epoch,
            "LOSS": float(np.mean(losses)),
            "SELECTION": selection,
            "MEAN_EER": float(metrics.MUSIC_EER.mean()),
            "WORST_EER": float(metrics.MUSIC_EER.max()),
        }
        history.append(record)
        if selection > best_selection + 1e-5:
            best_selection = selection
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            stale = 0
        else:
            stale += 1
            if stale >= args.patience:
                break
        print(json.dumps(record), flush=True)
    if best_state is None:
        raise RuntimeError("training produced no checkpoint")
    model.load_state_dict(best_state)
    metrics, predictions, selection = predict(
        model, dev, device, max(args.batch_size, 128)
    )
    projection_path = sorted(
        (args.stats_root / train[0].name).glob("shard_*.npz")
    )[0]
    with np.load(projection_path, allow_pickle=False) as state:
        projection = state["projection"].copy()
    checkpoint = {
        "model_type": "segmental_eat_music",
        "model": {key: value.cpu() for key, value in best_state.items()},
        "config": {
            "max_views": train[0].values.shape[1],
            "heads": args.heads,
            "layer_depth": args.layer_depth,
            "segment_depth": args.segment_depth,
            "dropout": args.dropout,
            "branch_mode": args.branch_mode,
        },
        "projection": projection,
        "best_epoch": best_epoch,
        "selection": selection,
        "seed": args.seed,
    }
    torch.save(checkpoint, args.output_dir / "segmental_eat_music.pt")
    pd.DataFrame(history).to_csv(args.output_dir / "history.csv", index=False)
    metrics.to_csv(args.output_dir / "dev_metrics.csv", index=False)
    predictions.to_csv(args.output_dir / "dev_predictions.csv", index=False)
    summary = {
        "best_selection": selection,
        "best_epoch": best_epoch,
        "train_examples": len(dataset),
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "branch_mode": args.branch_mode,
        "max_views": train[0].values.shape[1],
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(metrics.to_string(index=False))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
