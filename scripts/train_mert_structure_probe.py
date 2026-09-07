#!/usr/bin/env python3
"""Train a long-range within-track structure probe on frozen MERT statistics.

The feature deliberately discards most absolute timbre information.  It uses
changes and self-similarity among the beginning, middle, and final thirds of a
track, so it complements short-window decoder-artifact experts.
"""

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
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from data_guard import assert_no_locked_eval_leakage  # noqa: E402
from train_dual_domain_head import truth_path  # noqa: E402
from train_long_horizon_music_probe import (  # noqa: E402
    DEV_DEFAULT, TRAIN_DEFAULT, component_partners, generator_environments,
    sample_weights,
)
from train_mert_temporal_music_probe import (  # noqa: E402
    AUDIT_DEFAULT, evaluate, load_metadata, load_statistics,
)


DEFAULT_GROUPS = ((0, 1, 2, 3), (4, 5, 6, 7, 8), (9, 10, 11, 12))


def structure_features(
    statistics: np.ndarray,
    projection: torch.Tensor,
    device: torch.device,
    batch_size: int = 256,
    include_track_mean: bool = False,
) -> np.ndarray:
    """Return ordered change, curvature, and self-similarity features."""
    if statistics.shape[1:4] != (3, 13, 2):
        raise ValueError(f"unexpected MERT statistics shape: {statistics.shape}")
    if projection.shape[0] != statistics.shape[-1]:
        raise ValueError("projection and MERT feature dimensions differ")
    blocks = []
    for offset in range(0, len(statistics), batch_size):
        values = torch.from_numpy(
            statistics[offset:offset + batch_size].copy()
        ).to(device=device, dtype=torch.float32)
        values = F.layer_norm(values, (values.shape[-1],)) @ projection
        # Pool fixed shallow/middle/late MERT layer families.  This lowers the
        # dimensionality without fitting a label-dependent layer selector.
        grouped = torch.stack([
            values[:, :, list(group), :, :].mean(dim=2)
            for group in DEFAULT_GROUPS
        ], dim=2)  # [batch, thirds, layer_group, statistic, width]
        start, middle, end = grouped.unbind(dim=1)
        first = middle - start
        second = end - middle
        curvature = second - first
        dispersion = grouped.std(dim=1, unbiased=False)
        dynamic = torch.cat((first, second, curvature, dispersion), dim=-1)

        normalized = F.normalize(grouped, dim=-1, eps=1e-6)
        cosine = torch.stack((
            (normalized[:, 0] * normalized[:, 1]).sum(dim=-1),
            (normalized[:, 1] * normalized[:, 2]).sum(dim=-1),
            (normalized[:, 0] * normalized[:, 2]).sum(dim=-1),
        ), dim=-1)
        scale = float(projection.shape[1]) ** .5
        distance = torch.stack((
            first.norm(dim=-1) / scale,
            second.norm(dim=-1) / scale,
            (end - start).norm(dim=-1) / scale,
            curvature.norm(dim=-1) / scale,
        ), dim=-1)
        features = [dynamic.flatten(1), cosine.flatten(1), distance.flatten(1)]
        if include_track_mean:
            features.append(grouped.mean(dim=1).flatten(1))
        blocks.append(torch.cat(features, dim=1).cpu().numpy())
    return np.concatenate(blocks).astype(np.float32, copy=False)


class StructureProbe(nn.Module):
    def __init__(self, dimension: int, hidden: int, dropout: float) -> None:
        super().__init__()
        self.network = (
            nn.Linear(dimension, 1)
            if hidden <= 0 else nn.Sequential(
                nn.Linear(dimension, hidden), nn.LayerNorm(hidden), nn.GELU(),
                nn.Dropout(dropout), nn.Linear(hidden, 1),
            )
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.network(values).squeeze(-1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--statistics", type=Path, default=ROOT / "output/mert_temporal_v2"
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--train-datasets", nargs="+", default=list(TRAIN_DEFAULT))
    parser.add_argument("--dev-datasets", nargs="+", default=list(DEV_DEFAULT))
    parser.add_argument("--audit-datasets", nargs="+", default=list(AUDIT_DEFAULT))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--projection-width", type=int, default=64)
    parser.add_argument("--projection-seed", type=int, default=20260905)
    parser.add_argument("--include-track-mean", action="store_true")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=.15)
    parser.add_argument("--epochs", type=int, default=600)
    parser.add_argument("--eval-every", type=int, default=5)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--learning-rate", type=float, default=2e-3)
    parser.add_argument("--weight-decay", type=float, default=2e-2)
    parser.add_argument("--consistency-weight", type=float, default=.02)
    parser.add_argument("--group-dro-temperature", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=20260905)
    args = parser.parse_args()
    if args.projection_width <= 0 or args.hidden < 0:
        parser.error("projection width must be positive and hidden non-negative")
    for name in args.train_datasets:
        assert_no_locked_eval_leakage(
            truth_path(name), ROOT / "configs/data_partitions.yaml"
        )

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(args.seed)
    device = torch.device(args.device)
    datasets, ids, statistics = load_statistics(args.statistics)
    frame = load_metadata(datasets, ids)
    expected = set(args.train_datasets + args.dev_datasets + args.audit_datasets)
    missing = expected.difference(frame.DATASET.unique())
    if missing:
        raise ValueError(f"statistics miss configured datasets: {sorted(missing)}")
    generator = torch.Generator(device="cpu").manual_seed(args.projection_seed)
    projection = (
        torch.randn(768, args.projection_width, generator=generator)
        / np.sqrt(768)
    ).to(device)
    features = structure_features(
        statistics, projection, device, args.batch_size, args.include_track_mean
    )
    del statistics
    train_mask = frame.DATASET.isin(args.train_datasets).to_numpy(copy=True)
    train_frame = frame.loc[train_mask].reset_index(drop=True)
    train_features = features[train_mask]
    mean = train_features.mean(axis=0, dtype=np.float64).astype(np.float32)
    std = train_features.std(axis=0, dtype=np.float64).clip(1e-4).astype(np.float32)
    features = np.clip((features - mean) / std, -8, 8).astype(np.float32)
    train_features = features[train_mask]

    x = torch.from_numpy(train_features).to(device)
    y = torch.from_numpy(
        train_frame.MUSIC_FAKE.astype(np.float32).to_numpy(copy=True)
    ).to(device)
    weights = torch.from_numpy(sample_weights(train_frame)).to(device)
    pair_values = component_partners(train_frame)
    pairs = torch.from_numpy(pair_values).to(device)
    environment_values = generator_environments(train_frame)
    environments = [
        (
            name,
            torch.from_numpy(positive.copy()).to(device),
            torch.from_numpy(negative.copy()).to(device),
        )
        for name, positive, negative in environment_values
    ]
    evaluation_x = torch.from_numpy(features).to(device)
    model = StructureProbe(features.shape[1], args.hidden, args.dropout).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    print(json.dumps({
        "examples": len(frame), "train": len(train_frame),
        "features": features.shape[1], "pairs": len(pairs),
        "environments": len(environments),
    }), flush=True)

    best_selection, best_epoch, best_state, stale = -np.inf, -1, None, 0
    history = []
    for epoch in range(args.epochs + 1):
        model.train()
        logits = model(x)
        point = F.binary_cross_entropy_with_logits(logits, y, reduction="none")
        if args.group_dro_temperature > 0:
            losses = torch.stack([
                .5 * point[positive].mean() + .5 * point[negative].mean()
                for _, positive, negative in environments
            ])
            temperature = float(args.group_dro_temperature)
            base = temperature * (
                torch.logsumexp(losses / temperature, dim=0)
                - np.log(len(environments))
            )
        else:
            base = (point * weights).mean()
        consistency = logits.new_zeros(())
        if len(pairs) and args.consistency_weight > 0:
            consistency = F.smooth_l1_loss(
                logits[pairs[:, 0]], logits[pairs[:, 1]]
            )
        loss = base + args.consistency_weight * consistency
        optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step()
        if epoch % args.eval_every:
            continue
        model.eval()
        with torch.inference_mode():
            scores = model(evaluation_x).sigmoid().cpu().numpy()
        dev, selection = evaluate(frame, scores, args.dev_datasets)
        record = {
            "EPOCH": epoch, "LOSS": float(loss.detach()),
            "BASE_LOSS": float(base.detach()),
            "CONSISTENCY_LOSS": float(consistency.detach()),
            "SELECTION": selection, "MEAN_EER": float(dev.MUSIC_EER.mean()),
            "WORST_EER": float(dev.MUSIC_EER.max()),
        }
        history.append(record)
        if selection > best_selection + 1e-5:
            best_selection, best_epoch = selection, epoch
            best_state = copy.deepcopy(model.state_dict()); stale = 0
        else:
            stale += 1
            if stale >= args.patience:
                break
        if epoch % 50 == 0:
            print(record, flush=True)
    if best_state is None:
        raise RuntimeError("training produced no checkpoint")
    model.load_state_dict(best_state); model.eval()
    with torch.inference_mode():
        scores = model(evaluation_x).sigmoid().cpu().numpy()
    dev, selection = evaluate(frame, scores, args.dev_datasets)
    audit, _ = evaluate(frame, scores, args.audit_datasets)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "model": {key: value.cpu() for key, value in best_state.items()},
        "config": {
            "feature_dimension": features.shape[1], "hidden": args.hidden,
            "dropout": args.dropout, "include_track_mean": args.include_track_mean,
        },
        "projection": projection.cpu(), "mean": mean, "std": std,
        "layer_groups": DEFAULT_GROUPS, "best_epoch": best_epoch,
        "selection": selection, "seed": args.seed,
    }
    torch.save(checkpoint, args.output_dir / "mert_structure_head.pt")
    pd.DataFrame(history).to_csv(args.output_dir / "history.csv", index=False)
    dev.to_csv(args.output_dir / "dev_metrics.csv", index=False)
    audit.to_csv(args.output_dir / "audit_metrics.csv", index=False)
    pd.DataFrame({
        "DATASET": frame.DATASET, "ID": frame.ID,
        "MERT_STRUCTURE_MUSIC_PROB": scores,
    }).to_csv(args.output_dir / "predictions.csv", index=False)
    summary = {
        "best_epoch": best_epoch, "selection": selection,
        "mean_dev_eer": float(dev.MUSIC_EER.mean()),
        "worst_dev_eer": float(dev.MUSIC_EER.max()),
        "include_track_mean": args.include_track_mean,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(dev.to_string(index=False)); print(audit.to_string(index=False))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
