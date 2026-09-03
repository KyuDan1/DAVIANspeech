#!/usr/bin/env python3
"""Score frozen segment/self-similarity EAT music experts."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from evaluate_diagnostic import official_eer  # noqa: E402
from segmental_eat_music import SegmentalEatMusicHead  # noqa: E402
from train_hierarchical_eat_music import load_bank, predict  # noqa: E402


def load_model(path: Path, device: torch.device) -> SegmentalEatMusicHead:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if checkpoint.get("model_type") != "segmental_eat_music":
        raise ValueError(f"not a segmental EAT checkpoint: {path}")
    state = checkpoint["model"]
    model = SegmentalEatMusicHead(
        state["normalization_mean"], state["normalization_std"],
        **checkpoint["config"],
    )
    model.load_state_dict(state, strict=True)
    return model.to(device).eval()


def logit(values: np.ndarray) -> np.ndarray:
    values = np.clip(values, 1e-5, 1 - 1e-5)
    return np.log(values) - np.log1p(-values)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stats-root", type=Path,
        default=ROOT / "output/eat_segmental_stats_v1",
    )
    parser.add_argument("--checkpoints", type=Path, nargs="+", required=True)
    parser.add_argument("--datasets", nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args()
    device = torch.device(args.device)
    banks = [
        load_bank(args.stats_root, name, music_only=False)
        for name in args.datasets
    ]
    members, member_metrics = [], []
    for path in args.checkpoints:
        model = load_model(path, device)
        metrics, predictions, _ = predict(
            model, banks, device, args.batch_size
        )
        metrics["MEMBER"] = path.parent.name
        predictions["MEMBER"] = path.parent.name
        member_metrics.append(metrics)
        members.append(predictions)
    combined = members[0][
        ["DATASET", "ID", "MUSIC_FAKE", "MUSIC_PRESENT"]
    ].copy()
    logits = []
    for frame in members:
        aligned = combined[["DATASET", "ID"]].merge(
            frame, on=["DATASET", "ID"], validate="one_to_one"
        )
        logits.append(logit(aligned.HIERARCHICAL_EAT_MUSIC_PROB.to_numpy()))
    combined["SEGMENTAL_EAT_MUSIC_PROB"] = 1 / (
        1 + np.exp(-np.mean(logits, axis=0))
    )
    rows = []
    for dataset, block in combined.groupby("DATASET", sort=False):
        selected = block.MUSIC_PRESENT.eq(1)
        labels = block.loc[selected, "MUSIC_FAKE"]
        eer = (
            official_eer(labels, block.loc[selected, "SEGMENTAL_EAT_MUSIC_PROB"])
            if labels.nunique() == 2 else np.nan
        )
        rows.append({
            "DATASET": dataset, "MEMBER": "ensemble",
            "N": int(selected.sum()), "MUSIC_EER": eer,
        })
    args.output_dir.mkdir(parents=True, exist_ok=True)
    pd.concat(member_metrics, ignore_index=True).to_csv(
        args.output_dir / "member_metrics.csv", index=False
    )
    combined.to_csv(args.output_dir / "predictions.csv", index=False)
    metrics = pd.DataFrame(rows)
    metrics.to_csv(args.output_dir / "metrics.csv", index=False)
    print(metrics.to_string(index=False))


if __name__ == "__main__":
    main()
