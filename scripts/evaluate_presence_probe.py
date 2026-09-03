#!/usr/bin/env python3
"""Evaluate a frozen EAT presence probe on untouched audit banks."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy.special import expit

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from experiment_presence_probe import auc, load_features, truth_path  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--stats-root", type=Path,
                        default=ROOT / "output/dual_domain_stats_v1")
    parser.add_argument("--banks", nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    checkpoint = np.load(args.checkpoint)
    rows = []
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name in args.banks:
        ids, features = load_features(args.stats_root, name)
        features = np.clip(
            (features - checkpoint["mean"]) / checkpoint["std"], -8, 8
        )
        truth = pd.read_csv(truth_path(name), dtype={"ID": str}).set_index("ID").loc[ids]
        output = pd.DataFrame({"ID": ids})
        record = {"DATASET": name}
        for component in ("VOICE", "MUSIC"):
            decision = (
                features @ checkpoint[f"{component.lower()}_coefficient"].reshape(-1)
                + checkpoint[f"{component.lower()}_intercept"].item()
            )
            probability = expit(decision)
            output[f"PROBE_{component}_PRESENT_PROB"] = probability
            record[f"{component}_AUC"] = auc(
                truth[f"{component}_PRESENT"].to_numpy(np.int64), probability
            )
        output.to_csv(args.output_dir / f"{name}.csv", index=False)
        rows.append(record)
    metrics = pd.DataFrame(rows)
    metrics["CPS"] = 0.5 * (metrics.VOICE_AUC + metrics.MUSIC_AUC)
    metrics.to_csv(args.output_dir / "metrics.csv", index=False)
    print(metrics.to_string(index=False))


if __name__ == "__main__":
    main()
