#!/usr/bin/env python3
"""Score frozen temporal MERT statistics with a trained linear music head."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from train_mert_temporal_music_probe import (  # noqa: E402
    load_statistics,
    projected_features,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--statistics", type=Path, required=True)
    parser.add_argument("--head", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args()
    state = np.load(args.head, allow_pickle=False)
    datasets, ids, statistics = load_statistics(args.statistics)
    device = torch.device(args.device)
    projection = torch.from_numpy(
        np.asarray(state["projection"], dtype=np.float32)
    ).to(device)
    features = projected_features(
        statistics, projection, device, args.batch_size,
        str(state["feature_mode"]),
    )
    features = np.clip(
        (features - np.asarray(state["mean"], dtype=np.float32))
        / np.asarray(state["std"], dtype=np.float32), -8, 8,
    )
    margin = (
        features @ np.asarray(state["weight"], dtype=np.float32)
        + float(state["bias"])
    )
    probability = np.exp(-np.logaddexp(0.0, -margin.astype(np.float64)))
    result = pd.DataFrame({
        "DATASET": datasets, "ID": ids,
        "MERT_TEMPORAL_MUSIC_PROB": probability,
    })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)
    print(f"Wrote {len(result)} temporal MERT scores to {args.output}")


if __name__ == "__main__":
    main()
