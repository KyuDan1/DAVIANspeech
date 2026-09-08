#!/usr/bin/env python3
"""Score frozen predictions with the trainer's fixed development selection rule."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from train_three_stream_anchor_residual import (  # noqa: E402
    normalized_available_ads, score_frame,
)


PROBABILITY_COLUMNS = (
    "FILE_FAKE_PROB", "VOICE_FAKE_PROB", "MUSIC_FAKE_PROB",
    "VOICE_PRESENT_PROB", "MUSIC_PRESENT_PROB",
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prediction", type=Path, required=True)
    parser.add_argument(
        "--partition-config", type=Path,
        default=ROOT / "configs/data_partitions.yaml",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite {args.output_dir}")
    prediction = pd.read_csv(args.prediction, dtype={"ID": str})
    required = {"DATASET", "ID", *PROBABILITY_COLUMNS}
    if missing := required.difference(prediction):
        raise ValueError(f"prediction misses columns: {sorted(missing)}")
    if prediction[["DATASET", "ID"]].duplicated().any():
        raise ValueError("prediction has duplicate DATASET/ID rows")

    config = yaml.safe_load(args.partition_config.read_text("utf-8")) or {}
    relatives = config.get("development", [])
    if not isinstance(relatives, list) or not relatives:
        raise ValueError("partition config has no development role")
    root = args.partition_config.resolve().parent.parent
    metrics, scored_frames = [], []
    expected_keys = []
    for relative in relatives:
        truth_path = root / relative
        truth = pd.read_csv(truth_path, dtype={"ID": str})
        dataset = truth_path.parent.name
        current = prediction.loc[prediction.DATASET.eq(dataset)].copy()
        if len(current) != len(truth) or set(current.ID) != set(truth.ID):
            raise ValueError(f"prediction/truth mismatch for {dataset}")
        current = current.set_index("ID").loc[truth.ID]
        scored = truth.set_index("ID").join(current[list(PROBABILITY_COLUMNS)])
        metric = score_frame(scored)
        metrics.append({
            "DATASET": dataset, **metric,
            "NORMALIZED_AVAILABLE_ADS": normalized_available_ads(metric),
        })
        scored_frames.append(scored.reset_index(drop=True))
        expected_keys.extend((dataset, item) for item in truth.ID)
    if set(map(tuple, prediction[["DATASET", "ID"]].to_numpy())) != set(expected_keys):
        raise ValueError("prediction contains rows outside the complete development role")
    metric_frame = pd.DataFrame(metrics)
    domain = metric_frame.NORMALIZED_AVAILABLE_ADS.dropna().to_numpy(np.float64)
    pooled = float(score_frame(pd.concat(scored_frames, ignore_index=True))["ADS"])
    selection = float(0.50 * pooled + 0.25 * domain.mean() + 0.25 * domain.min())
    summary = {
        "selection": selection,
        "pooled_development_ads": pooled,
        "mean_normalized_domain_ads": float(domain.mean()),
        "worst_normalized_domain_ads": float(domain.min()),
        "worst_domain": metric_frame.loc[
            metric_frame.NORMALIZED_AVAILABLE_ADS.idxmin(), "DATASET"
        ],
    }
    args.output_dir.mkdir(parents=True)
    metric_frame.to_csv(args.output_dir / "development_metrics.csv", index=False)
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
