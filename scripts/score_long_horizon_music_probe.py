#!/usr/bin/env python3
"""Score a frozen long-horizon music probe on labelled audit banks."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from train_dual_domain_head import load_bank  # noqa: E402
from train_long_horizon_music_probe import (  # noqa: E402
    evaluate,
    load_projection,
    load_random_projection,
    projected_features,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--head", type=Path, required=True)
    parser.add_argument("--datasets", nargs="+", required=True)
    parser.add_argument("--stats-root", type=Path,
                        default=ROOT / "output/dual_domain_stats_v1")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args()

    checkpoint = torch.load(args.head, map_location="cpu", weights_only=False)
    device = torch.device(args.device)
    if checkpoint.get("projection_kind", "invariant") == "random":
        projection, projection_norm = load_random_projection(
            device, int(checkpoint["projection_width"]),
            int(checkpoint["projection_seed"]),
        )
    else:
        projection, projection_norm = load_projection(
            Path(checkpoint["projection_checkpoint"]), device
        )
    frames, score_blocks, prediction_blocks = [], [], []
    for name in args.datasets:
        bank = load_bank(args.stats_root, name, "clean")
        features = projected_features(
            bank, projection, projection_norm, device,
            list(checkpoint["layers"]), str(checkpoint["feature_mode"]),
            args.batch_size,
        )
        features = np.clip(
            (features - np.asarray(checkpoint["mean"]))
            / np.asarray(checkpoint["std"]), -8, 8,
        ).astype(np.float32)
        classifier = nn.Linear(features.shape[1], 1).to(device)
        classifier.load_state_dict(checkpoint["classifier"], strict=True)
        classifier.eval()
        with torch.inference_mode():
            score = classifier(
                torch.from_numpy(features).to(device)
            ).squeeze(1).sigmoid().cpu().numpy()
        frame = bank.truth.copy()
        frame["DATASET"] = name
        frames.append(frame)
        score_blocks.append(score)
        prediction_blocks.append(pd.DataFrame({
            "DATASET": name, "ID": bank.ids,
            "LONG_HORIZON_MUSIC_PROB": score,
        }))
        print(f"scored {name}: {len(score)}", flush=True)

    result, selection = evaluate(frames, score_blocks)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output_dir / "metrics.csv", index=False)
    pd.concat(prediction_blocks, ignore_index=True).to_csv(
        args.output_dir / "predictions.csv", index=False
    )
    summary = {"selection": selection, "datasets": args.datasets}
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(result.to_string(index=False))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
