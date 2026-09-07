#!/usr/bin/env python3
"""Score a frozen EAT patch-graph checkpoint on untouched audit banks."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from eat_patch_graph import EatPatchGraphHead  # noqa: E402
from evaluate_diagnostic import score_frame  # noqa: E402
from train_eat_patch_graph import load_block, move_to_device, predict  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--cache-root", type=Path, default=ROOT / "output/eat_patch_graph_v1"
    )
    parser.add_argument("--datasets", nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args()
    checkpoint = torch.load(
        args.checkpoint, map_location="cpu", weights_only=False
    )
    if checkpoint.get("model_type") != "eat_patch_graph_multitask":
        raise ValueError("not an EAT patch-graph checkpoint")
    model = EatPatchGraphHead(**checkpoint["config"])
    model.load_state_dict(checkpoint["model"], strict=True)
    device = torch.device(args.device)
    model = model.eval().to(device)

    metric_rows, prediction_rows = [], []
    for name in args.datasets:
        frame, block = load_block(args.cache_root, name, "holdout")
        if (
            not np.array_equal(block["projection"], checkpoint["projection"])
            or not np.array_equal(block["layers"], checkpoint["eat_layers"])
        ):
            raise ValueError(f"patch-graph metadata differs for {name}")
        probability = predict(
            model, move_to_device(block, device), device, args.batch_size
        )
        prediction = pd.DataFrame({
            "DATASET": name,
            "ID": frame.ID,
            "VOICE_FAKE_PROB": probability[:, 0],
            "MUSIC_FAKE_PROB": probability[:, 1],
            "FILE_FAKE_PROB": probability[:, 2],
        })
        joined = frame.set_index("ID").join(prediction.set_index("ID").drop(
            columns="DATASET"
        ))
        try:
            metric = score_frame(joined)
        except ValueError:
            metric = {"N": len(frame)}
        metric_rows.append({"DATASET": name, **metric})
        prediction_rows.append(prediction)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    metrics = pd.DataFrame(metric_rows)
    predictions = pd.concat(prediction_rows, ignore_index=True)
    metrics.to_csv(args.output_dir / "metrics.csv", index=False)
    predictions.to_csv(args.output_dir / "predictions.csv", index=False)
    suno = predictions.loc[predictions.DATASET.str.startswith("suno")]
    summary = {
        "checkpoint": str(args.checkpoint),
        "datasets": args.datasets,
        "suno_count": len(suno),
        "suno_file_above_half": int(suno.FILE_FAKE_PROB.gt(.5).sum()),
        "suno_music_above_half": int(suno.MUSIC_FAKE_PROB.gt(.5).sum()),
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(metrics.to_string(index=False))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
