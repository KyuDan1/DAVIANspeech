#!/usr/bin/env python3
"""Score a frozen SPEAR temporal joint head on cached audit banks."""

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

from evaluate_diagnostic import score_frame  # noqa: E402
from spear_temporal_joint_head import (  # noqa: E402
    SpearTemporalJointAttentionHead, SpearTemporalJointHead,
)
from train_spear_temporal_bin_mil import align, load_archive, truth  # noqa: E402
from train_spear_temporal_joint_mil import predict  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--head", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--datasets", nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    checkpoint = torch.load(args.head, map_location="cpu", weights_only=False)
    cache = load_archive(args.cache_root)
    if not np.array_equal(
        cache["__metadata__"]["projection"], checkpoint["projection"]
    ):
        raise ValueError("audit cache projection differs from the joint head")
    config, state = checkpoint["config"], checkpoint["model"]
    model_class = (
        SpearTemporalJointAttentionHead
        if config.get("architecture") == "attention" else SpearTemporalJointHead
    )
    extra = (
        {"layers": config["attention_layers"], "heads": config["attention_heads"]}
        if config.get("architecture") == "attention" else {}
    )
    model = model_class(
        config["feature_dimension"], state["mean"], state["std"],
        hidden=config["hidden"], dropout=config["dropout"],
        temperature=config["temperature"],
        minimum_presence_weight=config["minimum_presence_weight"],
        **extra,
    ).to(args.device)
    model.load_state_dict(state, strict=True); model.eval()
    rows, predictions = [], []
    for name in args.datasets:
        frame = truth(name)
        block = align(cache[name], frame)
        values = predict(model, block, torch.device(args.device))
        output = pd.DataFrame({
            "DATASET": name, "ID": frame.ID,
            "FILE_FAKE_PROB": values[0], "VOICE_FAKE_PROB": values[1],
            "MUSIC_FAKE_PROB": values[2], "VOICE_PRESENT_PROB": values[3],
            "MUSIC_PRESENT_PROB": values[4],
        })
        predictions.append(output)
        metrics = score_frame(
            frame.set_index("ID").join(output.set_index("ID").drop(columns="DATASET"))
        )
        rows.append({"DATASET": name, **metrics})
    metrics = pd.DataFrame(rows)
    predictions = pd.concat(predictions, ignore_index=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(args.output_dir / "metrics.csv", index=False)
    predictions.to_csv(args.output_dir / "predictions.csv", index=False)
    (args.output_dir / "summary.json").write_text(json.dumps({
        "head": str(args.head), "datasets": args.datasets,
    }, indent=2), encoding="utf-8")
    print(metrics.to_string(index=False))


if __name__ == "__main__":
    main()
