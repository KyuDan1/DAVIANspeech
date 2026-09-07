#!/usr/bin/env python3
"""Score a frozen SPEAR temporal-bin MIL expert on cached audit banks."""

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

from spear_temporal_bin_head import SpearTemporalBinHead  # noqa: E402
from train_spear_temporal_bin_mil import (  # noqa: E402
    align, evaluate, load_archive, truth,
)


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
    if not np.array_equal(cache["__metadata__"]["projection"], checkpoint["projection"]):
        raise ValueError("audit cache projection differs from the trained head")
    device = torch.device(args.device)
    state = checkpoint["model"]
    model = SpearTemporalBinHead(
        checkpoint["config"]["feature_dimension"], state["mean"], state["std"],
        hidden=checkpoint["config"]["hidden"],
        dropout=checkpoint["config"]["dropout"],
        temperature=checkpoint["config"]["temperature"],
        minimum_presence_weight=checkpoint["config"]["minimum_presence_weight"],
    ).to(device)
    model.load_state_dict(state, strict=True)
    frames, values = [], []
    for name in args.datasets:
        frame = truth(name)
        frame["DATASET"] = name
        block = align(cache[name], frame)
        block["features"] = block["features"].reshape(
            len(frame), block["features"].shape[1], block["features"].shape[2], -1
        )
        frames.append(frame); values.append(block)
    metrics, predictions, selection = evaluate(model, frames, values, device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(args.output_dir / "metrics.csv", index=False)
    predictions.to_csv(args.output_dir / "predictions.csv", index=False)
    (args.output_dir / "summary.json").write_text(json.dumps({
        "selection": selection, "datasets": args.datasets,
    }, indent=2), encoding="utf-8")
    print(metrics.to_string(index=False)); print(f"selection={selection:.6f}")


if __name__ == "__main__":
    main()
