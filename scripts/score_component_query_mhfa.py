#!/usr/bin/env python3
"""Score a frozen component-query MHFA checkpoint on untouched audit banks."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from component_query_mhfa import ComponentQueryMHFA  # noqa: E402
from train_component_query_mhfa import evaluate, load_block, move_to_device  # noqa: E402
from train_spear_temporal_bin_mil import load_archive  # noqa: E402


DEFAULT_DATASETS = (
    "factorial_eval_1200_v2_holdout", "phone_factorial_1200_v1",
    "yue_cross_component_audit_v1", "suno_vocals_v1",
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--eat-cache-root", type=Path, default=ROOT / "output/eat_patch_graph_v1"
    )
    parser.add_argument(
        "--spear-cache-root", type=Path,
        default=ROOT / "reports/spear_temporal_attention_v1/cache",
    )
    parser.add_argument("--datasets", nargs="+", default=list(DEFAULT_DATASETS))
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--batch-size", type=int, default=96)
    args = parser.parse_args()

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if checkpoint.get("model_type") != "component_query_mhfa_v1":
        raise ValueError("not a component-query MHFA checkpoint")
    spear_cache = load_archive(args.spear_cache_root)
    pairs = [
        load_block(args.eat_cache_root, spear_cache, name, "dev")
        for name in args.datasets
    ]
    spear_metadata = spear_cache["__metadata__"]
    for _, block in pairs:
        if (
            not np.array_equal(block["eat_projection"], checkpoint["eat_projection"])
            or not np.array_equal(block["eat_layers"], checkpoint["eat_layers"])
        ):
            raise ValueError("EAT audit metadata differs from checkpoint")
    for key, saved in (
        ("projection", "spear_projection"), ("layers", "spear_layers"),
        ("bins", "spear_bins"),
    ):
        if not np.array_equal(spear_metadata[key], checkpoint[saved]):
            raise ValueError(f"SPEAR {key} differs from checkpoint")
    device = torch.device(args.device)
    model = ComponentQueryMHFA(**checkpoint["config"]).to(device)
    model.load_state_dict(checkpoint["model"], strict=True)
    frames, numpy_blocks = zip(*pairs)
    blocks = tuple(move_to_device(block, device) for block in numpy_blocks)
    metrics, predictions, selection = evaluate(
        model, frames, blocks, device, args.batch_size
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(args.output_dir / "metrics.csv", index=False)
    predictions.to_csv(args.output_dir / "predictions.csv", index=False)
    summary = {"selection": selection, "datasets": args.datasets}
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(metrics.to_string(index=False))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
