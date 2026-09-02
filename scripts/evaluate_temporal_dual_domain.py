#!/usr/bin/env python3
"""Evaluate temporal MIL checkpoints on cached source/channel-disjoint banks."""

from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from temporal_dual_domain_head import TemporalDualDomainHead  # noqa: E402
from train_dual_domain_head import evaluate_banks, load_bank  # noqa: E402
from train_temporal_dual_domain_head import predict  # noqa: E402


LOCKED_NAMES = {
    "factorial_eval_1200_v2_holdout", "factorial_eval_1200_v2_locked",
    "source_disjoint_mixed_locked_v1", "yue_cross_component_audit_v1",
    "phone_factorial_1200_v1",
}


def load_model(path: Path, device: torch.device):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    config = {
        key: value for key, value in checkpoint["config"].items()
        if key in {"width", "heads", "dropout", "stream_dropout"}
    }
    model = TemporalDualDomainHead(**config).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    normal = checkpoint["normalization"]
    norm = {
        f"{stream}_{stat}": torch.from_numpy(normal[f"{stream}_{stat}"]).to(device)[None, None]
        for stream in ("eat", "spear") for stat in ("mean", "std")
    }
    return model, norm


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, nargs="+", required=True)
    parser.add_argument("--datasets", nargs="+", required=True)
    parser.add_argument("--channels", nargs="+", default=["clean"])
    parser.add_argument("--stats-root", type=Path, default=ROOT / "output/dual_domain_stats_v1")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--allow-locked", action="store_true")
    parser.add_argument("--predictions-only", action="store_true")
    args = parser.parse_args()
    requested_locked = LOCKED_NAMES.intersection(args.datasets)
    if requested_locked and not args.allow_locked:
        parser.error(f"locked evaluation requires --allow-locked: {sorted(requested_locked)}")

    banks = [
        load_bank(args.stats_root, name, channel)
        for name in args.datasets for channel in args.channels
    ]
    device = torch.device(args.device)
    members = []
    for path in args.checkpoint:
        model, norm = load_model(path, device)
        members.append(predict(model, banks, norm, device, args.batch_size))
        del model
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
    ensemble = {
        (bank.name, bank.channel): np.mean(
            [item[(bank.name, bank.channel)] for item in members], axis=0
        )
        for bank in banks
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for bank in banks:
        values = ensemble[(bank.name, bank.channel)]
        for sample_id, scores in zip(bank.ids, values):
            rows.append({
                "DATASET": bank.name, "CHANNEL": bank.channel, "ID": sample_id,
                "VOICE_FAKE_PROB": scores[0], "MUSIC_FAKE_PROB": scores[1],
                "FILE_FAKE_PROB": scores[2],
            })
    pd.DataFrame(rows).to_csv(args.output_dir / "predictions.csv", index=False)
    if args.predictions_only:
        print(f"Saved {len(rows)} predictions to {args.output_dir}")
        return
    metrics, selection = evaluate_banks(banks, ensemble)
    metrics.to_csv(args.output_dir / "metrics.csv", index=False)
    summary = {
        "checkpoints": [str(path) for path in args.checkpoint],
        "selection": selection, "mean_ads": float(metrics.ADS.mean()),
        "worst_ads": float(metrics.ADS.min()),
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(metrics.to_string(index=False))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
