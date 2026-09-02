#!/usr/bin/env python3
"""Evaluate one or more frozen dual-domain heads on cached holdout banks."""

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

from dual_domain_head import DualDomainHead  # noqa: E402
from train_dual_domain_head import (  # noqa: E402
    evaluate_banks,
    load_bank,
    predict,
)


LOCKED_NAMES = {
    "factorial_eval_1200_v2_locked",
    "source_disjoint_mixed_locked_v1",
    "yue_cross_component_audit_v1",
}


def load_model(path: Path, device: torch.device):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    model = DualDomainHead(**checkpoint["config"]).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    values = checkpoint["normalization"]
    norm = {
        "eat_mean": torch.from_numpy(values["eat_mean"]).to(device)[None, None],
        "eat_std": torch.from_numpy(values["eat_std"]).to(device)[None, None],
        "spear_mean": torch.from_numpy(values["spear_mean"]).to(device)[None, None],
        "spear_std": torch.from_numpy(values["spear_std"]).to(device)[None, None],
    }
    return model, norm


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, nargs="+", required=True)
    parser.add_argument("--datasets", nargs="+", required=True)
    parser.add_argument("--channels", nargs="+", default=["clean"])
    parser.add_argument("--stats-root", type=Path,
                        default=ROOT / "output/dual_domain_stats_v1")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--allow-locked", action="store_true")
    args = parser.parse_args()
    requested_locked = LOCKED_NAMES.intersection(args.datasets)
    if requested_locked and not args.allow_locked:
        parser.error(f"locked evaluation requires --allow-locked: {sorted(requested_locked)}")

    banks = [
        load_bank(args.stats_root, name, channel)
        for name in args.datasets for channel in args.channels
    ]
    device = torch.device(args.device)
    all_predictions = []
    members = []
    for path in args.checkpoint:
        model, norm = load_model(path, device)
        predictions = predict(model, banks, norm, device, args.batch_size)
        all_predictions.append(predictions)
        members.append({"checkpoint": str(path)})
        del model
        torch.cuda.empty_cache()

    ensemble = {}
    for bank in banks:
        key = (bank.name, bank.channel)
        ensemble[key] = np.mean([item[key] for item in all_predictions], axis=0)
    metrics_frame, selection = evaluate_banks(banks, ensemble)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metrics_frame.to_csv(args.output_dir / "metrics.csv", index=False)
    rows = []
    for bank in banks:
        values = ensemble[(bank.name, bank.channel)]
        for item, score in zip(bank.ids, values):
            rows.append({
                "DATASET": bank.name, "CHANNEL": bank.channel, "ID": item,
                "VOICE_FAKE_PROB": score[0], "MUSIC_FAKE_PROB": score[1],
                "FILE_FAKE_PROB": score[2],
            })
    pd.DataFrame(rows).to_csv(args.output_dir / "predictions.csv", index=False)
    summary = {
        "members": members, "selection": selection,
        "mean_ads": float(metrics_frame.ADS.mean()),
        "worst_ads": float(metrics_frame.ADS.min()),
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(metrics_frame.to_string(index=False))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
