#!/usr/bin/env python3
"""Evaluate a File-only attention router on the exact reconstructed v18 anchor."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from evaluate_codec_invariant_fusion import AUDITS, fuse, reconstruct_v18  # noqa: E402
from evaluate_diagnostic import score_frame  # noqa: E402
from evaluate_presence_weighted_file_fusion import reconstruct_dev_v18  # noqa: E402


EXPERT_DATASETS = {
    "dev": "factorial_eval_1200_v2",
    "factorial": "factorial_eval_1200_v2",
    "phone": "phone_factorial_1200_v1",
    "yue": "yue_cross_component_audit_v1",
}
ROUTERS = {
    "dev": ROOT / "reports/phone_presence_probe_v1/factorial_eval_1200_v2_router.csv",
    "factorial": ROOT / "reports/phone_presence_probe_v1/factorial_eval_1200_v2_router.csv",
    "phone": ROOT / "reports/phone_presence_probe_v1/phone_factorial_1200_v1_router.csv",
}


def load(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, dtype={"ID": str})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--attention-predictions", type=Path, required=True)
    parser.add_argument(
        "--weights", type=float, nargs="+",
        default=[0, .025, .05, .075, .10, .125, .15, .20, .25],
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    temporal_dev = load(
        ROOT / "reports/temporal_dual_domain_hybrid/ensemble_dev/predictions.csv"
    )
    temporal_audit = load(
        ROOT / "reports/temporal_dual_domain_hybrid/ensemble_audit/predictions.csv"
    )
    fakeprint = load(
        ROOT / "reports/modern_fakeprint_v1/independent_margin_scores.csv"
    )
    invariant_dev = load(
        ROOT / "reports/invariant_dual_domain_v2/dec_n1/dev_predictions.csv"
    )
    invariant_audit = load(
        ROOT / "reports/invariant_dual_domain_v2/dec_n1_audit/predictions.csv"
    )
    attention = load(args.attention_predictions)
    banks = {}
    for name in EXPERT_DATASETS:
        if name == "dev":
            truth, anchor = reconstruct_dev_v18(
                temporal_dev, fakeprint, invariant_dev
            )
        else:
            truth, anchor = reconstruct_v18(
                AUDITS[name], temporal_audit, fakeprint, invariant_audit
            )
        expert = (
            attention.loc[attention.DATASET.eq(EXPERT_DATASETS[name])]
            .set_index("ID").loc[anchor.index]
        )
        if name == "yue":
            phone = np.zeros(len(anchor), dtype=bool)
        else:
            phone = (
                load(ROUTERS[name]).set_index("ID")
                .loc[anchor.index, "IS_PHONE"].to_numpy(bool)
            )
        banks[name] = truth, anchor, expert, phone

    rows = []
    for nonphone_weight in args.weights:
        for phone_weight in args.weights:
            record = {
                "NONPHONE_W": nonphone_weight, "PHONE_W": phone_weight,
            }
            for name, (truth, anchor, expert, phone) in banks.items():
                candidate = anchor.copy()
                weight = np.where(phone, phone_weight, nonphone_weight)
                candidate["FILE_FAKE_PROB"] = fuse(
                    anchor.FILE_FAKE_PROB, expert.FILE_FAKE_PROB, weight
                )
                metrics = score_frame(truth.join(candidate))
                record[f"{name}_FILE_EER"] = metrics["FILE_EER"]
                record[f"{name}_ADS"] = metrics["ADS"]
            rows.append(record)

    result = pd.DataFrame(rows)
    baseline = result.loc[
        result.NONPHONE_W.eq(0) & result.PHONE_W.eq(0)
    ].iloc[0]
    for name in banks:
        result[f"{name}_DELTA"] = result[f"{name}_ADS"] - baseline[f"{name}_ADS"]
    audit = [f"{name}_DELTA" for name in ("factorial", "phone", "yue")]
    result["AUDIT_MIN_DELTA"] = result[audit].min(axis=1)
    result["AUDIT_MEAN_DELTA"] = result[audit].mean(axis=1)
    result = result.sort_values(
        ["dev_DELTA", "AUDIT_MIN_DELTA", "AUDIT_MEAN_DELTA"], ascending=False
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)
    print("v18", {
        name: round(float(baseline[f"{name}_ADS"]), 8) for name in banks
    })
    print(result[[
        "NONPHONE_W", "PHONE_W", "dev_DELTA", "factorial_DELTA",
        "phone_DELTA", "yue_DELTA", "AUDIT_MIN_DELTA", "AUDIT_MEAN_DELTA",
    ]].head(30).to_string(index=False))


if __name__ == "__main__":
    main()
