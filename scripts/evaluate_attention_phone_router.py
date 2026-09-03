#!/usr/bin/env python3
"""Evaluate soft/hard telephone routing for a residual attention expert.

The production joint MoE remains the anchor.  The router controls only the
amount of temporal-attention File evidence mixed in logit space; Voice, Music,
and presence predictions remain unchanged.  A zero non-phone weight provides
an especially useful test of whether routing isolates the expert's telephone
advantage without leaking its clean-domain regressions.
"""

from __future__ import annotations

import argparse
from itertools import product
from pathlib import Path
import sys

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from evaluate_diagnostic import official_eer  # noqa: E402
from evaluate_attention_residual_moe import (  # noqa: E402
    EXPERT_DATASETS, TRUTHS, indexed, logit, sigmoid,
)


ROUTERS = {
    "dev": ROOT / "reports/phone_presence_probe_v1/factorial_eval_1200_v2_router.csv",
    "factorial": ROOT / "reports/phone_presence_probe_v1/factorial_eval_1200_v2_router.csv",
    "phone": ROOT / "reports/phone_presence_probe_v1/phone_factorial_1200_v1_router.csv",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--anchor-predictions", type=Path,
        default=ROOT / "reports/spear_temporal_joint_v1/selected_v29_predictions.csv",
    )
    parser.add_argument("--attention-predictions", type=Path, required=True)
    parser.add_argument(
        "--weights", type=float, nargs="+",
        default=[0, .025, .05, .075, .10, .125, .15, .20, .25],
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    anchors = pd.read_csv(args.anchor_predictions, dtype={"ID": str})
    attentions = pd.read_csv(args.attention_predictions, dtype={"ID": str})
    banks = {}
    for name, truth_path in TRUTHS.items():
        truth = indexed(truth_path)
        anchor = (
            anchors.loc[anchors.DATASET.eq(name) & anchors.METHOD.eq("joint_moe")]
            .drop(columns=["DATASET", "METHOD"]).set_index("ID").loc[truth.index]
        )
        attention = (
            attentions.loc[attentions.DATASET.eq(EXPERT_DATASETS[name])]
            .set_index("ID").loc[truth.index]
        )
        phone_probability = (
            np.zeros(len(truth), dtype=np.float64)
            if name == "yue" else
            indexed(ROUTERS[name]).loc[truth.index, "PHONE_PROB"].to_numpy(np.float64)
        )
        banks[name] = truth, anchor, attention, phone_probability

    rows = []
    for route, (nonphone_weight, phone_weight) in product(
        ("soft", "hard"), product(args.weights, args.weights)
    ):
        record = {
            "ROUTE": route, "NONPHONE_W": nonphone_weight, "PHONE_W": phone_weight,
        }
        for name, (truth, anchor, attention, phone_probability) in banks.items():
            gate = phone_probability if route == "soft" else (phone_probability >= .5)
            weights = nonphone_weight + (phone_weight - nonphone_weight) * gate
            file_score = sigmoid(
                (1 - weights) * logit(anchor.FILE_FAKE_PROB)
                + weights * logit(attention.FILE_FAKE_PROB)
            )
            file_eer = official_eer(truth.FILE_FAKE.astype(int), file_score)
            voice = truth.VOICE_PRESENT.eq(1)
            music = truth.MUSIC_PRESENT.eq(1)
            voice_eer = official_eer(
                truth.loc[voice, "VOICE_FAKE"].astype(int),
                anchor.loc[voice, "VOICE_FAKE_PROB"],
            )
            music_eer = official_eer(
                truth.loc[music, "MUSIC_FAKE"].astype(int),
                anchor.loc[music, "MUSIC_FAKE_PROB"],
            )
            record[f"{name}_FILE_EER"] = file_eer
            record[f"{name}_ADS"] = (
                .5 * (1 - file_eer) + .2 * (1 - voice_eer) + .3 * (1 - music_eer)
            )
        rows.append(record)

    result = pd.DataFrame(rows)
    baseline = result.loc[
        result.ROUTE.eq("soft") & result.NONPHONE_W.eq(0) & result.PHONE_W.eq(0)
    ].iloc[0]
    for name in TRUTHS:
        result[f"{name}_DELTA"] = result[f"{name}_ADS"] - baseline[f"{name}_ADS"]
    audit = [f"{name}_DELTA" for name in ("factorial", "phone", "yue")]
    result["AUDIT_MIN_DELTA"] = result[audit].min(axis=1)
    result["AUDIT_MEAN_DELTA"] = result[audit].mean(axis=1)
    result = result.sort_values(
        ["dev_DELTA", "AUDIT_MIN_DELTA", "AUDIT_MEAN_DELTA"], ascending=False
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)
    print("anchor", {
        name: round(float(baseline[f"{name}_ADS"]), 8) for name in TRUTHS
    })
    print(result[[
        "ROUTE", "NONPHONE_W", "PHONE_W", "dev_DELTA", "factorial_DELTA",
        "phone_DELTA", "yue_DELTA", "AUDIT_MIN_DELTA", "AUDIT_MEAN_DELTA",
    ]].head(30).to_string(index=False))


if __name__ == "__main__":
    main()
