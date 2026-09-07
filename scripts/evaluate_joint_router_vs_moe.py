#!/usr/bin/env python3
"""Compare hard expert routers against fixed and domain-conditioned soft MoE."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from evaluate_diagnostic import score_frame  # noqa: E402
from train_joint_expert_router import (  # noqa: E402
    TRAIN_DATASETS, audit_bank, joint_predictions, logit, sigmoid, training_bank,
)


def fuse(bank: dict, strategy: str) -> pd.DataFrame:
    result = bank["anchor"].copy()
    phone = bank["phone"] >= .5
    mixed = (
        result.VOICE_PRESENT_PROB.to_numpy() >= .7
    ) & (result.MUSIC_PRESENT_PROB.to_numpy() >= .7)
    for axis in ("FILE", "VOICE"):
        column = f"{axis}_FAKE_PROB"
        anchor = logit(bank["anchor"][column])
        expert = logit(bank["expert"][column])
        if strategy == "anchor":
            weight = np.zeros(len(result))
        elif strategy == "fixed_soft_moe":
            weight = np.full(len(result), .25)
        elif strategy == "domain_soft_moe":
            if axis == "FILE":
                weight = np.where(phone, .40, .225)
            else:
                weight = np.where(phone, .35, .30)
        elif strategy == "phone_hard_expert":
            weight = phone.astype(float)
        elif strategy == "content_hard_expert":
            weight = mixed.astype(float)
        elif strategy == "confidence_hard_expert":
            weight = (np.abs(expert) > np.abs(anchor)).astype(float)
        else:
            raise ValueError(strategy)
        result[column] = sigmoid((1 - weight) * anchor + weight * expert)
    return result


def main() -> None:
    train_joint = joint_predictions()
    banks = {name: training_bank(train_joint, name) for name in TRAIN_DATASETS}
    v29 = pd.read_csv(
        ROOT / "reports/spear_temporal_joint_v1/selected_v29_predictions.csv",
        dtype={"ID": str},
    )
    audit_joint = pd.read_csv(
        ROOT / "reports/spear_temporal_joint_v1/seed09_audit/predictions.csv",
        dtype={"ID": str},
    )
    for name in ("dev", "factorial", "phone", "yue"):
        banks[name] = audit_bank(v29, audit_joint, name)

    strategies = (
        "anchor", "fixed_soft_moe", "domain_soft_moe",
        "phone_hard_expert", "content_hard_expert", "confidence_hard_expert",
    )
    rows = []
    for name, bank in banks.items():
        for strategy in strategies:
            prediction = fuse(bank, strategy)
            rows.append({
                "DATASET": name, "STRATEGY": strategy,
                **score_frame(bank["truth"].join(prediction)),
            })
    result = pd.DataFrame(rows)
    baseline = (
        result[result.STRATEGY.eq("fixed_soft_moe")]
        .set_index("DATASET").ADS
    )
    result["DELTA_VS_FIXED_MOE"] = [
        row.ADS - baseline[row.DATASET] for row in result.itertuples()
    ]
    summary = result.groupby("STRATEGY").agg(
        MEAN_ADS=("ADS", "mean"),
        MIN_DELTA=("DELTA_VS_FIXED_MOE", "min"),
        MEAN_DELTA=("DELTA_VS_FIXED_MOE", "mean"),
    ).reset_index().sort_values(["MIN_DELTA", "MEAN_DELTA"], ascending=False)
    output = ROOT / "reports/router_training_v1/joint_router_vs_moe"
    output.mkdir(parents=True, exist_ok=True)
    result.to_csv(output / "per_dataset.csv", index=False)
    summary.to_csv(output / "summary.csv", index=False)
    print(summary.to_string(index=False))
    print(result.pivot(index="DATASET", columns="STRATEGY", values="ADS").to_string())


if __name__ == "__main__":
    main()
