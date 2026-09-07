#!/usr/bin/env python3
"""Compare hard routing with soft temporal-bin expert fusion."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from evaluate_diagnostic import score_frame  # noqa: E402
from evaluate_presence_weighted_file_fusion import load, routed_v27  # noqa: E402
from evaluate_temporal_bin_fusion import (  # noqa: E402
    apply_expert, common_inputs, expert_for,
)
from long_horizon_music_inference import _logit  # noqa: E402


DATASETS = {
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


def phone_mask(name: str, ids: pd.Index) -> np.ndarray:
    if name not in ROUTERS:
        return np.zeros(len(ids), dtype=bool)
    router = load(ROUTERS[name]).set_index("ID")
    return router.loc[ids, "IS_PHONE"].to_numpy(bool)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--seed04", type=Path,
        default=ROOT / "reports/spear_temporal_bin_v1/mlp64_seed04_audit/predictions.csv",
    )
    parser.add_argument(
        "--seed05", type=Path,
        default=ROOT / "reports/spear_temporal_bin_v1/mlp64_seed05_audit/predictions.csv",
    )
    parser.add_argument(
        "--output", type=Path,
        default=ROOT / "reports/spear_temporal_bin_v1/router_vs_moe.csv",
    )
    args = parser.parse_args()
    experts = {"04": load(args.seed04), "05": load(args.seed05)}
    banks = {
        name: routed_v27(name, **common_inputs())
        for name in ("dev", "factorial", "phone", "yue")
    }

    rows: list[dict] = []
    predictions: dict[tuple[str, str], tuple[pd.DataFrame, pd.DataFrame]] = {}
    for name, (truth, anchor) in banks.items():
        scores = {
            seed: expert_for(frame, DATASETS[name], anchor.index)
            for seed, frame in experts.items()
        }
        phone = phone_mask(name, anchor.index)
        confidence = (
            np.abs(_logit(scores["05"]))
            > np.abs(_logit(anchor.MUSIC_FAKE_PROB.to_numpy()))
        )
        policies = {
            "anchor_v28": (scores["05"], np.zeros(len(anchor))),
            "soft_moe_w40": (scores["05"], np.full(len(anchor), .40)),
            "domain_weight_router_w70": (
                scores["05"], np.where(phone, .70, .40),
            ),
            "domain_head_router_seed04_w70": (
                np.where(phone, scores["04"], scores["05"]),
                np.where(phone, .70, .40),
            ),
            "hard_phone_router": (
                np.where(phone, scores["04"], scores["05"]),
                np.where(phone, 1.0, .40),
            ),
            "content_presence_hard": (
                scores["05"],
                (anchor.MUSIC_PRESENT_PROB.to_numpy() >= .70).astype(float),
            ),
            "confidence_hard": (scores["05"], confidence.astype(float)),
        }
        for method, (expert, weights) in policies.items():
            prediction = apply_expert(
                anchor, expert, weights, file_weight=0,
                file_consistency_update_weight=.75,
            )
            rows.append({
                "DATASET": name, "METHOD": method,
                "PHONE_COUNT": int(phone.sum()),
                **score_frame(truth.join(prediction)),
            })
            predictions[name, method] = truth, prediction

    methods = sorted({method for _, method in predictions})
    for method in methods:
        truths, outputs = [], []
        for name in ("factorial", "phone"):
            truth, prediction = predictions[name, method]
            truth, prediction = truth.copy(), prediction.copy()
            truth.index = name + "_" + truth.index
            prediction.index = truth.index
            truths.append(truth); outputs.append(prediction)
        rows.append({
            "DATASET": "factorial_plus_phone", "METHOD": method,
            "PHONE_COUNT": sum(
                int(phone_mask(name, banks[name][1].index).sum())
                for name in ("factorial", "phone")
            ),
            **score_frame(pd.concat(truths).join(pd.concat(outputs))),
        })

    result = pd.DataFrame(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)
    print(result[[
        "DATASET", "METHOD", "PHONE_COUNT", "FILE_EER", "VOICE_EER",
        "MUSIC_EER", "ADS",
    ]].to_string(index=False))


if __name__ == "__main__":
    main()
