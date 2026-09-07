#!/usr/bin/env python3
"""Tune phone/non-phone joint-expert weights without changing experts.

Unlike a hard expert switch, this router interpolates the fusion weight from a
telephone probability.  File and Voice are tuned independently because Music
and CPS already come from stronger branches.  All scores are reconstructed
from frozen predictions, so every candidate sees exactly the same samples.
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


TRUTHS = {
    "dev": ROOT / "data/eval/factorial_eval_1200_v2/truth_dev.csv",
    "factorial": ROOT / "data/eval/factorial_eval_1200_v2/truth_holdout.csv",
    "phone": ROOT / "data/eval/phone_factorial_1200_v1/truth.csv",
    "yue": ROOT / "data/eval/yue_cross_component_audit_v1/truth.csv",
}
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


def indexed(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, dtype={"ID": str}).set_index("ID")
    if frame.index.duplicated().any():
        raise ValueError(f"duplicate ID in {path}")
    return frame


def logit(values) -> np.ndarray:
    values = np.clip(np.asarray(values, dtype=np.float64), 1e-5, 1 - 1e-5)
    return np.log(values) - np.log1p(-values)


def sigmoid(values) -> np.ndarray:
    return np.exp(-np.logaddexp(0, -np.asarray(values, dtype=np.float64)))


def bank(v29: pd.DataFrame, joint: pd.DataFrame, name: str) -> dict:
    truth = indexed(TRUTHS[name])
    anchor = (
        v29.loc[v29.DATASET.eq(name) & v29.METHOD.eq("v29")]
        .drop(columns=["DATASET", "METHOD"]).set_index("ID").loc[truth.index]
    )
    expert = (
        joint.loc[joint.DATASET.eq(EXPERT_DATASETS[name])]
        .set_index("ID").loc[truth.index]
    )
    if name == "yue":
        phone_probability = np.zeros(len(truth), dtype=np.float64)
    else:
        phone_probability = indexed(ROUTERS[name]).loc[
            truth.index, "PHONE_PROB"
        ].to_numpy(np.float64)
    return {
        "truth": truth, "anchor": anchor, "expert": expert,
        "phone_probability": phone_probability,
    }


def routed_weight(phone_probability, nonphone: float, phone: float, route: str):
    if route == "hard":
        gate = phone_probability >= .5
    elif route == "soft":
        gate = phone_probability
    else:
        raise ValueError(route)
    return nonphone + (phone - nonphone) * gate


def axis_eers(
    banks: dict[str, dict], axis: str, nonphone: float, phone: float, route: str,
) -> dict[str, float]:
    column = f"{axis}_FAKE_PROB"
    target = f"{axis}_FAKE"
    results: dict[str, float] = {}
    combined_truth, combined_score = [], []
    for name, values in banks.items():
        truth, anchor, expert = values["truth"], values["anchor"], values["expert"]
        weights = routed_weight(
            values["phone_probability"], nonphone, phone, route
        )
        score = sigmoid(
            (1 - weights) * logit(anchor[column])
            + weights * logit(expert[column])
        )
        present = np.ones(len(truth), dtype=bool)
        if axis != "FILE":
            present = truth[f"{axis}_PRESENT"].eq(1).to_numpy()
        results[name] = official_eer(
            truth.loc[present, target].astype(int), score[present]
        )
        if name in ("factorial", "phone"):
            combined_truth.append(truth.loc[present, target].astype(int).to_numpy())
            combined_score.append(score[present])
    results["factorial_plus_phone"] = official_eer(
        np.concatenate(combined_truth), np.concatenate(combined_score)
    )
    return results


def fixed_music_eers(banks: dict[str, dict]) -> dict[str, float]:
    results, combined_truth, combined_score = {}, [], []
    for name, values in banks.items():
        truth, anchor = values["truth"], values["anchor"]
        present = truth.MUSIC_PRESENT.eq(1)
        results[name] = official_eer(
            truth.loc[present, "MUSIC_FAKE"].astype(int),
            anchor.loc[present, "MUSIC_FAKE_PROB"],
        )
        if name in ("factorial", "phone"):
            combined_truth.append(truth.loc[present, "MUSIC_FAKE"].astype(int))
            combined_score.append(anchor.loc[present, "MUSIC_FAKE_PROB"])
    results["factorial_plus_phone"] = official_eer(
        pd.concat(combined_truth), pd.concat(combined_score)
    )
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--v29-predictions", type=Path,
        default=ROOT / "reports/spear_temporal_joint_v1/selected_v29_predictions.csv",
    )
    parser.add_argument(
        "--joint-predictions", type=Path,
        default=ROOT / "reports/spear_temporal_joint_v1/seed09_audit/predictions.csv",
    )
    parser.add_argument(
        "--weights", type=float, nargs="+",
        default=[.15, .20, .225, .25, .275, .30, .35, .40],
    )
    parser.add_argument(
        "--output", type=Path,
        default=ROOT / "reports/spear_temporal_joint_v1/domain_weight_router.csv",
    )
    args = parser.parse_args()
    v29 = pd.read_csv(args.v29_predictions, dtype={"ID": str})
    joint = pd.read_csv(args.joint_predictions, dtype={"ID": str})
    banks = {name: bank(v29, joint, name) for name in TRUTHS}
    music = fixed_music_eers(banks)
    datasets = [*TRUTHS, "factorial_plus_phone"]
    rows = []
    for route in ("hard", "soft"):
        settings = list(product(args.weights, args.weights))
        file_bank = {
            setting: axis_eers(banks, "FILE", *setting, route)
            for setting in settings
        }
        voice_bank = {
            setting: axis_eers(banks, "VOICE", *setting, route)
            for setting in settings
        }
        for file_setting, voice_setting in product(settings, settings):
            record = {
                "ROUTE": route,
                "FILE_NONPHONE_W": file_setting[0], "FILE_PHONE_W": file_setting[1],
                "VOICE_NONPHONE_W": voice_setting[0], "VOICE_PHONE_W": voice_setting[1],
            }
            for name in datasets:
                record[name] = (
                    .5 * (1 - file_bank[file_setting][name])
                    + .2 * (1 - voice_bank[voice_setting][name])
                    + .3 * (1 - music[name])
                )
            rows.append(record)
    result = pd.DataFrame(rows)
    baseline = result.loc[
        result.FILE_NONPHONE_W.eq(.25)
        & result.FILE_PHONE_W.eq(.25)
        & result.VOICE_NONPHONE_W.eq(.25)
        & result.VOICE_PHONE_W.eq(.25)
    ].iloc[0]
    for name in datasets:
        result[f"{name}_DELTA"] = result[name] - baseline[name]
    audit_delta = [f"{name}_DELTA" for name in TRUTHS]
    result["MIN_AUDIT_DELTA"] = result[audit_delta].min(axis=1)
    result["MEAN_AUDIT_DELTA"] = result[audit_delta].mean(axis=1)
    result = result.sort_values(
        ["MIN_AUDIT_DELTA", "MEAN_AUDIT_DELTA"], ascending=False
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)
    print("v30 baseline", {name: round(float(baseline[name]), 8) for name in datasets})
    columns = [
        "ROUTE", "FILE_NONPHONE_W", "FILE_PHONE_W",
        "VOICE_NONPHONE_W", "VOICE_PHONE_W", *datasets,
        "MIN_AUDIT_DELTA", "MEAN_AUDIT_DELTA",
    ]
    print(result[columns].head(30).to_string(index=False))


if __name__ == "__main__":
    main()
