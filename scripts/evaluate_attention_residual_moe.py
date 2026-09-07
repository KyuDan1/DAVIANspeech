#!/usr/bin/env python3
"""Compare a temporal-attention residual expert with the frozen joint MoE.

The existing ``joint_moe`` predictions are treated as the production anchor.
Only File and Voice logits receive a small residual from the new expert; Music
and both presence outputs stay byte-for-byte equivalent to the anchor.  This
keeps the experiment focused and prevents a weak auxiliary head from changing
the already stronger Music/CPS branches.
"""

from __future__ import annotations

import argparse
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


def eer(truth: pd.DataFrame, score: np.ndarray, axis: str) -> float:
    keep = np.ones(len(truth), dtype=bool)
    if axis != "FILE":
        keep = truth[f"{axis}_PRESENT"].eq(1).to_numpy()
    return official_eer(truth.loc[keep, f"{axis}_FAKE"].astype(int), score[keep])


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

    all_anchor = pd.read_csv(args.anchor_predictions, dtype={"ID": str})
    all_attention = pd.read_csv(args.attention_predictions, dtype={"ID": str})
    banks = {}
    for name, truth_path in TRUTHS.items():
        truth = indexed(truth_path)
        anchor = (
            all_anchor.loc[
                all_anchor.DATASET.eq(name) & all_anchor.METHOD.eq("joint_moe")
            ]
            .drop(columns=["DATASET", "METHOD"]).set_index("ID").loc[truth.index]
        )
        attention = (
            all_attention.loc[
                all_attention.DATASET.eq(EXPERT_DATASETS[name])
            ].set_index("ID").loc[truth.index]
        )
        banks[name] = (truth, anchor, attention)

    rows = []
    for file_weight in args.weights:
        for voice_weight in args.weights:
            record = {"FILE_W": file_weight, "VOICE_W": voice_weight}
            for name, (truth, anchor, attention) in banks.items():
                file_score = sigmoid(
                    (1 - file_weight) * logit(anchor.FILE_FAKE_PROB)
                    + file_weight * logit(attention.FILE_FAKE_PROB)
                )
                voice_score = sigmoid(
                    (1 - voice_weight) * logit(anchor.VOICE_FAKE_PROB)
                    + voice_weight * logit(attention.VOICE_FAKE_PROB)
                )
                file_eer = eer(truth, file_score, "FILE")
                voice_eer = eer(truth, voice_score, "VOICE")
                music_eer = eer(
                    truth, anchor.MUSIC_FAKE_PROB.to_numpy(np.float64), "MUSIC"
                )
                record[f"{name}_FILE_EER"] = file_eer
                record[f"{name}_VOICE_EER"] = voice_eer
                record[f"{name}_ADS"] = (
                    .5 * (1 - file_eer)
                    + .2 * (1 - voice_eer)
                    + .3 * (1 - music_eer)
                )
            rows.append(record)

    result = pd.DataFrame(rows)
    baseline = result.loc[result.FILE_W.eq(0) & result.VOICE_W.eq(0)].iloc[0]
    for name in TRUTHS:
        result[f"{name}_DELTA"] = result[f"{name}_ADS"] - baseline[f"{name}_ADS"]
    result["AUDIT_MIN_DELTA"] = result[
        [f"{name}_DELTA" for name in ("factorial", "phone", "yue")]
    ].min(axis=1)
    result["AUDIT_MEAN_DELTA"] = result[
        [f"{name}_DELTA" for name in ("factorial", "phone", "yue")]
    ].mean(axis=1)
    result["DEV_THEN_AUDIT_RANK"] = (
        result.dev_DELTA.rank(method="min", ascending=False) * 10000
        + result.AUDIT_MIN_DELTA.rank(method="min", ascending=False)
    )
    result = result.sort_values(
        ["dev_DELTA", "AUDIT_MIN_DELTA", "AUDIT_MEAN_DELTA"], ascending=False
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)
    print("anchor", {
        name: round(float(baseline[f"{name}_ADS"]), 8) for name in TRUTHS
    })
    columns = [
        "FILE_W", "VOICE_W",
        *[f"{name}_DELTA" for name in TRUTHS],
        "AUDIT_MIN_DELTA", "AUDIT_MEAN_DELTA",
    ]
    print(result[columns].head(25).to_string(index=False))


if __name__ == "__main__":
    main()
