#!/usr/bin/env python3
"""Compare global MoE and phone-routed ArtifactNet temporal pooling on v30.

The ArtifactNet branch in the deployed anchor uses the median window score.
This audit either adds an alternative pooled score as a small expert (``moe``)
or applies only the alternative-vs-median logit residual (``residual``).  The
latter changes temporal aggregation without counting ArtifactNet twice.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from evaluate_diagnostic import score_frame  # noqa: E402


TRUTHS = {
    "dev": ROOT / "data/eval/factorial_eval_1200_v2/truth_dev.csv",
    "factorial": ROOT / "data/eval/factorial_eval_1200_v2/truth_holdout.csv",
    "phone": ROOT / "data/eval/phone_factorial_1200_v1/truth.csv",
    "yue": ROOT / "data/eval/yue_cross_component_audit_v1/truth.csv",
}
ARTIFACT_DATASETS = {
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


def reconstruct_v30(
    v29: pd.DataFrame, joint: pd.DataFrame, name: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    truth = indexed(TRUTHS[name])
    prediction = (
        v29.loc[v29.DATASET.eq(name) & v29.METHOD.eq("v29")]
        .drop(columns=["DATASET", "METHOD"])
        .set_index("ID")
        .loc[truth.index]
        .copy()
    )
    expert_dataset = ARTIFACT_DATASETS[name]
    expert = joint.loc[joint.DATASET.eq(expert_dataset)].set_index("ID").loc[truth.index]
    for column in ("FILE_FAKE_PROB", "VOICE_FAKE_PROB"):
        prediction[column] = sigmoid(
            .75 * logit(prediction[column]) + .25 * logit(expert[column])
        )
    return truth, prediction


def route_values(name: str, ids: pd.Index, policy: str) -> np.ndarray:
    if policy == "global" or name == "yue":
        return np.ones(len(ids), dtype=np.float64)
    router = indexed(ROUTERS[name]).loc[ids]
    if policy == "hard_nonphone":
        return 1 - router.IS_PHONE.to_numpy(np.float64)
    if policy == "soft_nonphone":
        return 1 - router.PHONE_PROB.to_numpy(np.float64)
    raise ValueError(f"unknown route policy: {policy}")


def apply_candidate(
    prediction: pd.DataFrame,
    pool: np.ndarray,
    median: np.ndarray,
    route: np.ndarray,
    mode: str,
    weight: float,
    file_ratio: float,
) -> pd.DataFrame:
    result = prediction.copy()
    if mode == "moe":
        music_delta = logit(pool) - logit(result.MUSIC_FAKE_PROB)
    elif mode == "residual":
        music_delta = logit(pool) - logit(median)
    else:
        raise ValueError(f"unknown mode: {mode}")
    effective = weight * route
    result["MUSIC_FAKE_PROB"] = sigmoid(
        logit(result.MUSIC_FAKE_PROB) + effective * music_delta
    )
    # Music evidence may alter File only when the existing CPS branch says
    # music is present. A continuous gate avoids a brittle 0.7 boundary.
    file_gate = np.clip(result.MUSIC_PRESENT_PROB.to_numpy(np.float64), 0, 1)
    result["FILE_FAKE_PROB"] = sigmoid(
        logit(result.FILE_FAKE_PROB)
        + effective * file_ratio * file_gate * music_delta
    )
    return result


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
        "--artifact-scores", type=Path,
        default=ROOT / "reports/artifactnet_pooling_v1/window_scores.csv",
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=ROOT / "reports/artifactnet_pooling_v1/v30_router",
    )
    parser.add_argument("--pools", nargs="+", default=["MEAN", "MAX", "Q75", "LME_T5"])
    parser.add_argument(
        "--weights", type=float, nargs="+",
        default=[.01, .025, .05, .075, .10, .15, .20, .30],
    )
    parser.add_argument("--file-ratios", type=float, nargs="+", default=[0, .25, .50, 1.0])
    args = parser.parse_args()
    v29 = pd.read_csv(args.v29_predictions, dtype={"ID": str})
    joint = pd.read_csv(args.joint_predictions, dtype={"ID": str})
    artifact = pd.read_csv(args.artifact_scores, dtype={"ID": str})
    banks = {
        name: reconstruct_v30(v29, joint, name)
        for name in TRUTHS
    }

    rows: list[dict] = []
    outputs: dict[tuple, tuple[pd.DataFrame, pd.DataFrame]] = {}
    for name, (truth, baseline) in banks.items():
        block = (
            artifact.loc[artifact.DATASET.eq(ARTIFACT_DATASETS[name])]
            .set_index("ID").loc[truth.index]
        )
        baseline_key = ("baseline", "none", "MEDIAN", 0.0, 0.0)
        rows.append({
            "DATASET": name, "MODE": "baseline", "ROUTE": "none",
            "POOL": "MEDIAN", "WEIGHT": 0.0, "FILE_RATIO": 0.0,
            **score_frame(truth.join(baseline)),
        })
        outputs[(name, *baseline_key)] = (truth, baseline)
        for mode in ("moe", "residual"):
            for policy in ("global", "hard_nonphone", "soft_nonphone"):
                route = route_values(name, truth.index, policy)
                for pool_name in args.pools:
                    pool = block[pool_name].to_numpy(np.float64)
                    median = block.MEDIAN.to_numpy(np.float64)
                    for weight in args.weights:
                        for file_ratio in args.file_ratios:
                            prediction = apply_candidate(
                                baseline, pool, median, route, mode,
                                weight, file_ratio,
                            )
                            key = (mode, policy, pool_name, weight, file_ratio)
                            rows.append({
                                "DATASET": name, "MODE": mode, "ROUTE": policy,
                                "POOL": pool_name, "WEIGHT": weight,
                                "FILE_RATIO": file_ratio,
                                **score_frame(truth.join(prediction)),
                            })
                            outputs[(name, *key)] = (truth, prediction)

    candidates = sorted({key[1:] for key in outputs if key[0] == "factorial"})
    for key in candidates:
        truths, predictions = [], []
        for name in ("factorial", "phone"):
            truth, prediction = outputs[(name, *key)]
            truth, prediction = truth.copy(), prediction.copy()
            truth.index = name + "_" + truth.index
            prediction.index = truth.index
            truths.append(truth); predictions.append(prediction)
        rows.append({
            "DATASET": "factorial_plus_phone", "MODE": key[0],
            "ROUTE": key[1], "POOL": key[2], "WEIGHT": key[3],
            "FILE_RATIO": key[4],
            **score_frame(pd.concat(truths).join(pd.concat(predictions))),
        })

    result = pd.DataFrame(rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output_dir / "sweep.csv", index=False)
    baseline = result.loc[result.MODE.eq("baseline")].set_index("DATASET").ADS
    scored = result.loc[~result.MODE.eq("baseline")].copy()
    scored["ADS_DELTA"] = scored.apply(
        lambda row: row.ADS - baseline.loc[row.DATASET], axis=1
    )
    summary = (
        scored.pivot_table(
            index=["MODE", "ROUTE", "POOL", "WEIGHT", "FILE_RATIO"],
            columns="DATASET", values="ADS_DELTA",
        )
        .reset_index()
    )
    domains = ["dev", "factorial", "phone", "yue"]
    summary["MIN_AUDIT_DELTA"] = summary[domains].min(axis=1)
    summary["MEAN_AUDIT_DELTA"] = summary[domains].mean(axis=1)
    summary = summary.sort_values(
        ["MIN_AUDIT_DELTA", "MEAN_AUDIT_DELTA"], ascending=False
    )
    summary.to_csv(args.output_dir / "summary.csv", index=False)
    print("Baseline ADS")
    print(baseline.to_string())
    print("\nBest worst-domain candidates (ADS delta)")
    print(summary.head(30).to_string(index=False))


if __name__ == "__main__":
    main()
