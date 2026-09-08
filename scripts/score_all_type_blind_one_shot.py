#!/usr/bin/env python3
"""Score one frozen candidate against one anchor on an all-type blind bank.

The acceptance rule is intentionally fixed in code before the v9 truth is
opened.  It rewards an overall ADS gain while rejecting a candidate that buys
that gain with a large task, channel, or component-topology regression.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve


TASKS = ("FILE", "VOICE", "MUSIC")
PROBABILITY_COLUMNS = tuple(f"{task}_FAKE_PROB" for task in TASKS) + (
    "VOICE_PRESENT_PROB", "MUSIC_PRESENT_PROB",
)
LABEL_COLUMNS = tuple(f"{task}_FAKE" for task in TASKS) + (
    "VOICE_PRESENT", "MUSIC_PRESENT",
)
ADS_WEIGHTS = {"FILE": 0.5, "VOICE": 0.2, "MUSIC": 0.3}
MINIMUM_OVERALL_ADS_GAIN = 0.0025
MAXIMUM_TASK_EER_REGRESSION = 0.025
MAXIMUM_GROUP_ADS_REGRESSION = 0.035


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def official_eer(labels, scores) -> float:
    labels = np.asarray(labels, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    if np.unique(labels).size != 2:
        return float("nan")
    fpr, tpr, _ = roc_curve(
        labels, scores, pos_label=1, drop_intermediate=False,
    )
    fnr = 1 - tpr
    index = int(np.argmin(np.abs(fpr - fnr)))
    return float((fpr[index] + fnr[index]) / 2)


def auc(labels, scores) -> float:
    labels = np.asarray(labels, dtype=np.int64)
    if np.unique(labels).size != 2:
        return float("nan")
    return float(roc_auc_score(labels, np.asarray(scores, dtype=np.float64)))


def validate_and_align(truth_path: Path, prediction_path: Path) -> pd.DataFrame:
    truth = pd.read_csv(truth_path, dtype={"ID": str})
    prediction = pd.read_csv(prediction_path, dtype={"ID": str})
    required_truth = {"ID", "CHANNEL", "MIX_MODE", *LABEL_COLUMNS}
    required_prediction = {"ID", *PROBABILITY_COLUMNS}
    if missing := required_truth.difference(truth):
        raise ValueError(f"truth misses columns: {sorted(missing)}")
    if missing := required_prediction.difference(prediction):
        raise ValueError(f"prediction misses columns: {sorted(missing)}")
    if truth.ID.duplicated().any() or prediction.ID.duplicated().any():
        raise ValueError("truth and prediction IDs must be unique")
    if set(truth.ID) != set(prediction.ID):
        raise ValueError("truth and prediction ID sets differ")
    # Presence and File labels are defined for every file.  Component fake
    # labels are intentionally undefined when that component is absent.
    for column in ("VOICE_PRESENT", "MUSIC_PRESENT", "FILE_FAKE"):
        truth[column] = pd.to_numeric(truth[column], errors="raise")
        if not truth[column].isin((0, 1)).all():
            raise ValueError(f"{column} is not binary")
        truth[column] = truth[column].astype(np.int64)
    for task in ("VOICE", "MUSIC"):
        column = f"{task}_FAKE"
        presence = truth[f"{task}_PRESENT"].eq(1)
        values = pd.to_numeric(truth[column], errors="coerce")
        if values.loc[presence].isna().any() or not values.loc[presence].isin((0, 1)).all():
            raise ValueError(f"{column} is not binary where {task} is present")
        if values.loc[~presence].notna().any() and not values.loc[~presence].isin((0, 1)).all():
            raise ValueError(f"{column} has an invalid absent-component label")
        truth[column] = values.fillna(0).astype(np.int64)
    if not truth.FILE_FAKE.equals(truth.VOICE_FAKE | truth.MUSIC_FAKE):
        raise ValueError("FILE_FAKE must equal VOICE_FAKE OR MUSIC_FAKE")
    aligned = prediction.set_index("ID").loc[truth.ID].reset_index()
    for column in PROBABILITY_COLUMNS:
        aligned[column] = pd.to_numeric(aligned[column], errors="raise")
        values = aligned[column].to_numpy(np.float64)
        if not np.isfinite(values).all() or np.any((values < 0) | (values > 1)):
            raise ValueError(f"{column} must contain finite probabilities")
    frame = truth.copy().reset_index(drop=True)
    for column in PROBABILITY_COLUMNS:
        frame[column] = aligned[column].to_numpy()
    frame["AUDIO_TYPE"] = np.select(
        (
            frame.VOICE_PRESENT.eq(1) & frame.MUSIC_PRESENT.eq(0),
            frame.VOICE_PRESENT.eq(0) & frame.MUSIC_PRESENT.eq(1),
            frame.VOICE_PRESENT.eq(1) & frame.MUSIC_PRESENT.eq(1),
        ),
        ("voice_only", "music_only", "mixed"),
        default="neither",
    )
    frame["PHONE"] = np.where(frame.CHANNEL.eq("clean"), "clean", "phone")
    return frame


def metrics(frame: pd.DataFrame) -> dict[str, float | int]:
    result: dict[str, float | int] = {"N": len(frame)}
    available_ads = 0.0
    available_weight = 0.0
    for task in TASKS:
        selected = frame
        if task != "FILE":
            selected = frame.loc[frame[f"{task}_PRESENT"].eq(1)]
        eer = official_eer(
            selected[f"{task}_FAKE"], selected[f"{task}_FAKE_PROB"],
        )
        result[f"{task}_N"] = len(selected)
        result[f"{task}_EER"] = eer
        if np.isfinite(eer):
            available_ads += ADS_WEIGHTS[task] * (1 - eer)
            available_weight += ADS_WEIGHTS[task]
    result["AVAILABLE_ADS"] = (
        float(available_ads / available_weight)
        if available_weight else float("nan")
    )
    voice_auc = auc(frame.VOICE_PRESENT, frame.VOICE_PRESENT_PROB)
    music_auc = auc(frame.MUSIC_PRESENT, frame.MUSIC_PRESENT_PROB)
    result["VOICE_PRESENCE_AUC"] = voice_auc
    result["MUSIC_PRESENCE_AUC"] = music_auc
    result["CPS"] = (
        float(0.5 * (voice_auc + music_auc))
        if np.isfinite(voice_auc) and np.isfinite(music_auc)
        else float("nan")
    )
    return result


def grouped(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    axes = {
        "CHANNEL": "CHANNEL",
        "PHONE": "PHONE",
        "AUDIO_TYPE": "AUDIO_TYPE",
        "MIX_MODE": "MIX_MODE",
    }
    for axis, column in axes.items():
        for value, block in frame.groupby(column, sort=True, dropna=False):
            rows.append({"AXIS": axis, "GROUP": str(value), **metrics(block)})
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--truth", type=Path, required=True)
    parser.add_argument("--anchor", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite {args.output_dir}")

    anchor = validate_and_align(args.truth, args.anchor)
    candidate = validate_and_align(args.truth, args.candidate)
    if not anchor[["ID", *LABEL_COLUMNS]].equals(
        candidate[["ID", *LABEL_COLUMNS]]
    ):
        raise ValueError("anchor and candidate did not align to identical truth")

    anchor_overall, candidate_overall = metrics(anchor), metrics(candidate)
    anchor_groups, candidate_groups = grouped(anchor), grouped(candidate)
    keys = ["AXIS", "GROUP"]
    comparison = anchor_groups.merge(
        candidate_groups, on=keys, suffixes=("_ANCHOR", "_CANDIDATE"),
        validate="one_to_one",
    )
    comparison["ADS_DELTA"] = (
        comparison.AVAILABLE_ADS_CANDIDATE
        - comparison.AVAILABLE_ADS_ANCHOR
    )
    task_regressions = {
        task: float(
            candidate_overall[f"{task}_EER"]
            - anchor_overall[f"{task}_EER"]
        )
        for task in TASKS
    }
    finite_group_deltas = comparison.ADS_DELTA.dropna()
    overall_gain = float(
        candidate_overall["AVAILABLE_ADS"] - anchor_overall["AVAILABLE_ADS"]
    )
    accepted = bool(
        overall_gain >= MINIMUM_OVERALL_ADS_GAIN
        and max(task_regressions.values()) <= MAXIMUM_TASK_EER_REGRESSION
        and (
            finite_group_deltas.empty
            or float(finite_group_deltas.min()) >= -MAXIMUM_GROUP_ADS_REGRESSION
        )
    )
    summary = {
        "accepted": accepted,
        "fixed_gate": {
            "minimum_overall_ads_gain": MINIMUM_OVERALL_ADS_GAIN,
            "maximum_task_eer_regression": MAXIMUM_TASK_EER_REGRESSION,
            "maximum_group_ads_regression": MAXIMUM_GROUP_ADS_REGRESSION,
        },
        "anchor": anchor_overall,
        "candidate": candidate_overall,
        "overall_ads_delta": overall_gain,
        "task_eer_regressions": task_regressions,
        "worst_group_ads_delta": (
            float(finite_group_deltas.min())
            if not finite_group_deltas.empty else None
        ),
        "hashes": {
            "truth": sha256(args.truth),
            "anchor": sha256(args.anchor),
            "candidate": sha256(args.candidate),
        },
    }
    args.output_dir.mkdir(parents=True)
    comparison.to_csv(args.output_dir / "groups.csv", index=False)
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, allow_nan=True))


if __name__ == "__main__":
    main()
