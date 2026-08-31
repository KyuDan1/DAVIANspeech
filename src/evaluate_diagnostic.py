"""Competition-exact scoring plus diagnostic breakdowns.

The truth CSV may contain optional diagnostic columns such as ``AUDIO_TYPE``,
``SOURCE``, ``CODEC`` and ``CONDITION``. Scores are reported for the complete
set and for every sufficiently populated value of those columns.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve

PREDICTION_COLUMNS = [
    "FILE_FAKE_PROB", "VOICE_FAKE_PROB", "MUSIC_FAKE_PROB",
    "VOICE_PRESENT_PROB", "MUSIC_PRESENT_PROB",
]
LABEL_COLUMNS = [
    "FILE_FAKE", "VOICE_FAKE", "MUSIC_FAKE", "VOICE_PRESENT", "MUSIC_PRESENT",
]
GROUP_COLUMNS = [
    "AUDIO_TYPE", "SOURCE", "GENERATOR", "CODEC", "CONDITION",
    "SPLIT", "CHANNEL", "FORMAT", "MIX_MODE", "VOICE_GENERATOR",
    "MUSIC_GENERATOR", "SNR_DB", "OVERLAP_FRACTION", "ORDER",
    "FILE_EXTENSION", "COMPONENT_CASE", "EVAL_CELL",
]


def official_eer(labels, scores) -> float:
    """EER matching the competition's published roc_curve implementation."""
    labels = np.asarray(labels, dtype=int)
    scores = np.asarray(scores, dtype=float)
    if len(np.unique(labels)) < 2:
        return float("nan")
    fpr, tpr, _ = roc_curve(labels, scores, pos_label=1, drop_intermediate=False)
    fnr = 1 - tpr
    idx = np.argmin(np.abs(fpr - fnr))
    return float((fpr[idx] + fnr[idx]) / 2)


def _metric(frame, score_column, label_column, presence_column=None, auc=False):
    if score_column not in frame or label_column not in frame:
        return float("nan"), 0
    selected = frame
    if presence_column is not None:
        selected = selected[selected[presence_column] == 1]
    selected = selected.dropna(subset=[score_column, label_column])
    if len(selected) == 0 or selected[label_column].nunique() < 2:
        return float("nan"), len(selected)
    labels = selected[label_column].astype(int)
    scores = selected[score_column].astype(float)
    value = roc_auc_score(labels, scores) if auc else official_eer(labels, scores)
    return float(value), len(selected)


def score_frame(frame: pd.DataFrame) -> dict[str, float | int]:
    file_eer, file_n = _metric(frame, "FILE_FAKE_PROB", "FILE_FAKE")
    voice_eer, voice_n = _metric(
        frame, "VOICE_FAKE_PROB", "VOICE_FAKE", "VOICE_PRESENT"
    )
    music_eer, music_n = _metric(
        frame, "MUSIC_FAKE_PROB", "MUSIC_FAKE", "MUSIC_PRESENT"
    )
    voice_auc, _ = _metric(
        frame, "VOICE_PRESENT_PROB", "VOICE_PRESENT", auc=True
    )
    music_auc, _ = _metric(
        frame, "MUSIC_PRESENT_PROB", "MUSIC_PRESENT", auc=True
    )
    ads = 0.5 * (1 - file_eer) + 0.2 * (1 - voice_eer) + 0.3 * (1 - music_eer)
    cps = 0.5 * voice_auc + 0.5 * music_auc
    return {
        "N": len(frame), "FILE_N": file_n, "VOICE_N": voice_n, "MUSIC_N": music_n,
        "FILE_EER": file_eer, "VOICE_EER": voice_eer, "MUSIC_EER": music_eer,
        "VOICE_AUC": voice_auc, "MUSIC_AUC": music_auc,
        "ADS": ads, "CPS": cps, "SCORE": 0.9 * ads + 0.1 * cps,
    }


def evaluate_diagnostic(prediction_path: Path, truth_path: Path, min_group_size=10):
    prediction = pd.read_csv(prediction_path, dtype={"ID": str})
    truth = pd.read_csv(truth_path, dtype={"ID": str})
    if prediction["ID"].duplicated().any() or truth["ID"].duplicated().any():
        raise ValueError("IDs must be unique in both files")
    missing_predictions = sorted(set(truth["ID"]) - set(prediction["ID"]))
    if missing_predictions:
        raise ValueError(f"Missing predictions for {len(missing_predictions)} IDs")
    frame = truth.merge(prediction[["ID", *PREDICTION_COLUMNS]], on="ID", validate="one_to_one")

    records = [{"GROUP": "ALL", "VALUE": "ALL", **score_frame(frame)}]
    for column in GROUP_COLUMNS:
        if column not in frame:
            continue
        for value, group in frame.groupby(column, dropna=False):
            if len(group) >= min_group_size:
                records.append({"GROUP": column, "VALUE": str(value), **score_frame(group)})
    return pd.DataFrame(records)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("prediction", type=Path)
    parser.add_argument("truth", type=Path)
    parser.add_argument("--min-group-size", type=int, default=10)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    table = evaluate_diagnostic(args.prediction, args.truth, args.min_group_size)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(args.output, index=False)
    print(table.round(5).to_string(index=False))


if __name__ == "__main__":
    main()
