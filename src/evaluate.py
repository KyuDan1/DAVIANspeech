"""Local scoring for a submission CSV against a labelled ground-truth CSV.

Reports ROC-AUC and EER for every probability column that has a matching
label column, so a run can be judged offline before spending a leaderboard
submission.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

PAIRS = [
    ("FILE_FAKE_PROB", "FILE_FAKE"),
    ("VOICE_FAKE_PROB", "VOICE_FAKE"),
    ("MUSIC_FAKE_PROB", "MUSIC_FAKE"),
    ("VOICE_PRESENT_PROB", "VOICE_PRESENT"),
    ("MUSIC_PRESENT_PROB", "MUSIC_PRESENT"),
]


def equal_error_rate(y_true: np.ndarray, scores: np.ndarray) -> float:
    """EER via the ASVspoof DET-curve formulation."""
    target = scores[y_true == 1]
    nontarget = scores[y_true == 0]
    if target.size == 0 or nontarget.size == 0:
        return float("nan")

    all_scores = np.concatenate([target, nontarget])
    labels = np.concatenate([np.ones(target.size), np.zeros(nontarget.size)])
    order = np.argsort(all_scores, kind="mergesort")
    labels = labels[order]

    tar_cumulative = np.cumsum(labels)
    non_cumulative = nontarget.size - (np.arange(1, all_scores.size + 1) - tar_cumulative)
    frr = np.concatenate([[0], tar_cumulative / target.size])
    far = np.concatenate([[1], non_cumulative / nontarget.size])
    index = np.nanargmin(np.abs(frr - far))
    return float(np.mean([frr[index], far[index]]))


def evaluate(submission_path: Path, truth_path: Path) -> pd.DataFrame:
    submission = pd.read_csv(submission_path)
    truth = pd.read_csv(truth_path)
    merged = submission.merge(truth, on="ID", suffixes=("", "_true"))
    if merged.empty:
        raise ValueError("No overlapping IDs between submission and ground truth")
    if len(merged) < len(submission):
        print(f"warning: {len(submission) - len(merged)} submission rows lack labels")

    records = []
    for score_column, label_column in PAIRS:
        if score_column not in merged or label_column not in merged:
            continue
        labels = merged[label_column].to_numpy()
        scores = merged[score_column].to_numpy(dtype=float)
        keep = ~pd.isna(labels)
        labels, scores = labels[keep].astype(int), scores[keep]
        if len(np.unique(labels)) < 2:
            continue
        records.append({
            "column": score_column,
            "n": len(labels),
            "positives": int(labels.sum()),
            "auc": roc_auc_score(labels, scores),
            "eer": equal_error_rate(labels, scores),
        })

    if not records:
        raise ValueError(
            f"No comparable column pairs. Submission has {list(submission.columns)}, "
            f"truth has {list(truth.columns)}"
        )

    table = pd.DataFrame(records).set_index("column")
    table.loc["MEAN"] = {
        "n": table["n"].max(),
        "positives": np.nan,
        "auc": table["auc"].mean(),
        "eer": table["eer"].mean(),
    }
    return table


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("submission", type=Path)
    parser.add_argument("truth", type=Path)
    args = parser.parse_args()
    print(evaluate(args.submission, args.truth).round(5).to_string())


if __name__ == "__main__":
    main()
