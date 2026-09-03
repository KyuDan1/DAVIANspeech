"""Evaluate one frozen telephone router and threshold without retuning."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve


def eer(labels, scores):
    fpr, tpr, _ = roc_curve(labels, scores, drop_intermediate=False)
    fnr = 1 - tpr
    index = np.argmin(np.abs(fpr - fnr))
    return float((fpr[index] + fnr[index]) / 2)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    archive = np.load(args.features)
    model = joblib.load(args.model)
    selection = json.loads(args.selection.read_text("utf-8"))["selected"]
    threshold = float(selection["threshold"])
    labels = archive["label"].astype(int)
    scores = model.predict_proba(archive["features"])[:, 1]
    predictions = scores >= threshold
    records = []
    for group, values in [
        ("ALL", np.repeat("ALL", len(labels))),
        ("VARIANT", archive["variant"]),
        ("DATASET", archive["dataset"]),
        ("AUDIO_TYPE", archive["audio_type"]),
    ]:
        for value in np.unique(values):
            selected = values == value
            y, score, prediction = labels[selected], scores[selected], predictions[selected]
            records.append({
                "GROUP": group, "VALUE": str(value), "N": int(selected.sum()),
                "AUC": roc_auc_score(y, score) if len(np.unique(y)) == 2 else np.nan,
                "EER": eer(y, score) if len(np.unique(y)) == 2 else np.nan,
                "THRESHOLD": threshold,
                "TPR": prediction[y == 1].mean() if np.any(y == 1) else np.nan,
                "FPR": prediction[y == 0].mean() if np.any(y == 0) else np.nan,
            })
    table = pd.DataFrame(records)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.output, index=False)
    print(table.round(6).to_string(index=False))


if __name__ == "__main__":
    main()
