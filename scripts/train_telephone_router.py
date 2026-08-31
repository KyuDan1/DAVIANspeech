"""Compare low-cost telephone routers on source- and codec-disjoint data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from telephone_router import (  # noqa: E402
    GLOBAL_BAND_FEATURE_INDEX,
    ROBUST_BAND_FEATURE_INDEX,
)
from telephone_channel import POSITIVE_VARIANTS  # noqa: E402


def load(path: Path) -> dict[str, np.ndarray]:
    archive = np.load(path)
    return {name: archive[name] for name in archive.files}


def load_many(paths: list[Path]) -> dict[str, np.ndarray]:
    archives = [load(path) for path in paths]
    names = set.intersection(*(set(archive) for archive in archives))
    combined = {
        name: np.concatenate([archive[name] for archive in archives]) for name in names
    }
    # Older feature caches may have treated wideband call codecs as positives.
    # The router target is instead whether to invoke a narrowband expert.
    combined["label"] = np.isin(combined["variant"], POSITIVE_VARIANTS).astype(np.int8)
    return combined


def eer(labels, scores) -> float:
    fpr, tpr, _ = roc_curve(labels, scores, drop_intermediate=False)
    fnr = 1 - tpr
    index = np.argmin(np.abs(fpr - fnr))
    return float((fpr[index] + fnr[index]) / 2)


def low_fpr_threshold(labels, scores, target=.005) -> float:
    negatives = np.sort(np.asarray(scores)[np.asarray(labels) == 0])
    allowed = int(np.floor(target * len(negatives)))
    index = max(0, len(negatives) - allowed - 1)
    return float(np.nextafter(negatives[index], np.inf))


def evaluate(name: str, data: dict[str, np.ndarray], scores: np.ndarray,
             target_fpr=.005):
    labels = data["label"].astype(int)
    threshold = low_fpr_threshold(labels, scores, target_fpr)
    prediction = scores >= threshold
    records = [{
        "MODEL": name, "GROUP": "ALL", "VALUE": "ALL", "N": len(labels),
        "AUC": roc_auc_score(labels, scores), "EER": eer(labels, scores),
        "THRESHOLD": threshold,
        "TPR": prediction[labels == 1].mean(),
        "FPR": prediction[labels == 0].mean(),
    }]
    for group_name in ("variant", "dataset", "audio_type"):
        for value in np.unique(data[group_name]):
            selected = data[group_name] == value
            group_labels = labels[selected]
            group_scores = scores[selected]
            records.append({
                "MODEL": name, "GROUP": group_name.upper(), "VALUE": str(value),
                "N": int(selected.sum()),
                "AUC": roc_auc_score(group_labels, group_scores)
                    if len(np.unique(group_labels)) == 2 else np.nan,
                "EER": eer(group_labels, group_scores)
                    if len(np.unique(group_labels)) == 2 else np.nan,
                "THRESHOLD": threshold,
                "TPR": prediction[selected & (labels == 1)].mean()
                    if np.any(selected & (labels == 1)) else np.nan,
                "FPR": prediction[selected & (labels == 0)].mean()
                    if np.any(selected & (labels == 0)) else np.nan,
            })
    table = pd.DataFrame(records)
    positive = table[(table.GROUP == "VARIANT") & table.TPR.notna()]
    return table, {
        "name": name,
        "auc": float(records[0]["AUC"]),
        "eer": float(records[0]["EER"]),
        "tpr": float(records[0]["TPR"]),
        "fpr": float(records[0]["FPR"]),
        "threshold": threshold,
        "worst_variant_tpr": float(positive.TPR.min()),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, action="append", required=True)
    parser.add_argument("--development", type=Path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--target-fpr", type=float, default=.005)
    args = parser.parse_args()
    train, development = load_many(args.train), load_many(args.development)
    x_train, y_train = train["features"], train["label"].astype(int)
    x_dev = development["features"]

    candidates = {
        "global_band_ratio": None,
        "robust_frame_band": None,
        "logistic_c0.01": make_pipeline(
            StandardScaler(), LogisticRegression(
                C=.01, class_weight="balanced", max_iter=2_000
            )
        ),
        "logistic_c0.1": make_pipeline(
            StandardScaler(), LogisticRegression(
                C=.1, class_weight="balanced", max_iter=2_000
            )
        ),
        "extra_depth8_leaf5": ExtraTreesClassifier(
            n_estimators=400, max_depth=8, min_samples_leaf=5,
            max_features=.7, class_weight="balanced", n_jobs=-1,
            random_state=20260831,
        ),
        "extra_depth12_leaf3": ExtraTreesClassifier(
            n_estimators=400, max_depth=12, min_samples_leaf=3,
            max_features=.7, class_weight="balanced", n_jobs=-1,
            random_state=20260831,
        ),
        "hist_leaf15_l2": HistGradientBoostingClassifier(
            max_iter=250, max_leaf_nodes=15, min_samples_leaf=20,
            l2_regularization=1.0, learning_rate=.08, random_state=20260831,
        ),
        "hist_leaf31_l2": HistGradientBoostingClassifier(
            max_iter=250, max_leaf_nodes=31, min_samples_leaf=20,
            l2_regularization=3.0, learning_rate=.08, random_state=20260831,
        ),
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    tables, summaries, fitted = [], [], {}
    baseline_names = {"global_band_ratio", "robust_frame_band"}
    score_by_name = {
        "global_band_ratio": -x_dev[:, GLOBAL_BAND_FEATURE_INDEX],
        "robust_frame_band": -x_dev[:, ROBUST_BAND_FEATURE_INDEX],
    }
    for name, model in candidates.items():
        if model is not None:
            model.fit(x_train, y_train)
            fitted[name] = model
            score_by_name[name] = model.predict_proba(x_dev)[:, 1]
        table, summary = evaluate(
            name, development, score_by_name[name], args.target_fpr
        )
        tables.append(table)
        summaries.append(summary)
        print(summary)

    # Prioritize the worst unseen-codec recall, then overall recall and AUC.
    selected = max(
        (row for row in summaries if row["name"] not in baseline_names),
        key=lambda row: (row["worst_variant_tpr"], row["tpr"], row["auc"]),
    )
    selected_model = fitted[selected["name"]]
    joblib.dump(selected_model, args.output_dir / "telephone-router.joblib")
    if selected["name"].startswith("logistic_"):
        scaler = selected_model.named_steps["standardscaler"]
        classifier = selected_model.named_steps["logisticregression"]
        np.savez_compressed(
            args.output_dir / "telephone-router.npz",
            format_version=np.asarray(1, dtype=np.int32),
            mean=scaler.mean_.astype(np.float32),
            scale=scaler.scale_.astype(np.float32),
            weight=classifier.coef_[0].astype(np.float32),
            bias=np.asarray(classifier.intercept_[0], dtype=np.float32),
            threshold=np.asarray(selected["threshold"], dtype=np.float32),
        )
    pd.concat(tables).to_csv(args.output_dir / "development_metrics.csv", index=False)
    (args.output_dir / "selection.json").write_text(
        json.dumps({"selected": selected, "all": summaries}, indent=2),
        encoding="utf-8",
    )
    print("selected", selected)


if __name__ == "__main__":
    main()
