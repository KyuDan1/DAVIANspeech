#!/usr/bin/env python3
"""Compare v56 direct/fused scores to v50 and v55 on authorised development."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from eat_large_aasist_data import partition_paths  # noqa: E402
from evaluate_diagnostic import official_eer, score_frame  # noqa: E402


PROBABILITY_COLUMNS = (
    "FILE_FAKE_PROB", "VOICE_FAKE_PROB", "MUSIC_FAKE_PROB",
)


def logit(values) -> np.ndarray:
    values = np.clip(np.asarray(values, dtype=np.float64), 1e-6, 1 - 1e-6)
    return np.log(values) - np.log1p(-values)


def sigmoid(values) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    return np.exp(-np.logaddexp(0.0, -values))


def load_truth(config: Path, dataset: str) -> pd.DataFrame:
    matches = [
        path for path in partition_paths(config, "development")
        if path.parent.name == dataset
    ]
    if len(matches) != 1:
        raise ValueError(f"expected one development manifest for {dataset}: {matches}")
    return pd.read_csv(matches[0], dtype={"ID": str})


def align(path: Path, truth: pd.DataFrame, dataset: str | None = None) -> pd.DataFrame:
    frame = pd.read_csv(path, dtype={"ID": str})
    if dataset is not None and "DATASET" in frame:
        frame = frame.loc[frame.DATASET.eq(dataset)].copy()
    frame = frame.loc[frame.ID.isin(set(truth.ID))].copy()
    if frame.ID.duplicated().any() or set(frame.ID) != set(truth.ID):
        raise ValueError(f"prediction IDs differ from truth: {path}")
    return frame.set_index("ID").loc[truth.ID]


def full_development_music_sweep(
    config: Path, v56_path: Path, anchor_root: Path, output_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """File=0 diagnostic sweep over exact-v47 Music on every dev domain."""
    expert_all = pd.read_csv(v56_path, dtype={"ID": str})
    if "DATASET" not in expert_all:
        raise ValueError("full development prediction needs a DATASET column")
    weights = (0., .05, .10, .15, .20, .30)
    domain_rows, sweep_rows = [], []
    by_weight: dict[float, list[tuple[np.ndarray, np.ndarray]]] = {
        value: [] for value in weights
    }
    domain_eers: dict[float, list[float]] = {value: [] for value in weights}
    for truth_path in partition_paths(config, "development"):
        dataset = truth_path.parent.name
        truth = pd.read_csv(truth_path, dtype={"ID": str})
        selected = truth.MUSIC_PRESENT.eq(1) & truth.MUSIC_FAKE.notna()
        labels = truth.loc[selected, "MUSIC_FAKE"].astype(int).to_numpy()
        if len(np.unique(labels)) < 2:
            continue
        expert = align(v56_path, truth, dataset)
        anchor_path = anchor_root / dataset / "exact_v47_predictions.csv"
        anchor = align(anchor_path, truth)
        selected_ids = truth.loc[selected, "ID"]
        for weight in weights:
            score = sigmoid(
                (1 - weight) * logit(anchor.loc[selected_ids, "MUSIC_FAKE_PROB"])
                + weight * logit(expert.loc[selected_ids, "MUSIC_FAKE_PROB"])
            )
            eer = official_eer(labels, score)
            domain_rows.append({
                "DATASET": dataset, "MUSIC_WEIGHT": weight,
                "MUSIC_EER": eer, "N": int(selected.sum()),
            })
            domain_eers[weight].append(eer)
            by_weight[weight].append((labels, score))
    for weight in weights:
        labels = np.concatenate([item[0] for item in by_weight[weight]])
        scores = np.concatenate([item[1] for item in by_weight[weight]])
        eers = np.asarray(domain_eers[weight], dtype=np.float64)
        pooled = official_eer(labels, scores)
        selection = .50 * (1 - pooled) + .25 * (1 - eers.mean()) + .25 * (
            1 - eers.max()
        )
        sweep_rows.append({
            "MUSIC_WEIGHT": weight, "POOLED_MUSIC_EER": pooled,
            "MEAN_DOMAIN_MUSIC_EER": eers.mean(),
            "WORST_DOMAIN_MUSIC_EER": eers.max(), "SELECTION": selection,
        })
    domain = pd.DataFrame(domain_rows)
    sweep = pd.DataFrame(sweep_rows).sort_values(
        ["SELECTION", "MUSIC_WEIGHT"], ascending=[False, True]
    )
    domain.to_csv(output_dir / "full_development_music_by_domain.csv", index=False)
    sweep.to_csv(output_dir / "full_development_music_only_sweep.csv", index=False)
    return domain, sweep


def fuse(
    anchor: pd.DataFrame, expert: pd.DataFrame,
    file_weight: float, music_weight: float,
) -> pd.DataFrame:
    result = anchor.copy()
    result["FILE_FAKE_PROB"] = sigmoid(
        (1 - file_weight) * logit(anchor.FILE_FAKE_PROB)
        + file_weight * logit(expert.FILE_FAKE_PROB)
    )
    result["MUSIC_FAKE_PROB"] = sigmoid(
        (1 - music_weight) * logit(anchor.MUSIC_FAKE_PROB)
        + music_weight * logit(expert.MUSIC_FAKE_PROB)
    )
    return result


def task_eers(truth: pd.DataFrame, prediction: pd.DataFrame) -> dict[str, float]:
    scored = truth.set_index("ID").join(prediction[list(PROBABILITY_COLUMNS)])
    return score_frame(scored)


def subgroup_rows(
    truth: pd.DataFrame, methods: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    rows = []
    columns = [
        column for column in (
            "AUDIO_TYPE", "CHANNEL", "EVAL_CELL", "COMPONENT_CASE", "MIX_MODE",
        ) if column in truth
    ]
    for column in columns:
        for value, block in truth.groupby(column, dropna=False):
            for method, prediction in methods.items():
                current = prediction.loc[block.ID]
                row: dict[str, object] = {
                    "GROUP_COLUMN": column, "GROUP": str(value),
                    "METHOD": method, "N": len(block),
                }
                for task, presence in (
                    ("FILE", np.ones(len(block), dtype=bool)),
                    ("VOICE", block.VOICE_PRESENT.eq(1).to_numpy()),
                    ("MUSIC", block.MUSIC_PRESENT.eq(1).to_numpy()),
                ):
                    label = block[f"{task}_FAKE"].to_numpy()[presence]
                    score = current[f"{task}_FAKE_PROB"].to_numpy()[presence]
                    finite = pd.notna(label) & np.isfinite(score)
                    row[f"{task}_EER"] = (
                        official_eer(label[finite].astype(int), score[finite])
                        if finite.any() and len(np.unique(label[finite])) == 2
                        else np.nan
                    )
                rows.append(row)
    return pd.DataFrame(rows)


def cell_diagnostics(
    truth: pd.DataFrame, methods: dict[str, pd.DataFrame],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return valid cell contrasts and cell errors at each task's EER threshold.

    A single RR/RF/FR/FF cell has a constant target and therefore no EER.  The
    contrasts below keep the other component fixed, while cell error rates use
    the global EER operating point.
    """
    cell_column = "COMPONENT_CASE" if "COMPONENT_CASE" in truth else None
    if cell_column is None:
        return pd.DataFrame(), pd.DataFrame()
    contrasts = (
        ("FILE", "RR", "RF"), ("FILE", "RR", "FR"),
        ("FILE", "RR", "FF"),
        ("VOICE", "RR", "FR"), ("VOICE", "RF", "FF"),
        ("MUSIC", "RR", "RF"), ("MUSIC", "FR", "FF"),
    )
    contrast_rows, error_rows = [], []
    indexed_truth = truth.set_index("ID")
    for method, prediction in methods.items():
        for task in ("FILE", "VOICE", "MUSIC"):
            selected = np.ones(len(truth), dtype=bool) if task == "FILE" else (
                truth[f"{task}_PRESENT"].eq(1).to_numpy()
            )
            labels = truth.loc[selected, f"{task}_FAKE"].astype(int).to_numpy()
            scores = prediction.loc[
                truth.loc[selected, "ID"], f"{task}_FAKE_PROB"
            ].to_numpy()
            fpr, tpr, thresholds = roc_curve(
                labels, scores, pos_label=1, drop_intermediate=False
            )
            threshold = float(thresholds[np.argmin(np.abs(fpr - (1 - tpr)))])
            for cell, block in truth.loc[selected].groupby(cell_column):
                cell_labels = block[f"{task}_FAKE"].astype(int).to_numpy()
                cell_scores = prediction.loc[
                    block.ID, f"{task}_FAKE_PROB"
                ].to_numpy()
                cell_predictions = cell_scores >= threshold
                error_rows.append({
                    "METHOD": method, "TASK": task, "CELL": str(cell),
                    "N": len(block), "LABEL": int(cell_labels[0]),
                    "GLOBAL_EER_THRESHOLD": threshold,
                    "CELL_ERROR_RATE": float(np.mean(cell_predictions != cell_labels)),
                    "SCORE_MEAN": float(cell_scores.mean()),
                    "SCORE_MEDIAN": float(np.median(cell_scores)),
                })
        for task, negative, positive in contrasts:
            block = truth.loc[truth[cell_column].isin((negative, positive))].copy()
            if task != "FILE":
                block = block.loc[block[f"{task}_PRESENT"].eq(1)]
            labels = block[f"{task}_FAKE"].astype(int).to_numpy()
            scores = prediction.loc[block.ID, f"{task}_FAKE_PROB"].to_numpy()
            contrast_rows.append({
                "METHOD": method, "TASK": task,
                "CONTRAST": f"{negative}_vs_{positive}", "N": len(block),
                "EER": official_eer(labels, scores),
            })
    return pd.DataFrame(contrast_rows), pd.DataFrame(error_rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--partition-config", type=Path,
        default=ROOT / "configs/data_partitions.yaml",
    )
    parser.add_argument("--dataset", default="codec_mixed_dev_v4")
    parser.add_argument("--v56-prediction", type=Path, required=True)
    parser.add_argument(
        "--v50-prediction", type=Path,
        default=ROOT / "reports/v50_candidate_codec_dev_v4/predictions.csv",
    )
    parser.add_argument(
        "--v55-prediction", type=Path,
        default=ROOT / "reports/eat_music_adapter_v55/seed00/dev_predictions.csv",
    )
    parser.add_argument("--v55-weight", type=float, default=.15)
    parser.add_argument(
        "--v47-cache-root", type=Path,
        default=(ROOT / "reports/v47_anchor_cache_train_dev_v2/cache/datasets"),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        parser.error(f"refusing to overwrite {args.output_dir}")

    truth = load_truth(args.partition_config, args.dataset)
    v50 = align(args.v50_prediction, truth, args.dataset)
    v56 = align(args.v56_prediction, truth, args.dataset)
    v55_raw = align(args.v55_prediction, truth)
    v55 = v50.copy()
    v55["MUSIC_FAKE_PROB"] = sigmoid(
        (1 - args.v55_weight) * logit(v50.MUSIC_FAKE_PROB)
        + args.v55_weight * logit(v55_raw.EAT_ADAPTER_MUSIC_PROB)
    )

    sweep_rows = []
    weights = (0., .05, .10, .15, .20, .30)
    for file_weight in weights:
        for music_weight in weights:
            candidate = fuse(v50, v56, file_weight, music_weight)
            metric = task_eers(truth, candidate)
            quality = .625 * (1 - metric["FILE_EER"]) + .375 * (
                1 - metric["MUSIC_EER"]
            )
            sweep_rows.append({
                "FILE_WEIGHT": file_weight, "MUSIC_WEIGHT": music_weight,
                "FILE_EER": metric["FILE_EER"],
                "VOICE_EER": metric["VOICE_EER"],
                "MUSIC_EER": metric["MUSIC_EER"], "QUALITY": quality,
            })
    sweep = pd.DataFrame(sweep_rows).sort_values(
        ["QUALITY", "FILE_WEIGHT", "MUSIC_WEIGHT"],
        ascending=[False, True, True],
    )
    selected = sweep.iloc[0]
    v56_fused = fuse(
        v50, v56, float(selected.FILE_WEIGHT), float(selected.MUSIC_WEIGHT)
    )
    methods = {
        "v50": v50, "v55": v55,
        "v56_direct": v56, "v56_fused_dev_sweep": v56_fused,
    }
    overall = pd.DataFrame([
        {"METHOD": method, **task_eers(truth, prediction)}
        for method, prediction in methods.items()
    ])
    subgroups = subgroup_rows(truth, methods)
    contrasts, cell_errors = cell_diagnostics(truth, methods)
    args.output_dir.mkdir(parents=True)
    sweep.to_csv(args.output_dir / "fusion_sweep.csv", index=False)
    overall.to_csv(args.output_dir / "overall.csv", index=False)
    subgroups.to_csv(args.output_dir / "subgroups.csv", index=False)
    contrasts.to_csv(args.output_dir / "cell_contrast_eers.csv", index=False)
    cell_errors.to_csv(args.output_dir / "cell_error_rates.csv", index=False)
    development = pd.read_csv(args.v56_prediction, dtype={"ID": str})
    if "DATASET" in development:
        np.savez_compressed(
            args.output_dir / "raw_expert_logits.npz",
            datasets=development.DATASET.astype(str).to_numpy(),
            ids=development.ID.astype(str).to_numpy(),
            voice_logits=logit(development.VOICE_FAKE_PROB),
            music_logits=logit(development.MUSIC_FAKE_PROB),
            file_logits=logit(development.FILE_FAKE_PROB),
        )
        _, full_sweep = full_development_music_sweep(
            args.partition_config, args.v56_prediction,
            args.v47_cache_root, args.output_dir,
        )
        full_selected = full_sweep.iloc[0]
    else:
        full_selected = None
    summary = {
        "dataset": args.dataset,
        "selected_file_weight": float(selected.FILE_WEIGHT),
        "selected_music_weight": float(selected.MUSIC_WEIGHT),
        "selected_quality": float(selected.QUALITY),
        "note": "development sweep is diagnostic, not a frozen deployment weight",
    }
    if full_selected is not None:
        summary.update({
            "full_dev_music_only_weight": float(full_selected.MUSIC_WEIGHT),
            "full_dev_pooled_music_eer": float(full_selected.POOLED_MUSIC_EER),
            "full_dev_worst_music_eer": float(full_selected.WORST_DOMAIN_MUSIC_EER),
        })
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(overall.to_string(index=False))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
