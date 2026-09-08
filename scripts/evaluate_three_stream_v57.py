#!/usr/bin/env python3
"""Evaluate frozen v57 candidates only on its declared development subset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import roc_curve


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from evaluate_diagnostic import PREDICTION_COLUMNS, score_frame  # noqa: E402
from train_three_stream_anchor_residual import (  # noqa: E402
    authorized_partitions,
    normalized_available_ads,
)


AXIS_COLUMNS = (
    "CELL_V57", "LAYOUT_V57", "CHANNEL_V57", "AUDIO_TYPE",
    "VOICE_GENERATOR_V57", "MUSIC_GENERATOR_V57",
)
DECISION_AXES = ("CELL_V57", "LAYOUT_V57", "CHANNEL_V57")
RESIDUAL_COLUMNS = (
    "VOICE_LOGIT_RESIDUAL", "MUSIC_LOGIT_RESIDUAL",
    "DIRECT_FILE_LOGIT_RESIDUAL",
)
COMPARISON_METRICS = (
    "FILE_EER", "VOICE_EER", "MUSIC_EER", "ADS", "CPS",
    "NORMALIZED_AVAILABLE_ADS",
)


def _coalesce(frame: pd.DataFrame, columns: tuple[str, ...]) -> pd.Series:
    output = pd.Series("", index=frame.index, dtype=object)
    for column in columns:
        if column not in frame:
            continue
        values = frame[column].fillna("").astype(str).str.strip()
        selected = output.eq("") & values.ne("")
        output.loc[selected] = values.loc[selected]
    return output.replace("", "unknown")


def _component_generator(
    frame: pd.DataFrame,
    columns: tuple[str, ...],
    present_column: str,
    fake_column: str,
) -> pd.Series:
    result = pd.Series("", index=frame.index, dtype=object)
    for column in columns:
        if column not in frame:
            continue
        values = frame[column].fillna("").astype(str).str.strip()
        for index in frame.index[values.ne("")]:
            current = [item for item in result.loc[index].split("+") if item]
            value = values.loc[index]
            if value not in current:
                current.append(value)
            result.loc[index] = "+".join(sorted(current))
    present = pd.to_numeric(frame[present_column], errors="coerce").eq(1)
    fake = pd.to_numeric(frame[fake_column], errors="coerce").eq(1)
    result.loc[~present] = "absent"
    result.loc[present & result.eq("") & ~fake] = "real_unspecified"
    result.loc[present & result.eq("") & fake] = "unknown_fake"
    return result


def add_diagnostic_axes(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    voice_present = pd.to_numeric(result["VOICE_PRESENT"], errors="coerce").eq(1)
    music_present = pd.to_numeric(result["MUSIC_PRESENT"], errors="coerce").eq(1)
    voice_fake = pd.to_numeric(result["VOICE_FAKE"], errors="coerce").fillna(0).eq(1)
    music_fake = pd.to_numeric(result["MUSIC_FAKE"], errors="coerce").fillna(0).eq(1)
    result["CELL_V57"] = "not_mixed"
    mixed = voice_present & music_present
    result.loc[mixed, "CELL_V57"] = (
        np.where(voice_fake[mixed], "F", "R")
        + np.where(music_fake[mixed], "F", "R")
    )
    result["LAYOUT_V57"] = _coalesce(
        result,
        ("MIX_MODE", "LAYOUT", "CONDITION", "MUSIC_LAYOUT", "CONVERSATION_MODE"),
    )
    result["CHANNEL_V57"] = _coalesce(
        result, ("CHANNEL", "STRESS_VARIANT", "CODEC"),
    )
    result["VOICE_GENERATOR_V57"] = _component_generator(
        result,
        ("VOICE_GENERATOR", "FIRST_GENERATOR", "SECOND_GENERATOR", "ATTACK"),
        "VOICE_PRESENT", "VOICE_FAKE",
    )
    result["MUSIC_GENERATOR_V57"] = _component_generator(
        result, ("MUSIC_GENERATOR",), "MUSIC_PRESENT", "MUSIC_FAKE",
    )
    return result


def load_truths(matrix_path: Path, partition_config: Path) -> pd.DataFrame:
    matrix = yaml.safe_load(matrix_path.read_text("utf-8")) or {}
    key = matrix.get("development_set")
    requested = matrix.get(key) if isinstance(key, str) else None
    if not isinstance(requested, list) or not requested:
        raise ValueError("v57 matrix has no explicit development set")
    partitions = authorized_partitions(
        partition_config, "development", requested,
    )
    frames = []
    for partition in partitions:
        frame = pd.read_csv(partition.truth_path, dtype={"ID": str})
        frame.insert(0, "DATASET", partition.name)
        frames.append(frame)
    result = pd.concat(frames, ignore_index=True, sort=False)
    if result[["DATASET", "ID"]].duplicated().any():
        raise ValueError("duplicate development DATASET/ID")
    return add_diagnostic_axes(result)


def load_anchor_predictions(
    truth: pd.DataFrame, anchor_root: Path,
) -> pd.DataFrame:
    frames = []
    for dataset in truth["DATASET"].drop_duplicates():
        path = anchor_root / dataset / "predictions.csv"
        frame = pd.read_csv(path, dtype={"ID": str})
        frame.insert(0, "DATASET", dataset)
        frames.append(frame[["DATASET", "ID", *PREDICTION_COLUMNS]])
    return pd.concat(frames, ignore_index=True)


def load_candidate_predictions(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, dtype={"ID": str, "DATASET": str})
    required = {"DATASET", "ID", *PREDICTION_COLUMNS}
    if missing := required.difference(frame):
        raise ValueError(f"{path} misses prediction columns: {sorted(missing)}")
    if frame[["DATASET", "ID"]].duplicated().any():
        raise ValueError(f"{path} has duplicate DATASET/ID")
    retained = [
        *PREDICTION_COLUMNS,
        *(column for column in RESIDUAL_COLUMNS if column in frame),
    ]
    return frame[["DATASET", "ID", *retained]]


def _logit(values: pd.Series) -> np.ndarray:
    clipped = np.clip(values.to_numpy(dtype=np.float64), 1e-7, 1 - 1e-7)
    return np.log(clipped) - np.log1p(-clipped)


def _sigmoid(values: np.ndarray) -> np.ndarray:
    return np.where(
        values >= 0,
        1.0 / (1.0 + np.exp(-values)),
        np.exp(values) / (1.0 + np.exp(values)),
    )


def task_isolated_predictions(
    anchor: pd.DataFrame,
    candidate: pd.DataFrame,
    task: str,
) -> pd.DataFrame:
    """Apply exactly one learned task residual and keep four columns anchored.

    File isolation means the direct File residual through the model's fixed
    0.7 direct-file coefficient. Voice/Music isolation intentionally does not
    propagate through noisy-OR into File, making each result a literal
    one-output-column intervention suitable for downstream expert selection.
    """
    specification = {
        "voice": ("VOICE_FAKE_PROB", "VOICE_LOGIT_RESIDUAL", 1.0),
        "music": ("MUSIC_FAKE_PROB", "MUSIC_LOGIT_RESIDUAL", 1.0),
        "file": ("FILE_FAKE_PROB", "DIRECT_FILE_LOGIT_RESIDUAL", 0.7),
    }
    if task not in specification:
        raise ValueError(f"unknown isolated task: {task}")
    missing = set(RESIDUAL_COLUMNS).difference(candidate)
    if missing:
        raise ValueError(
            "candidate misses residual columns required for isolated tasks: "
            f"{sorted(missing)}"
        )
    anchor_indexed = anchor.set_index(["DATASET", "ID"]).sort_index()
    candidate_indexed = candidate.set_index(["DATASET", "ID"]).sort_index()
    if not anchor_indexed.index.equals(candidate_indexed.index):
        raise ValueError("anchor/candidate keys differ for isolated task")
    output = anchor_indexed[list(PREDICTION_COLUMNS)].copy()
    probability, residual, coefficient = specification[task]
    output[probability] = _sigmoid(
        _logit(output[probability])
        + coefficient * candidate_indexed[residual].to_numpy(dtype=np.float64)
    )
    return output.reset_index()


def full_residual_predictions(
    anchor: pd.DataFrame, candidate: pd.DataFrame,
) -> pd.DataFrame:
    """Apply all v57 residuals to any aligned anchor with exact 70/30 File fusion."""
    missing = set(RESIDUAL_COLUMNS).difference(candidate)
    if missing:
        raise ValueError(f"candidate misses residual columns: {sorted(missing)}")
    anchor_indexed = anchor.set_index(["DATASET", "ID"]).sort_index()
    candidate_indexed = candidate.set_index(["DATASET", "ID"]).sort_index()
    if not anchor_indexed.index.equals(candidate_indexed.index):
        raise ValueError("anchor/candidate keys differ for full residual")
    output = anchor_indexed[list(PREDICTION_COLUMNS)].copy()
    anchor_voice = _logit(output.VOICE_FAKE_PROB)
    anchor_music = _logit(output.MUSIC_FAKE_PROB)
    voice = anchor_voice + candidate_indexed.VOICE_LOGIT_RESIDUAL.to_numpy(
        dtype=np.float64
    )
    music = anchor_music + candidate_indexed.MUSIC_LOGIT_RESIDUAL.to_numpy(
        dtype=np.float64
    )

    def noisy_or_logit(voice_logit: np.ndarray, music_logit: np.ndarray) -> np.ndarray:
        probability = 1 - (1 - _sigmoid(voice_logit)) * (1 - _sigmoid(music_logit))
        clipped = np.clip(probability, 1e-7, 1 - 1e-7)
        return np.log(clipped) - np.log1p(-clipped)

    file_logit = (
        _logit(output.FILE_FAKE_PROB)
        + 0.7 * candidate_indexed.DIRECT_FILE_LOGIT_RESIDUAL.to_numpy(
            dtype=np.float64
        )
        + 0.3 * (
            noisy_or_logit(voice, music)
            - noisy_or_logit(anchor_voice, anchor_music)
        )
    )
    output["VOICE_FAKE_PROB"] = _sigmoid(voice)
    output["MUSIC_FAKE_PROB"] = _sigmoid(music)
    output["FILE_FAKE_PROB"] = _sigmoid(file_logit)
    return output.reset_index()


def expert_export(
    anchor: pd.DataFrame, candidate: pd.DataFrame,
) -> pd.DataFrame:
    """Export standalone head residuals/logits for a later frozen v50 combiner."""
    missing = set(RESIDUAL_COLUMNS).difference(candidate)
    if missing:
        raise ValueError(f"candidate misses expert residuals: {sorted(missing)}")
    anchor_indexed = anchor.set_index(["DATASET", "ID"]).sort_index()
    candidate_indexed = candidate.set_index(["DATASET", "ID"]).sort_index()
    if not anchor_indexed.index.equals(candidate_indexed.index):
        raise ValueError("anchor/candidate keys differ for expert export")
    output = pd.DataFrame(index=anchor_indexed.index)
    mapping = (
        ("VOICE", "VOICE_FAKE_PROB", "VOICE_LOGIT_RESIDUAL", 1.0),
        ("MUSIC", "MUSIC_FAKE_PROB", "MUSIC_LOGIT_RESIDUAL", 1.0),
        ("FILE", "FILE_FAKE_PROB", "DIRECT_FILE_LOGIT_RESIDUAL", 0.7),
    )
    for task, probability, residual, coefficient in mapping:
        anchor_logit = _logit(anchor_indexed[probability])
        residual_value = candidate_indexed[residual].to_numpy(dtype=np.float64)
        output[f"ANCHOR_{task}_LOGIT"] = anchor_logit
        output[residual] = residual_value
        output[f"ISOLATED_{task}_EXPERT_LOGIT"] = (
            anchor_logit + coefficient * residual_value
        )
    return output.reset_index()


def join_predictions(truth: pd.DataFrame, prediction: pd.DataFrame) -> pd.DataFrame:
    if set(map(tuple, truth[["DATASET", "ID"]].to_numpy())) != set(
        map(tuple, prediction[["DATASET", "ID"]].to_numpy())
    ):
        raise ValueError("prediction keys do not exactly match v57 development")
    return truth.merge(
        prediction, on=["DATASET", "ID"], how="left", validate="one_to_one"
    )


def metric_row(frame: pd.DataFrame) -> dict[str, float | int]:
    metric = score_frame(frame)
    metric["NORMALIZED_AVAILABLE_ADS"] = normalized_available_ads(metric)
    return metric


def evaluate_model(
    name: str, frame: pd.DataFrame, minimum_group_size: int,
) -> tuple[dict[str, object], list[dict[str, object]], list[dict[str, object]]]:
    pooled = {"MODEL": name, **metric_row(frame)}
    domains = []
    for dataset, group in frame.groupby("DATASET", sort=True):
        domains.append({
            "MODEL": name, "DATASET": str(dataset), **metric_row(group),
        })
    subgroups = []
    for axis in AXIS_COLUMNS:
        if axis not in frame:
            continue
        for value, group in frame.groupby(axis, dropna=False, sort=True):
            if len(group) < minimum_group_size:
                continue
            subgroups.append({
                "MODEL": name, "AXIS": axis, "GROUP": str(value),
                **metric_row(group),
            })
    return pooled, domains, subgroups


def cell_operating_rows(name: str, frame: pd.DataFrame) -> list[dict[str, object]]:
    """Measure homogeneous RR/RF/FR/FF errors at each task's global EER point."""
    definitions = (
        ("FILE", "FILE_FAKE", "FILE_FAKE_PROB", None, 0.5),
        ("VOICE", "VOICE_FAKE", "VOICE_FAKE_PROB", "VOICE_PRESENT", 0.2),
        ("MUSIC", "MUSIC_FAKE", "MUSIC_FAKE_PROB", "MUSIC_PRESENT", 0.3),
    )
    thresholds: dict[str, float] = {}
    for task, label, score, presence, _ in definitions:
        selected = frame if presence is None else frame.loc[frame[presence].eq(1)]
        labels = selected[label].astype(int).to_numpy()
        scores = selected[score].astype(float).to_numpy()
        fpr, tpr, values = roc_curve(
            labels, scores, pos_label=1, drop_intermediate=False,
        )
        index = int(np.argmin(np.abs(fpr - (1 - tpr))))
        thresholds[task] = float(values[index])
    rows = []
    mixed = frame.loc[frame.CELL_V57.isin(("RR", "RF", "FR", "FF"))]
    for cell, group in mixed.groupby("CELL_V57", sort=True):
        row: dict[str, object] = {"MODEL": name, "CELL": cell, "N": len(group)}
        weighted_error = 0.0
        for task, label, score, _, weight in definitions:
            errors = (
                group[score].astype(float).to_numpy() >= thresholds[task]
            ) != group[label].astype(int).to_numpy()
            value = float(errors.mean())
            row[f"{task}_ERROR"] = value
            row[f"{task}_THRESHOLD"] = thresholds[task]
            weighted_error += weight * value
        row["WEIGHTED_CELL_ERROR"] = weighted_error
        rows.append(row)
    return rows


def _finite_lookup(
    frame: pd.DataFrame, keys: list[str], value: str,
) -> dict[tuple[str, ...], float]:
    result = {}
    for _, row in frame.iterrows():
        current = float(row[value])
        if np.isfinite(current):
            result[tuple(str(row[key]) for key in keys)] = current
    return result


def candidate_decision(
    candidate: str,
    overall: pd.DataFrame,
    domains: pd.DataFrame,
    subgroups: pd.DataFrame,
) -> dict[str, object]:
    base_overall = overall.loc[overall.MODEL.eq("identity")].iloc[0]
    current_overall = overall.loc[overall.MODEL.eq(candidate)].iloc[0]
    base_domains = domains.loc[domains.MODEL.eq("identity")]
    current_domains = domains.loc[domains.MODEL.eq(candidate)]
    base_domain = _finite_lookup(
        base_domains, ["DATASET"], "NORMALIZED_AVAILABLE_ADS"
    )
    current_domain = _finite_lookup(
        current_domains, ["DATASET"], "NORMALIZED_AVAILABLE_ADS"
    )
    common_domains = sorted(set(base_domain) & set(current_domain))
    domain_deltas = {
        key: current_domain[key] - base_domain[key] for key in common_domains
    }

    selected_base = subgroups.loc[
        subgroups.MODEL.eq("identity") & subgroups.AXIS.isin(DECISION_AXES)
    ]
    selected_current = subgroups.loc[
        subgroups.MODEL.eq(candidate) & subgroups.AXIS.isin(DECISION_AXES)
    ]
    base_groups = _finite_lookup(
        selected_base, ["AXIS", "GROUP"], "NORMALIZED_AVAILABLE_ADS"
    )
    current_groups = _finite_lookup(
        selected_current, ["AXIS", "GROUP"], "NORMALIZED_AVAILABLE_ADS"
    )
    common_groups = sorted(set(base_groups) & set(current_groups))
    group_deltas = {
        key: current_groups[key] - base_groups[key] for key in common_groups
    }
    pooled_delta = float(current_overall.ADS - base_overall.ADS)
    mean_delta = float(
        np.mean(list(current_domain.values())) - np.mean(list(base_domain.values()))
    )
    worst_delta = float(
        min(current_domain.values()) - min(base_domain.values())
    )
    worst_domain = min(domain_deltas.items(), key=lambda item: item[1])
    worst_group = min(group_deltas.items(), key=lambda item: item[1])
    passed = (
        pooled_delta > 0
        and mean_delta > 0
        and worst_delta > 0
        and worst_domain[1] >= -0.02
        and worst_group[1] >= -0.02
    )
    return {
        "candidate": candidate,
        "accepted": bool(passed),
        "pooled_ads_delta": pooled_delta,
        "mean_domain_ads_delta": mean_delta,
        "worst_domain_ads_delta": worst_delta,
        "largest_dataset_regression": {
            "dataset": worst_domain[0][0], "delta": worst_domain[1],
        },
        "largest_decision_group_regression": {
            "axis": worst_group[0][0], "group": worst_group[0][1],
            "delta": worst_group[1],
        },
        "threshold": -0.02,
    }


def regression_table(
    domains: pd.DataFrame, subgroups: pd.DataFrame,
) -> pd.DataFrame:
    """Return complete domain/cell/layout/channel deltas against identity."""
    rows: list[dict[str, object]] = []
    sources = [
        ("DOMAIN", domains.rename(columns={"DATASET": "GROUP"})),
        *(
            (axis.replace("_V57", ""), subgroups.loc[subgroups.AXIS.eq(axis)])
            for axis in DECISION_AXES
        ),
    ]
    for axis, source in sources:
        baseline = source.loc[source.MODEL.eq("identity")].set_index("GROUP")
        for model in source.MODEL.drop_duplicates():
            if model == "identity":
                continue
            current = source.loc[source.MODEL.eq(model)].set_index("GROUP")
            for group in sorted(set(baseline.index) & set(current.index)):
                base_row = baseline.loc[group]
                current_row = current.loc[group]
                row: dict[str, object] = {
                    "MODEL": model, "AXIS": axis, "GROUP": str(group),
                    "N": int(current_row["N"]),
                }
                for metric in COMPARISON_METRICS:
                    base_value = float(base_row[metric])
                    current_value = float(current_row[metric])
                    row[f"IDENTITY_{metric}"] = base_value
                    row[f"CANDIDATE_{metric}"] = current_value
                    row[f"DELTA_{metric}"] = current_value - base_value
                rows.append(row)
    return pd.DataFrame(rows)


def parse_candidate(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("candidate must be NAME=CSV")
    name, path = value.split("=", 1)
    if not name or name == "identity":
        raise argparse.ArgumentTypeError("candidate needs a non-identity name")
    return name, Path(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", type=Path, required=True)
    parser.add_argument("--partition-config", type=Path, required=True)
    parser.add_argument("--anchor-root", type=Path, required=True)
    parser.add_argument("--candidate", action="append", type=parse_candidate, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--minimum-group-size", type=int, default=20)
    args = parser.parse_args()
    if args.output_dir.exists():
        parser.error(f"refusing to overwrite {args.output_dir}")
    truth = load_truths(args.matrix, args.partition_config)
    anchor = load_anchor_predictions(truth, args.anchor_root)
    predictions = {"identity": anchor}
    expert_exports: dict[str, pd.DataFrame] = {}
    for name, path in args.candidate:
        candidate = load_candidate_predictions(path)
        predictions[name] = candidate
        if set(RESIDUAL_COLUMNS).issubset(candidate):
            expert_exports[name] = expert_export(anchor, candidate)
            for task in ("voice", "music", "file"):
                predictions[f"{name}__{task}_only"] = task_isolated_predictions(
                    anchor, candidate, task,
                )
    overall_rows, domain_rows, subgroup_rows, cell_rows = [], [], [], []
    for name, prediction in predictions.items():
        pooled, domains, subgroups = evaluate_model(
            name, join_predictions(truth, prediction), args.minimum_group_size,
        )
        overall_rows.append(pooled)
        domain_rows.extend(domains)
        subgroup_rows.extend(subgroups)
        cell_rows.extend(cell_operating_rows(name, join_predictions(truth, prediction)))
    overall = pd.DataFrame(overall_rows)
    domains = pd.DataFrame(domain_rows)
    subgroups = pd.DataFrame(subgroup_rows)
    regressions = regression_table(domains, subgroups)
    cell_operating = pd.DataFrame(cell_rows)
    baseline_cells = cell_operating.loc[
        cell_operating.MODEL.eq("identity")
    ].set_index("CELL")
    for task in ("FILE", "VOICE", "MUSIC", "WEIGHTED_CELL"):
        column = f"{task}_ERROR"
        cell_operating[f"DELTA_{column}"] = cell_operating.apply(
            lambda row: float(row[column] - baseline_cells.loc[row.CELL, column]),
            axis=1,
        )
    decisions = [
        candidate_decision(name, overall, domains, subgroups)
        for name in predictions if name != "identity"
    ]
    args.output_dir.mkdir(parents=True)
    overall.to_csv(args.output_dir / "overall.csv", index=False)
    domains.to_csv(args.output_dir / "domains.csv", index=False)
    subgroups.to_csv(args.output_dir / "subgroups.csv", index=False)
    regressions.to_csv(args.output_dir / "regressions.csv", index=False)
    cell_operating.to_csv(args.output_dir / "cell_operating_points.csv", index=False)
    for name, frame in expert_exports.items():
        frame.to_csv(args.output_dir / f"{name}_expert_logits.csv", index=False)
    summary = {
        "development_rows": len(truth),
        "development_datasets": truth.DATASET.drop_duplicates().tolist(),
        "minimum_group_size": args.minimum_group_size,
        "models": list(predictions),
        "decisions": decisions,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
