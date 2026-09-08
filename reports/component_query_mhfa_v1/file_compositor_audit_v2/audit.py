#!/usr/bin/env python3
"""Dev-select monotone File compositors, then audit one frozen candidate.

The development sweep sees only factorial ``truth_dev``.  Locked truth is
loaded after the selection has been finalized.  No audio is opened and no
shared model/source file is changed.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.model_selection import StratifiedKFold


ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]

from evaluate_diagnostic import official_eer, score_frame  # noqa: E402
from evaluate_wpt_v38_residual import (  # noqa: E402
    reconstruct_dev_anchors, reconstruct_v18_locked,
)


PATCH_DEV = ROOT / "reports/eat_patch_graph_v1/seed01_balanced_fm/dev_predictions.csv"
PATCH_LOCKED = ROOT / "reports/eat_patch_graph_v1/seed01_balanced_fm_locked/predictions.csv"
QUERY_DEV = ROOT / "reports/component_query_mhfa_v1/seed04_base/dev_predictions.csv"
QUERY_LOCKED = ROOT / "reports/component_query_mhfa_v1/seed04_base_locked/predictions.csv"
LOCKED_REFERENCE = ROOT / "reports/component_query_mhfa_v1/v47_locked_audit.csv"
LOCKED_DATASETS = {
    "factorial": "factorial_eval_1200_v2_holdout",
    "phone": "phone_factorial_1200_v1",
    "yue": "yue_cross_component_audit_v1",
}


def logit(values) -> np.ndarray:
    values = np.clip(np.asarray(values, dtype=np.float64), 1e-6, 1 - 1e-6)
    return np.log(values) - np.log1p(-values)


def sigmoid(values) -> np.ndarray:
    return np.exp(-np.logaddexp(0, -np.asarray(values, dtype=np.float64)))


def fuse(anchor, expert, weight: float) -> np.ndarray:
    return sigmoid((1 - weight) * logit(anchor) + weight * logit(expert))


def select(frame: pd.DataFrame, dataset: str, index: pd.Index) -> pd.DataFrame:
    result = frame.loc[frame.DATASET.eq(dataset)].set_index("ID")
    missing = index.difference(result.index)
    if len(missing):
        raise ValueError(f"{dataset}: missing {len(missing)} predictions")
    return result.loc[index]


def apply_v46(
    anchor: pd.DataFrame, patch: pd.DataFrame, query: pd.DataFrame,
) -> pd.DataFrame:
    """Exact deployed order immediately before the v47 noisy-OR."""
    result = anchor.copy()
    result["FILE_FAKE_PROB"] = fuse(
        result.FILE_FAKE_PROB, patch.FILE_FAKE_PROB, .05
    )
    result["MUSIC_FAKE_PROB"] = fuse(
        result.MUSIC_FAKE_PROB, patch.MUSIC_FAKE_PROB, .05
    )
    result["FILE_FAKE_PROB"] = fuse(
        result.FILE_FAKE_PROB, query.FILE_FAKE_PROB, .025
    )
    result["MUSIC_FAKE_PROB"] = fuse(
        result.MUSIC_FAKE_PROB, query.MUSIC_FAKE_PROB, .05
    )
    return result


def v47_file(v46: pd.DataFrame) -> np.ndarray:
    evidence = 1 - (
        (1 - v46.VOICE_FAKE_PROB) * (1 - v46.MUSIC_FAKE_PROB)
    )
    return fuse(v46.FILE_FAKE_PROB, evidence, .30)


def load_dev() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    truth, anchor, _ = reconstruct_dev_anchors()
    patch = select(
        pd.read_csv(PATCH_DEV, dtype={"ID": str}),
        "factorial_eval_1200_v2_dev", anchor.index,
    )
    query = select(
        pd.read_csv(QUERY_DEV, dtype={"ID": str}),
        "factorial_eval_1200_v2_dev", anchor.index,
    )
    return truth, apply_v46(anchor, patch, query), query


def controlled_indices(truth: pd.DataFrame) -> dict[str, np.ndarray]:
    result = {
        "overall": np.arange(len(truth)),
        "voice_only": np.flatnonzero(truth.MIX_MODE.eq("voice_only")),
        "music_only": np.flatnonzero(truth.MIX_MODE.eq("music_only")),
    }
    for mode in ("concurrent", "partial_overlap", "sequential"):
        selected = truth.MIX_MODE.eq(mode)
        rr = selected & truth.VOICE_FAKE.eq(0) & truth.MUSIC_FAKE.eq(0)
        for voice, music, case in (
            (1, 0, "vfmr"), (0, 1, "vrmf"), (1, 1, "vfmf"),
        ):
            positive = (
                selected & truth.VOICE_FAKE.eq(voice)
                & truth.MUSIC_FAKE.eq(music)
            )
            result[f"{mode}_{case}"] = np.flatnonzero(rr | positive)
    return result


def contrast_metrics(
    truth: pd.DataFrame, scores: np.ndarray,
    indices: dict[str, np.ndarray],
) -> dict[str, float]:
    labels = truth.FILE_FAKE.to_numpy(np.int64)
    return {
        name: official_eer(labels[current], scores[current])
        for name, current in indices.items()
    }


def candidate_row(
    truth: pd.DataFrame, indices: dict[str, np.ndarray], baseline: dict[str, float],
    method: str, scores: np.ndarray, outer: float = np.nan,
    temperature: float = np.nan, query_voice_weight: float = np.nan,
    query_extra_mass: float = np.nan,
) -> dict[str, object]:
    metrics = contrast_metrics(truth, scores, indices)
    protected = [
        name for name in indices
        if name not in {"overall", "concurrent_vfmr"}
    ]
    deltas = [metrics[name] - baseline[name] for name in protected]
    return {
        "METHOD": method, "OUTER_WEIGHT": outer,
        "TEMPERATURE": temperature,
        "QUERY_VOICE_WEIGHT": query_voice_weight,
        "QUERY_EXTRA_MASS": query_extra_mass,
        **{f"EER_{key.upper()}": value for key, value in metrics.items()},
        "TARGET_DELTA": metrics["concurrent_vfmr"] - baseline["concurrent_vfmr"],
        "OVERALL_DELTA": metrics["overall"] - baseline["overall"],
        "MAX_OTHER_DELTA": max(deltas), "SUM_OTHER_DELTA": sum(deltas),
    }


def formula_sweep(
    truth: pd.DataFrame, v46: pd.DataFrame, query: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series, dict[str, float]]:
    indices = controlled_indices(truth)
    baseline_scores = v47_file(v46)
    baseline = contrast_metrics(truth, baseline_scores, indices)
    file_logit = logit(v46.FILE_FAKE_PROB)
    voice_logit = logit(v46.VOICE_FAKE_PROB)
    music_logit = logit(v46.MUSIC_FAKE_PROB)
    query_voice_logit = logit(query.VOICE_FAKE_PROB)
    weights = np.linspace(0, 1, 41)
    rows = [candidate_row(
        truth, indices, baseline, "v46_identity",
        np.asarray(v46.FILE_FAKE_PROB), outer=0,
    ), candidate_row(
        truth, indices, baseline, "v47_noisy_or",
        baseline_scores, outer=.30,
    )]

    noisy_or = 1 - (
        (1 - v46.VOICE_FAKE_PROB) * (1 - v46.MUSIC_FAKE_PROB)
    )
    maximum = sigmoid(np.maximum(voice_logit, music_logit))
    for outer in weights:
        rows.append(candidate_row(
            truth, indices, baseline, "noisy_or",
            fuse(v46.FILE_FAKE_PROB, noisy_or, outer), outer=outer,
        ))
        rows.append(candidate_row(
            truth, indices, baseline, "max",
            fuse(v46.FILE_FAKE_PROB, maximum, outer), outer=outer,
        ))
        for temperature in (.25, .50, 1., 2., 4.):
            evidence_logit = temperature * np.logaddexp(
                voice_logit / temperature, music_logit / temperature
            )
            rows.append(candidate_row(
                truth, indices, baseline, "logsumexp",
                sigmoid((1 - outer) * file_logit + outer * evidence_logit),
                outer=outer, temperature=temperature,
            ))

    # A small query-Voice contribution is permissible: the query model is
    # already computed in v47, and all coefficients below remain nonnegative.
    for query_weight in (0, .025, .05, .075, .10, .15, .20, .30, .40, .50):
        blended_voice_logit = (
            (1 - query_weight) * voice_logit
            + query_weight * query_voice_logit
        )
        blended_voice = sigmoid(blended_voice_logit)
        blended_or = 1 - (1 - blended_voice) * (1 - v46.MUSIC_FAKE_PROB)
        for outer in weights:
            rows.append(candidate_row(
                truth, indices, baseline, "query_voice_blend_noisy_or",
                fuse(v46.FILE_FAKE_PROB, blended_or, outer), outer=outer,
                query_voice_weight=query_weight,
            ))
            for temperature in (.50, 1., 2., 4.):
                evidence_logit = temperature * np.logaddexp(
                    blended_voice_logit / temperature,
                    music_logit / temperature,
                )
                rows.append(candidate_row(
                    truth, indices, baseline, "query_voice_blend_logsumexp",
                    sigmoid(
                        (1 - outer) * file_logit + outer * evidence_logit
                    ), outer=outer, temperature=temperature,
                    query_voice_weight=query_weight,
                ))

    for mass in (.025, .05, .10, .20, .30, .50, 1.):
        for temperature in (.50, 1., 2., 4.):
            evidence_logit = temperature * np.log(
                np.exp(voice_logit / temperature)
                + np.exp(music_logit / temperature)
                + mass * np.exp(query_voice_logit / temperature)
            )
            for outer in weights:
                rows.append(candidate_row(
                    truth, indices, baseline, "three_source_logsumexp",
                    sigmoid(
                        (1 - outer) * file_logit + outer * evidence_logit
                    ), outer=outer, temperature=temperature,
                    query_extra_mass=mass,
                ))

    # A convex nonnegative logit compositor is the simplest learned-family
    # approximation without fitting labels.
    for voice_weight in np.linspace(0, 1, 41):
        remaining_steps = int(round((1 - voice_weight) / .025))
        for music_weight in np.linspace(0, 1 - voice_weight, remaining_steps + 1):
            file_weight = 1 - voice_weight - music_weight
            rows.append(candidate_row(
                truth, indices, baseline, "convex_logit",
                sigmoid(
                    file_weight * file_logit + voice_weight * voice_logit
                    + music_weight * music_logit
                ), outer=1 - file_weight,
                query_voice_weight=voice_weight,
                query_extra_mass=music_weight,
            ))

    sweep = pd.DataFrame(rows)
    eligible = sweep.loc[
        (sweep.TARGET_DELTA < -1e-12)
        & (sweep.OVERALL_DELTA <= 1e-12)
        & (sweep.MAX_OTHER_DELTA <= 1e-12)
    ]
    if eligible.empty:
        raise RuntimeError("no compositor satisfies the frozen dev safety rule")
    selected = eligible.sort_values([
        "EER_CONCURRENT_VFMR", "EER_OVERALL", "SUM_OTHER_DELTA",
        "OUTER_WEIGHT", "METHOD",
    ]).iloc[0]
    return sweep, selected, baseline


def learned_cv(
    truth: pd.DataFrame, v46: pd.DataFrame, baseline: dict[str, float],
) -> pd.DataFrame:
    """Five-fold cell-stratified CV for nonnegative logistic alternatives."""
    file_logit = logit(v46.FILE_FAKE_PROB)
    voice_logit = logit(v46.VOICE_FAKE_PROB)
    music_logit = logit(v46.MUSIC_FAKE_PROB)
    voice_presence = v46.VOICE_PRESENT_PROB.to_numpy(np.float64)
    music_presence = v46.MUSIC_PRESENT_PROB.to_numpy(np.float64)
    mixed = voice_presence * music_presence
    configurations = {
        "logit": (
            np.c_[file_logit, voice_logit, music_logit, np.ones(len(truth))], 3,
        ),
        "logit_presence": (
            np.c_[file_logit, voice_logit, music_logit,
                  voice_presence, music_presence, mixed, np.ones(len(truth))], 3,
        ),
        "logit_monotone_mixed_interaction": (
            np.c_[file_logit, voice_logit, music_logit,
                  voice_logit * mixed, music_logit * mixed,
                  voice_presence, music_presence, mixed, np.ones(len(truth))], 5,
        ),
        "probability_presence": (
            np.c_[v46.FILE_FAKE_PROB, v46.VOICE_FAKE_PROB,
                  v46.MUSIC_FAKE_PROB, voice_presence, music_presence,
                  mixed, np.ones(len(truth))], 3,
        ),
    }
    labels = truth.FILE_FAKE.to_numpy(np.float64)
    counts = truth.EVAL_CELL.value_counts()
    sample_weight = np.asarray([1 / counts[cell] for cell in truth.EVAL_CELL])
    sample_weight /= sample_weight.mean()
    folds = StratifiedKFold(5, shuffle=True, random_state=20260904)
    indices = controlled_indices(truth)
    rows = []
    for name, (features, monotone_count) in configurations.items():
        bounds = [(0, 10)] * monotone_count + [
            (None, None)
        ] * (features.shape[1] - monotone_count)
        for regularization in (0, 1e-4, 1e-3, 1e-2, 1e-1, 1.):
            predictions = np.zeros(len(truth), dtype=np.float64)
            coefficients = []
            for train, validation in folds.split(features, truth.EVAL_CELL):
                def objective(weight):
                    score = features[train] @ weight
                    loss = np.logaddexp(0, score) - labels[train] * score
                    penalty = .5 * regularization * np.square(weight[:-1]).sum()
                    return np.average(loss, weights=sample_weight[train]) + penalty

                def gradient(weight):
                    score = features[train] @ weight
                    residual = (sigmoid(score) - labels[train]) * sample_weight[train]
                    value = features[train].T @ residual / sample_weight[train].sum()
                    value[:-1] += regularization * weight[:-1]
                    return value

                initial = np.zeros(features.shape[1], dtype=np.float64)
                initial[:3] = (.5, .25, .25)
                fitted = minimize(
                    objective, initial, jac=gradient, bounds=bounds,
                    method="L-BFGS-B", options={"maxiter": 1_000},
                )
                if not fitted.success:
                    raise RuntimeError(f"monotone fit failed: {fitted.message}")
                predictions[validation] = features[validation] @ fitted.x
                coefficients.append(fitted.x)
            metrics = contrast_metrics(truth, predictions, indices)
            protected = [
                key for key in indices
                if key not in {"overall", "concurrent_vfmr"}
            ]
            other_deltas = [metrics[key] - baseline[key] for key in protected]
            rows.append({
                "MODEL": name, "L2": regularization,
                **{f"EER_{key.upper()}": value for key, value in metrics.items()},
                "TARGET_DELTA": metrics["concurrent_vfmr"] - baseline["concurrent_vfmr"],
                "OVERALL_DELTA": metrics["overall"] - baseline["overall"],
                "MAX_OTHER_DELTA": max(other_deltas),
                "SUM_OTHER_DELTA": sum(other_deltas),
                "MEAN_CV_COEFFICIENTS": json.dumps(
                    np.mean(coefficients, axis=0).round(8).tolist()
                ),
            })
    return pd.DataFrame(rows)


def family_summary(sweep: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for method, group in sweep.groupby("METHOD", sort=True):
        safe = group.loc[
            (group.TARGET_DELTA < -1e-12)
            & (group.OVERALL_DELTA <= 1e-12)
            & (group.MAX_OTHER_DELTA <= 1e-12)
        ]
        pool = safe if len(safe) else group
        best = pool.sort_values([
            "EER_CONCURRENT_VFMR", "EER_OVERALL", "MAX_OTHER_DELTA",
            "SUM_OTHER_DELTA", "OUTER_WEIGHT",
        ]).iloc[0]
        rows.append({
            "METHOD": method, "CANDIDATES": len(group),
            "STRICT_SAFE_CANDIDATES": len(safe),
            **{
                key: best[key] for key in (
                    "OUTER_WEIGHT", "TEMPERATURE", "QUERY_VOICE_WEIGHT",
                    "QUERY_EXTRA_MASS", "EER_OVERALL",
                    "EER_CONCURRENT_VFMR", "TARGET_DELTA",
                    "OVERALL_DELTA", "MAX_OTHER_DELTA", "SUM_OTHER_DELTA",
                )
            },
        })
    return pd.DataFrame(rows)


def apply_selected(
    v46: pd.DataFrame, query: pd.DataFrame, selected: pd.Series,
) -> np.ndarray:
    if selected.METHOD != "three_source_logsumexp":
        raise ValueError(f"unexpected selected family: {selected.METHOD}")
    temperature = float(selected.TEMPERATURE)
    evidence_logit = temperature * np.log(
        np.exp(logit(v46.VOICE_FAKE_PROB) / temperature)
        + np.exp(logit(v46.MUSIC_FAKE_PROB) / temperature)
        + float(selected.QUERY_EXTRA_MASS)
        * np.exp(logit(query.VOICE_FAKE_PROB) / temperature)
    )
    outer = float(selected.OUTER_WEIGHT)
    return sigmoid(
        (1 - outer) * logit(v46.FILE_FAKE_PROB) + outer * evidence_logit
    )


def locked_audit(selected: pd.Series) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    patch_all = pd.read_csv(PATCH_LOCKED, dtype={"ID": str})
    query_all = pd.read_csv(QUERY_LOCKED, dtype={"ID": str})
    metric_rows, cell_rows, prediction_rows = [], [], []
    for short, dataset in LOCKED_DATASETS.items():
        truth, anchor = reconstruct_v18_locked(short)
        patch = select(patch_all, dataset, anchor.index)
        query = select(query_all, dataset, anchor.index)
        v46 = apply_v46(anchor, patch, query)
        predictions = {
            "v46": np.asarray(v46.FILE_FAKE_PROB),
            "v47": v47_file(v46),
            "dev_selected": apply_selected(v46, query, selected),
        }
        for method, file_score in predictions.items():
            current = v46.copy()
            current["FILE_FAKE_PROB"] = file_score
            metric_rows.append({
                "DATASET": short, "METHOD": method,
                **score_frame(truth.join(current)),
            })
        for sample_id, v47_score, selected_score in zip(
            anchor.index, predictions["v47"], predictions["dev_selected"]
        ):
            prediction_rows.append({
                "DATASET": short, "ID": sample_id,
                "V47_FILE_FAKE_PROB": v47_score,
                "SELECTED_FILE_FAKE_PROB": selected_score,
            })
        if short in {"factorial", "yue"}:
            mode_column = "MIX_MODE"
            modes = ("concurrent", "partial_overlap", "sequential")
        else:
            mode_column = "CONDITION"
            modes = ("simultaneous",)
        for mode in modes:
            within = truth[mode_column].eq(mode)
            rr = within & truth.VOICE_FAKE.eq(0) & truth.MUSIC_FAKE.eq(0)
            for voice, music, case in (
                (1, 0, "vfmr"), (0, 1, "vrmf"), (1, 1, "vfmf"),
            ):
                positive = (
                    within & truth.VOICE_FAKE.eq(voice)
                    & truth.MUSIC_FAKE.eq(music)
                )
                ids = np.flatnonzero((rr | positive).to_numpy())
                if not len(ids):
                    continue
                for method, file_score in predictions.items():
                    cell_rows.append({
                        "DATASET": short, "LAYOUT": mode, "CASE": case,
                        "METHOD": method, "N": len(ids),
                        "FILE_EER": official_eer(
                            truth.FILE_FAKE.iloc[ids], file_score[ids]
                        ),
                    })

    metrics = pd.DataFrame(metric_rows)
    reference = pd.read_csv(LOCKED_REFERENCE)
    reconstructed = metrics.loc[metrics.METHOD.eq("v47")].copy()
    reconstructed["DATASET"] = reconstructed.DATASET.map(LOCKED_DATASETS)
    check = reconstructed.merge(reference, on="DATASET", suffixes=("_NEW", "_REF"))
    for column in ("FILE_EER", "VOICE_EER", "MUSIC_EER", "ADS"):
        if not np.allclose(check[f"{column}_NEW"], check[f"{column}_REF"], atol=1e-9):
            raise AssertionError(f"exact v47 reconstruction failed for {column}")
    return metrics, pd.DataFrame(cell_rows), pd.DataFrame(prediction_rows)


def main() -> None:
    truth, v46, query = load_dev()
    sweep, selected, baseline = formula_sweep(truth, v46, query)
    learned = learned_cv(truth, v46, baseline)
    metrics, cells, predictions = locked_audit(selected)
    sweep.to_csv(HERE / "dev_sweep.csv", index=False)
    family_summary(sweep).to_csv(HERE / "dev_family_summary.csv", index=False)
    learned.to_csv(HERE / "learned_cv.csv", index=False)
    metrics.to_csv(HERE / "locked_metrics.csv", index=False)
    cells.to_csv(HERE / "locked_cells.csv", index=False)
    predictions.to_csv(HERE / "locked_predictions.csv", index=False)
    selection = {
        "selection_data": "factorial_eval_1200_v2_dev only",
        "safety_rule": (
            "target EER improves; overall and every other controlled File "
            "contrast EER do not regress versus exact v47"
        ),
        "selected": {
            key: (None if pd.isna(selected[key]) else selected[key])
            for key in (
                "METHOD", "OUTER_WEIGHT", "TEMPERATURE",
                "QUERY_VOICE_WEIGHT", "QUERY_EXTRA_MASS", "EER_OVERALL",
                "EER_CONCURRENT_VFMR", "MAX_OTHER_DELTA",
            )
        },
        "locked_truth_loaded_after_selection": True,
        "v47_reconstruction_matches_reference_atol": 1e-9,
    }
    (HERE / "selection.json").write_text(
        json.dumps(selection, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(selection, indent=2))
    print(metrics[[
        "DATASET", "METHOD", "FILE_EER", "VOICE_EER", "MUSIC_EER", "ADS"
    ]].round(6).to_string(index=False))


if __name__ == "__main__":
    main()
