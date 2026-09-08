#!/usr/bin/env python3
"""Dev-only source-restricted fusion of already-computed query/WPT Music scores.

The current v48 runtime already executes both the component-query EAT/SPEAR
head and the WPT/Spectra head.  This experiment therefore adds no backbone
pass: it asks whether their Music logits form a source-robust consensus.  The
coarse fusion grid is selected on the seven authorized model-development
banks.  Leave-one-development-bank-out selection treats each bank as an
unseen source audit.  No locked/holdout/phone/YuE file is opened.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from evaluate_diagnostic import official_eer, score_frame  # noqa: E402


DEV_BANKS = (
    "mixfake_music_dev_v1",
    "external_mixed_v1",
    "source_disjoint_mixed_v1",
    "source_disjoint_mixed_equal_v1",
    "source_disjoint_music_v1",
    "factorial_eval_1200_v2_dev",
    "telephone_mixed_dev_v1",
)
WEIGHTS = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5)


def truth_path(name: str) -> Path:
    if name == "factorial_eval_1200_v2_dev":
        return ROOT / "data/eval/factorial_eval_1200_v2/truth_dev.csv"
    return ROOT / "data/eval" / name / "truth.csv"


def indexed(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, dtype={"ID": str})
    if frame.ID.duplicated().any():
        raise ValueError(f"duplicate IDs in {path}")
    return frame.set_index("ID")


def select(frame: pd.DataFrame, dataset: str, ids: pd.Index) -> pd.DataFrame:
    block = frame.loc[frame.DATASET.eq(dataset)].set_index("ID")
    missing = ids.difference(block.index)
    if len(missing):
        raise ValueError(f"{dataset}: missing {len(missing)} predictions")
    return block.loc[ids]


def logit(values, epsilon: float = 1e-6) -> np.ndarray:
    values = np.clip(np.asarray(values, dtype=np.float64), epsilon, 1 - epsilon)
    return np.log(values) - np.log1p(-values)


def sigmoid(values) -> np.ndarray:
    return np.exp(-np.logaddexp(0, -np.asarray(values, dtype=np.float64)))


def fuse(anchor, expert, weight: float, epsilon: float = 1e-6) -> np.ndarray:
    return sigmoid(
        (1 - weight) * logit(anchor, epsilon)
        + weight * logit(expert, epsilon)
    )


def music_eer(truth: pd.DataFrame, score) -> float:
    selected = truth.MUSIC_PRESENT.eq(1) & truth.MUSIC_FAKE.notna()
    return official_eer(
        truth.loc[selected, "MUSIC_FAKE"].astype(int),
        np.asarray(score)[selected.to_numpy()],
    )


def dev_tables(
    query: pd.DataFrame, wpt: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[float, np.ndarray], pd.DataFrame]:
    rows = []
    by_weight: dict[float, list[float]] = {weight: [] for weight in WEIGHTS}
    predictions = []
    for name in DEV_BANKS:
        truth = indexed(truth_path(name))
        query_block = select(query, name, truth.index)
        wpt_block = select(wpt, name, truth.index)
        baseline = music_eer(truth, query_block.MUSIC_FAKE_PROB)
        for weight in WEIGHTS:
            score = fuse(
                query_block.MUSIC_FAKE_PROB,
                wpt_block.MUSIC_FAKE_PROB,
                weight,
            )
            eer = music_eer(truth, score)
            by_weight[weight].append(eer)
            rows.append({
                "DATASET": name,
                "WPT_WEIGHT": weight,
                "N": int(truth.MUSIC_PRESENT.eq(1).sum()),
                "MUSIC_EER": eer,
                "DELTA_VS_QUERY": eer - baseline,
            })
            predictions.append(pd.DataFrame({
                "DATASET": name,
                "ID": truth.index,
                "WPT_WEIGHT": weight,
                "QUERY_MUSIC_PROB": query_block.MUSIC_FAKE_PROB.to_numpy(),
                "WPT_MUSIC_PROB": wpt_block.MUSIC_FAKE_PROB.to_numpy(),
                "FUSED_MUSIC_PROB": score,
            }))
    return (
        pd.DataFrame(rows),
        {weight: np.asarray(values) for weight, values in by_weight.items()},
        pd.concat(predictions, ignore_index=True),
    )


def robust_score(eers: np.ndarray, indices: list[int]) -> float:
    values = eers[indices]
    return float(1 - 0.5 * (values.mean() + values.max()))


def choose(
    by_weight: dict[float, np.ndarray], indices: list[int],
) -> tuple[float, float]:
    baseline = by_weight[0.0][indices]
    eligible = []
    for weight in WEIGHTS:
        current = by_weight[weight][indices]
        if np.all(current <= baseline + 1e-12):
            # Prefer the smaller coefficient only when robust scores tie.
            eligible.append((robust_score(by_weight[weight], indices), -weight))
    if not eligible:
        raise RuntimeError("no source-safe fusion weight")
    best_score, negative_weight = max(eligible)
    return float(-negative_weight), float(best_score)


def lodo_table(by_weight: dict[float, np.ndarray]) -> pd.DataFrame:
    rows = []
    for held_index, held_name in enumerate(DEV_BANKS):
        fit = [index for index in range(len(DEV_BANKS)) if index != held_index]
        weight, fit_score = choose(by_weight, fit)
        baseline = by_weight[0.0][held_index]
        held_eer = by_weight[weight][held_index]
        rows.append({
            "HELD_OUT_DATASET": held_name,
            "SELECTED_WPT_WEIGHT": weight,
            "FIT_ROBUST_SCORE": fit_score,
            "HELD_OUT_QUERY_EER": baseline,
            "HELD_OUT_FUSED_EER": held_eer,
            "HELD_OUT_DELTA": held_eer - baseline,
        })
    return pd.DataFrame(rows)


def reconstruct_v18_dev() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Reproduce the frozen v18 factorial-dev chain without model inference."""
    truth = indexed(truth_path("factorial_eval_1200_v2_dev"))
    anchor = indexed(
        ROOT / "output/factorial_eval_1200_v2_lme_spear.csv"
    ).loc[truth.index].copy()
    temporal = pd.read_csv(
        ROOT / "reports/temporal_dual_domain_hybrid/ensemble_dev/predictions.csv",
        dtype={"ID": str},
    )
    temporal = select(temporal, "factorial_eval_1200_v2_dev", anchor.index)
    for column in ("FILE_FAKE_PROB", "VOICE_FAKE_PROB", "MUSIC_FAKE_PROB"):
        anchor[column] = fuse(anchor[column], temporal[column], 0.05, 1e-5)
    mert = indexed(
        ROOT / "reports/sofia_mert_v1/factorial_dev_safe_exact.csv"
    ).loc[anchor.index, "SOFIA_MERT_FAKE_PROB"]
    anchor["FILE_FAKE_PROB"] = fuse(
        anchor.FILE_FAKE_PROB, mert, 0.025, 1e-5
    )
    anchor["MUSIC_FAKE_PROB"] = fuse(
        anchor.MUSIC_FAKE_PROB, mert, 0.0125, 1e-5
    )
    fakeprint = pd.read_csv(
        ROOT / "reports/modern_fakeprint_v1/independent_margin_scores.csv",
        dtype={"ID": str},
    )
    fakeprint = select(fakeprint, "factorial_eval_1200_v2", anchor.index)
    for column in ("FILE_FAKE_PROB", "MUSIC_FAKE_PROB"):
        anchor[column] = fuse(
            anchor[column], fakeprint.MODERN_FAKEPRINT_PROB, 0.025, 1e-5
        )
    invariant = pd.read_csv(
        ROOT / "reports/invariant_dual_domain_v2/dec_n1/dev_predictions.csv",
        dtype={"ID": str},
    )
    invariant = select(invariant, "factorial_eval_1200_v2_dev", anchor.index)
    for column in ("FILE_FAKE_PROB", "VOICE_FAKE_PROB", "MUSIC_FAKE_PROB"):
        anchor[column] = fuse(anchor[column], invariant[column], 0.05, 1e-5)
    return truth, anchor


def exact_v47(
    anchor: pd.DataFrame, patch: pd.DataFrame, query: pd.DataFrame,
) -> pd.DataFrame:
    result = anchor.copy()
    result["FILE_FAKE_PROB"] = fuse(
        result.FILE_FAKE_PROB, patch.FILE_FAKE_PROB, 0.05
    )
    result["MUSIC_FAKE_PROB"] = fuse(
        result.MUSIC_FAKE_PROB, patch.MUSIC_FAKE_PROB, 0.05
    )
    result["FILE_FAKE_PROB"] = fuse(
        result.FILE_FAKE_PROB, query.FILE_FAKE_PROB, 0.025
    )
    result["MUSIC_FAKE_PROB"] = fuse(
        result.MUSIC_FAKE_PROB, query.MUSIC_FAKE_PROB, 0.05
    )
    evidence = 1 - (
        (1 - result.VOICE_FAKE_PROB) * (1 - result.MUSIC_FAKE_PROB)
    )
    result["FILE_FAKE_PROB"] = fuse(result.FILE_FAKE_PROB, evidence, 0.30)
    return result


def factorial_v47_tables(
    query_all: pd.DataFrame, wpt_all: pd.DataFrame, weight: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    truth, anchor = reconstruct_v18_dev()
    name = "factorial_eval_1200_v2_dev"
    patch_all = pd.read_csv(
        ROOT / "reports/eat_patch_graph_v1/seed01_balanced_fm/dev_predictions.csv",
        dtype={"ID": str},
    )
    patch = select(patch_all, name, anchor.index)
    query = select(query_all, name, anchor.index)
    wpt = select(wpt_all, name, anchor.index)
    candidate_query = query.copy()
    candidate_query["MUSIC_FAKE_PROB"] = fuse(
        query.MUSIC_FAKE_PROB, wpt.MUSIC_FAKE_PROB, weight
    )
    predictions = {
        "v47_query_only": exact_v47(anchor, patch, query),
        "source_restricted_consensus": exact_v47(
            anchor, patch, candidate_query
        ),
    }
    metrics, channels = [], []
    for method, prediction in predictions.items():
        metrics.append({"METHOD": method, **score_frame(truth.join(prediction))})
        for channel, ids in truth.groupby("CHANNEL").groups.items():
            try:
                value = score_frame(truth.loc[ids].join(prediction.loc[ids]))
            except ValueError:
                continue
            if np.isfinite(value["ADS"]):
                channels.append({"METHOD": method, "CHANNEL": channel, **value})
    exported = pd.DataFrame({
        "ID": truth.index,
        "BASE_FILE_FAKE_PROB": predictions["v47_query_only"].FILE_FAKE_PROB,
        "BASE_MUSIC_FAKE_PROB": predictions["v47_query_only"].MUSIC_FAKE_PROB,
        "FUSED_FILE_FAKE_PROB": predictions[
            "source_restricted_consensus"
        ].FILE_FAKE_PROB,
        "FUSED_MUSIC_FAKE_PROB": predictions[
            "source_restricted_consensus"
        ].MUSIC_FAKE_PROB,
    })
    return pd.DataFrame(metrics), pd.DataFrame(channels), exported


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--query-predictions", type=Path,
        default=ROOT / "reports/component_query_mhfa_v1/seed04_base/dev_predictions.csv",
    )
    parser.add_argument(
        "--wpt-predictions", type=Path,
        default=ROOT / "reports/wpt_spectra_v1/seed_20260906/dev_predictions.csv",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    query = pd.read_csv(args.query_predictions, dtype={"ID": str})
    wpt = pd.read_csv(args.wpt_predictions, dtype={"ID": str})
    sweep, by_weight, all_predictions = dev_tables(query, wpt)
    selected_weight, selection = choose(
        by_weight, list(range(len(DEV_BANKS)))
    )
    lodo = lodo_table(by_weight)
    v47_metrics, v47_channels, v47_predictions = factorial_v47_tables(
        query, wpt, selected_weight
    )
    base_channels = v47_channels.loc[
        v47_channels.METHOD.eq("v47_query_only")
    ].set_index("CHANNEL")
    fused_channels = v47_channels.loc[
        v47_channels.METHOD.eq("source_restricted_consensus")
    ].set_index("CHANNEL")
    channel_delta = (
        fused_channels.ADS - base_channels.ADS
    ).dropna()
    base_metric = v47_metrics.iloc[0]
    fused_metric = v47_metrics.iloc[1]
    selected_eers = by_weight[selected_weight]
    baseline_eers = by_weight[0.0]
    accepted = bool(
        selected_weight > 0
        and np.all(selected_eers <= baseline_eers + 1e-12)
        and (lodo.HELD_OUT_DELTA <= 1e-12).all()
        and fused_metric.MUSIC_EER <= base_metric.MUSIC_EER + 1e-12
        and fused_metric.FILE_EER <= base_metric.FILE_EER + 1e-12
        and (channel_delta >= -1e-12).all()
    )
    summary = {
        "selection_data": list(DEV_BANKS),
        "locked_data_opened": False,
        "weights": list(WEIGHTS),
        "selected_wpt_weight_inside_query_expert": selected_weight,
        "query_weight_inside_consensus": 1 - selected_weight,
        "baseline_robust_score": robust_score(
            baseline_eers, list(range(len(DEV_BANKS)))
        ),
        "selected_robust_score": selection,
        "baseline_mean_music_eer": float(baseline_eers.mean()),
        "selected_mean_music_eer": float(selected_eers.mean()),
        "baseline_worst_music_eer": float(baseline_eers.max()),
        "selected_worst_music_eer": float(selected_eers.max()),
        "lodo_non_regressing_banks": int((lodo.HELD_OUT_DELTA <= 1e-12).sum()),
        "lodo_banks": len(lodo),
        "factorial_v47_file_eer_delta": float(
            fused_metric.FILE_EER - base_metric.FILE_EER
        ),
        "factorial_v47_music_eer_delta": float(
            fused_metric.MUSIC_EER - base_metric.MUSIC_EER
        ),
        "minimum_factorial_channel_ads_delta": float(channel_delta.min()),
        "decision": "adopt_candidate" if accepted else "reject_candidate",
        "deployment_note": (
            "Replace the raw component-query Music expert with the 50/50 "
            "query/WPT logit consensus before the existing 5% v47 Music "
            "residual. The WPT forward already runs in v48."
            if accepted else
            "Do not alter deployment; at least one predeclared dev safety "
            "condition failed."
        ),
    }
    args.output_dir.mkdir(parents=True, exist_ok=False)
    sweep.to_csv(args.output_dir / "bank_sweep.csv", index=False)
    lodo.to_csv(args.output_dir / "leave_one_domain_out.csv", index=False)
    all_predictions.loc[
        all_predictions.WPT_WEIGHT.eq(selected_weight)
    ].to_csv(args.output_dir / "selected_dev_predictions.csv", index=False)
    v47_metrics.to_csv(args.output_dir / "factorial_v47_metrics.csv", index=False)
    v47_channels.to_csv(args.output_dir / "factorial_v47_channels.csv", index=False)
    v47_predictions.to_csv(
        args.output_dir / "factorial_v47_predictions.csv", index=False
    )
    (args.output_dir / "selection.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    print(sweep.loc[sweep.WPT_WEIGHT.eq(selected_weight)].to_string(index=False))
    print(lodo.to_string(index=False))
    print(v47_metrics.to_string(index=False))


if __name__ == "__main__":
    main()
