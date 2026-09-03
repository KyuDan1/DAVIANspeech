#!/usr/bin/env python3
"""Reconstruct v18 and audit an additional codec-invariant expert.

The script uses only frozen prediction files.  It follows the deployed order
(LME+SPEAR -> temporal -> MERT -> fakeprint -> v18 invariant -> codec expert)
and writes both a weight sweep and subgroup diagnostics.  This makes the
telephone experiment reproducible without rerunning any audio backbone.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from evaluate_diagnostic import GROUP_COLUMNS, score_frame  # noqa: E402


PREDICTION_COLUMNS = (
    "FILE_FAKE_PROB", "VOICE_FAKE_PROB", "MUSIC_FAKE_PROB",
)


@dataclass(frozen=True)
class AuditSpec:
    truth: Path
    anchor: Path
    temporal_name: str
    mert: Path
    fakeprint_name: str


AUDITS = {
    "factorial": AuditSpec(
        ROOT / "data/eval/factorial_eval_1200_v2/truth_holdout.csv",
        ROOT / "output/factorial_eval_1200_v2_holdout_lme_spear.csv",
        "factorial_eval_1200_v2_holdout",
        ROOT / "reports/sofia_mert_v1/factorial_holdout.csv",
        "factorial_eval_1200_v2",
    ),
    "phone": AuditSpec(
        ROOT / "data/eval/phone_factorial_1200_v1/truth.csv",
        ROOT / "output/phone_factorial_1200_v1_lme_spear.csv",
        "phone_factorial_1200_v1",
        ROOT / "reports/sofia_mert_v1/phone_factorial.csv",
        "phone_factorial_1200_v1",
    ),
    "yue": AuditSpec(
        ROOT / "data/eval/yue_cross_component_audit_v1/truth.csv",
        ROOT / "output/yue_cross_component_audit_v1_lme_spear.csv",
        "yue_cross_component_audit_v1",
        ROOT / "reports/sofia_mert_v1/yue.csv",
        "yue_cross_component_audit_v1",
    ),
}


def logit(values) -> np.ndarray:
    values = np.clip(np.asarray(values, dtype=np.float64), 1e-5, 1 - 1e-5)
    return np.log(values) - np.log1p(-values)


def fuse(anchor, expert, weight: float) -> np.ndarray:
    mixed = (1 - weight) * logit(anchor) + weight * logit(expert)
    return 1 / (1 + np.exp(-mixed))


def indexed(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, dtype={"ID": str})
    if frame.ID.duplicated().any():
        raise ValueError(f"Duplicate IDs in {path}")
    return frame.set_index("ID")


def select_predictions(
    frame: pd.DataFrame, dataset: str, ids: pd.Index,
) -> pd.DataFrame:
    selected = frame.loc[frame.DATASET.eq(dataset)].set_index("ID")
    missing = ids.difference(selected.index)
    if len(missing):
        raise ValueError(f"Missing {len(missing)} predictions for {dataset}")
    return selected.loc[ids]


def reconstruct_v18(
    spec: AuditSpec,
    temporal: pd.DataFrame,
    fakeprint: pd.DataFrame,
    old_invariant: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    anchor = indexed(spec.anchor)
    truth = indexed(spec.truth).loc[anchor.index]
    temporal_scores = select_predictions(
        temporal, spec.temporal_name, anchor.index
    )
    for column in PREDICTION_COLUMNS:
        anchor[column] = fuse(anchor[column], temporal_scores[column], 0.05)

    mert = indexed(spec.mert).loc[anchor.index, "SOFIA_MERT_FAKE_PROB"]
    anchor["FILE_FAKE_PROB"] = fuse(anchor.FILE_FAKE_PROB, mert, 0.025)
    anchor["MUSIC_FAKE_PROB"] = fuse(anchor.MUSIC_FAKE_PROB, mert, 0.0125)

    fakeprint_scores = select_predictions(
        fakeprint, spec.fakeprint_name, anchor.index
    ).MODERN_FAKEPRINT_PROB
    for column in ("FILE_FAKE_PROB", "MUSIC_FAKE_PROB"):
        anchor[column] = fuse(anchor[column], fakeprint_scores, 0.025)

    invariant_scores = select_predictions(
        old_invariant, spec.temporal_name, anchor.index
    )
    for column in PREDICTION_COLUMNS:
        anchor[column] = fuse(anchor[column], invariant_scores[column], 0.05)
    return truth, anchor


def diagnostics(
    dataset: str, truth: pd.DataFrame, prediction: pd.DataFrame,
    min_group_size: int,
) -> list[dict]:
    frame = truth.join(prediction[list(PREDICTION_COLUMNS)], how="inner")
    records = [{"DATASET": dataset, "GROUP": "ALL", "VALUE": "ALL",
                **score_frame(frame)}]
    for column in GROUP_COLUMNS:
        if column not in frame:
            continue
        for value, group in frame.groupby(column, dropna=False):
            if len(group) >= min_group_size:
                records.append({
                    "DATASET": dataset, "GROUP": column, "VALUE": str(value),
                    **score_frame(group),
                })
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--temporal-predictions", type=Path,
        default=ROOT / "reports/temporal_dual_domain_hybrid/ensemble_audit/predictions.csv",
    )
    parser.add_argument(
        "--fakeprint-predictions", type=Path,
        default=ROOT / "reports/modern_fakeprint_v1/independent_margin_scores.csv",
    )
    parser.add_argument(
        "--old-invariant-predictions", type=Path,
        default=ROOT / "reports/invariant_dual_domain_v2/dec_n1_audit/predictions.csv",
    )
    parser.add_argument("--codec-predictions", type=Path, required=True)
    parser.add_argument("--weights", type=float, nargs="+", default=[
        0, 0.005, 0.01, 0.025, 0.05, 0.075, 0.10, 0.15,
    ])
    parser.add_argument("--selected-weight", type=float, default=0.01)
    parser.add_argument("--min-group-size", type=int, default=10)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if any(weight < 0 or weight > 1 for weight in args.weights):
        parser.error("weights must be in [0, 1]")
    if not 0 <= args.selected_weight <= 1:
        parser.error("selected-weight must be in [0, 1]")

    temporal = pd.read_csv(args.temporal_predictions, dtype={"ID": str})
    fakeprint = pd.read_csv(args.fakeprint_predictions, dtype={"ID": str})
    old_invariant = pd.read_csv(
        args.old_invariant_predictions, dtype={"ID": str}
    )
    codec = pd.read_csv(args.codec_predictions, dtype={"ID": str})
    args.output_dir.mkdir(parents=True, exist_ok=True)

    sweep_rows: list[dict] = []
    diagnostic_rows: list[dict] = []
    for name, spec in AUDITS.items():
        truth, v18 = reconstruct_v18(
            spec, temporal, fakeprint, old_invariant
        )
        codec_scores = select_predictions(codec, spec.temporal_name, v18.index)
        for weight in args.weights:
            candidate = v18.copy()
            for column in PREDICTION_COLUMNS:
                candidate[column] = fuse(
                    candidate[column], codec_scores[column], weight
                )
            sweep_rows.append({
                "DATASET": name, "CODEC_WEIGHT": weight,
                **score_frame(truth.join(
                    candidate[list(PREDICTION_COLUMNS)], how="inner"
                )),
            })

        selected = v18.copy()
        for column in PREDICTION_COLUMNS:
            selected[column] = fuse(
                selected[column], codec_scores[column], args.selected_weight
            )
        selected.reset_index().to_csv(
            args.output_dir / f"{name}_w{args.selected_weight:g}.csv", index=False
        )
        diagnostic_rows.extend(diagnostics(
            name, truth, selected, args.min_group_size
        ))

    sweep = pd.DataFrame(sweep_rows)
    sweep.to_csv(args.output_dir / "fusion_sweep.csv", index=False)
    diagnostics_frame = pd.DataFrame(diagnostic_rows)
    diagnostics_frame.to_csv(args.output_dir / "selected_diagnostics.csv", index=False)
    print(sweep[[
        "DATASET", "CODEC_WEIGHT", "FILE_EER", "VOICE_EER",
        "MUSIC_EER", "ADS",
    ]].to_string(index=False))


if __name__ == "__main__":
    main()
