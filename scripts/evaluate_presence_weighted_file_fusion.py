#!/usr/bin/env python3
"""Reconstruct v27 and audit soft component-to-File consistency."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from evaluate_codec_invariant_fusion import (  # noqa: E402
    AUDITS, fuse, indexed, reconstruct_v18, select_predictions,
)
from evaluate_diagnostic import official_eer, score_frame  # noqa: E402
from evaluate_long_horizon_music_routing import ROUTERS, selected  # noqa: E402
from long_horizon_music_inference import _fuse  # noqa: E402
from presence_weighted_file_fusion import component_file_evidence  # noqa: E402


def load(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, dtype={"ID": str})


def reconstruct_dev_v18(
    temporal: pd.DataFrame,
    fakeprint: pd.DataFrame,
    invariant: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    truth = indexed(ROOT / "data/eval/factorial_eval_1200_v2/truth_dev.csv")
    anchor = indexed(
        ROOT / "output/factorial_eval_1200_v2_lme_spear.csv"
    ).loc[truth.index].copy()
    temporal_score = select_predictions(
        temporal, "factorial_eval_1200_v2_dev", anchor.index
    )
    for column in ("FILE_FAKE_PROB", "VOICE_FAKE_PROB", "MUSIC_FAKE_PROB"):
        anchor[column] = fuse(anchor[column], temporal_score[column], 0.05)
    mert = indexed(
        ROOT / "reports/sofia_mert_v1/factorial_dev_safe_exact.csv"
    ).loc[anchor.index, "SOFIA_MERT_FAKE_PROB"]
    anchor["FILE_FAKE_PROB"] = fuse(anchor.FILE_FAKE_PROB, mert, 0.025)
    anchor["MUSIC_FAKE_PROB"] = fuse(anchor.MUSIC_FAKE_PROB, mert, 0.0125)
    fakeprint_score = select_predictions(
        fakeprint, "factorial_eval_1200_v2", anchor.index
    ).MODERN_FAKEPRINT_PROB
    anchor["FILE_FAKE_PROB"] = fuse(
        anchor.FILE_FAKE_PROB, fakeprint_score, 0.025
    )
    anchor["MUSIC_FAKE_PROB"] = fuse(
        anchor.MUSIC_FAKE_PROB, fakeprint_score, 0.025
    )
    invariant_score = select_predictions(
        invariant, "factorial_eval_1200_v2_dev", anchor.index
    )
    for column in ("FILE_FAKE_PROB", "VOICE_FAKE_PROB", "MUSIC_FAKE_PROB"):
        anchor[column] = fuse(anchor[column], invariant_score[column], 0.05)
    return truth, anchor


def routed_v27(
    name: str,
    temporal_dev: pd.DataFrame,
    temporal_audit: pd.DataFrame,
    fakeprint: pd.DataFrame,
    invariant_dev: pd.DataFrame,
    invariant_audit: pd.DataFrame,
    average_dev: pd.DataFrame,
    robust_dev: pd.DataFrame,
    average_audit: pd.DataFrame,
    robust_audit: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if name == "dev":
        truth, anchor = reconstruct_dev_v18(
            temporal_dev, fakeprint, invariant_dev
        )
        dataset = "factorial_eval_1200_v2_dev"
        average, robust = average_dev, robust_dev
        router_path = (
            ROOT / "reports/phone_presence_probe_v1/"
            "factorial_eval_1200_v2_router.csv"
        )
    else:
        specification = AUDITS[name]
        truth, anchor = reconstruct_v18(
            specification, temporal_audit, fakeprint, invariant_audit
        )
        dataset = specification.temporal_name
        average, robust = average_audit, robust_audit
        router_path = ROUTERS.get(name)

    phone_mask = np.zeros(len(anchor), dtype=bool)
    if router_path is not None:
        router = load(router_path).set_index("ID")
        phone_mask = router.loc[anchor.index, "IS_PHONE"].to_numpy(bool)
    average_score = selected(average, dataset, anchor.index)
    robust_score = selected(robust, dataset, anchor.index)
    expert = np.where(
        phone_mask, average_score, _fuse(average_score, robust_score, 0.60)
    )
    prediction = anchor.copy()
    prediction["MUSIC_FAKE_PROB"] = _fuse(
        prediction.MUSIC_FAKE_PROB, expert, 0.40
    )
    file_route = prediction.MUSIC_PRESENT_PROB.to_numpy() >= 0.70
    prediction["FILE_FAKE_PROB"] = _fuse(
        prediction.FILE_FAKE_PROB, expert, 0.20 * file_route
    )
    return truth, prediction


def apply_candidate(prediction: pd.DataFrame, weight: float) -> pd.DataFrame:
    result = prediction.copy()
    evidence = component_file_evidence(
        result.VOICE_FAKE_PROB, result.MUSIC_FAKE_PROB,
        result.VOICE_PRESENT_PROB, result.MUSIC_PRESENT_PROB,
        presence_logit_weight=0.50,
    )
    result["FILE_FAKE_PROB"] = _fuse(
        result.FILE_FAKE_PROB, evidence, weight
    )
    return result


def diagnostics(
    name: str, truth: pd.DataFrame, prediction: pd.DataFrame,
) -> list[dict]:
    frame = truth.join(prediction)
    groups = [("ALL", "ALL", frame)]
    for column in ("AUDIO_TYPE", "MIX_MODE", "CONDITION", "CHANNEL"):
        if column in frame:
            groups.extend(
                (column, str(value), group)
                for value, group in frame.groupby(column, dropna=False)
                if len(group) >= 20
            )
    rows = []
    for group_name, value, group in groups:
        file_eer = (
            official_eer(group.FILE_FAKE.astype(int), group.FILE_FAKE_PROB)
            if group.FILE_FAKE.nunique() == 2 else np.nan
        )
        rows.append({
            "DATASET": name, "GROUP": group_name, "VALUE": value,
            "N": len(group), "FILE_EER": file_eer,
        })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--weights", type=float, nargs="+",
        default=[0, 0.3, 0.4, 0.5, 0.6, 0.7],
    )
    parser.add_argument("--selected-weight", type=float, default=0.5)
    parser.add_argument(
        "--output-dir", type=Path,
        default=ROOT / "reports/presence_weighted_file_v28",
    )
    args = parser.parse_args()
    if any(not 0 <= weight <= 1 for weight in args.weights):
        parser.error("weights must be in [0, 1]")
    if not 0 <= args.selected_weight <= 1:
        parser.error("selected-weight must be in [0, 1]")

    common = {
        "temporal_dev": load(
            ROOT / "reports/temporal_dual_domain_hybrid/ensemble_dev/predictions.csv"
        ),
        "temporal_audit": load(
            ROOT / "reports/temporal_dual_domain_hybrid/ensemble_audit/predictions.csv"
        ),
        "fakeprint": load(
            ROOT / "reports/modern_fakeprint_v1/independent_margin_scores.csv"
        ),
        "invariant_dev": load(
            ROOT / "reports/invariant_dual_domain_v2/dec_n1/dev_predictions.csv"
        ),
        "invariant_audit": load(
            ROOT / "reports/invariant_dual_domain_v2/dec_n1_audit/predictions.csv"
        ),
        "average_dev": load(
            ROOT / "reports/long_horizon_music_v2/random_both_all/dev_predictions.csv"
        ),
        "robust_dev": load(
            ROOT / "reports/long_horizon_music_v2/groupdro_0p10/dev_predictions.csv"
        ),
        "average_audit": load(
            ROOT / "reports/long_horizon_music_v2/random_both_all_audit/predictions.csv"
        ),
        "robust_audit": load(
            ROOT / "reports/long_horizon_music_v2/groupdro_0p10_audit/predictions.csv"
        ),
    }
    banks = {
        name: routed_v27(name, **common)
        for name in ("dev", "factorial", "phone", "yue")
    }
    rows = []
    group_rows = []
    for name, (truth, prediction) in banks.items():
        for weight in args.weights:
            candidate = apply_candidate(prediction, weight)
            rows.append({
                "DATASET": name, "FILE_WEIGHT": weight,
                **score_frame(truth.join(candidate)),
            })
            for record in diagnostics(name, truth, candidate):
                group_rows.append({"FILE_WEIGHT": weight, **record})

    combined_truth, combined_prediction = [], []
    for name in ("factorial", "phone"):
        truth, prediction = banks[name]
        truth, prediction = truth.copy(), prediction.copy()
        truth.index = name + "_" + truth.index
        prediction.index = truth.index
        combined_truth.append(truth)
        combined_prediction.append(prediction)
    for weight in args.weights:
        candidate = apply_candidate(pd.concat(combined_prediction), weight)
        rows.append({
            "DATASET": "factorial_plus_phone", "FILE_WEIGHT": weight,
            **score_frame(pd.concat(combined_truth).join(candidate)),
        })

    selected_diagnostics = []
    for name, (truth, prediction) in banks.items():
        selected_diagnostics.extend(diagnostics(
            name, truth, apply_candidate(prediction, args.selected_weight)
        ))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    sweep = pd.DataFrame(rows)
    sweep.to_csv(args.output_dir / "sweep.csv", index=False)
    pd.DataFrame(group_rows).to_csv(
        args.output_dir / "group_sweep.csv", index=False
    )
    pd.DataFrame(selected_diagnostics).to_csv(
        args.output_dir / "selected_diagnostics.csv", index=False
    )
    print(sweep[[
        "DATASET", "FILE_WEIGHT", "FILE_EER", "VOICE_EER", "MUSIC_EER", "ADS",
    ]].to_string(index=False))


if __name__ == "__main__":
    main()
