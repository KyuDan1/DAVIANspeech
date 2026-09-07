#!/usr/bin/env python3
"""Audit frozen fixed-MoE and narrow-band-routed music fusion after v18."""

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
    AUDITS,
    reconstruct_v18,
)
from evaluate_diagnostic import score_frame  # noqa: E402
from long_horizon_music_inference import _fuse  # noqa: E402


ROUTERS = {
    "factorial": ROOT / "reports/phone_presence_probe_v1/factorial_eval_1200_v2_router.csv",
    "phone": ROOT / "reports/phone_presence_probe_v1/phone_factorial_1200_v1_router.csv",
}


def selected(frame: pd.DataFrame, dataset: str, ids: pd.Index) -> np.ndarray:
    block = frame.loc[frame.DATASET.eq(dataset)].set_index("ID")
    missing = ids.difference(block.index)
    if len(missing):
        raise ValueError(f"Missing {len(missing)} music predictions for {dataset}")
    return block.loc[ids, "LONG_HORIZON_MUSIC_PROB"].to_numpy()


def apply_candidate(
    anchor: pd.DataFrame,
    average: np.ndarray,
    robust: np.ndarray,
    phone_mask: np.ndarray,
    robust_weight: float,
    phone_robust_weight: float,
    music_weight: float,
    file_weight: float,
    presence_threshold: float,
) -> pd.DataFrame:
    internal_weight = np.where(
        phone_mask, phone_robust_weight, robust_weight
    )
    expert = _fuse(average, robust, internal_weight)
    result = anchor.copy()
    result["MUSIC_FAKE_PROB"] = _fuse(
        result.MUSIC_FAKE_PROB, expert, music_weight
    )
    file_route = result.MUSIC_PRESENT_PROB.to_numpy() >= presence_threshold
    result["FILE_FAKE_PROB"] = _fuse(
        result.FILE_FAKE_PROB, expert, file_weight * file_route
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--average-predictions", type=Path,
        default=ROOT / "reports/long_horizon_music_v2/random_both_all_audit/predictions.csv",
    )
    parser.add_argument(
        "--robust-predictions", type=Path,
        default=ROOT / "reports/long_horizon_music_v2/groupdro_0p10_audit/predictions.csv",
    )
    parser.add_argument(
        "--temporal-predictions", type=Path,
        default=ROOT / "reports/temporal_dual_domain_hybrid/ensemble_audit/predictions.csv",
    )
    parser.add_argument(
        "--fakeprint-predictions", type=Path,
        default=ROOT / "reports/modern_fakeprint_v1/independent_margin_scores.csv",
    )
    parser.add_argument(
        "--invariant-predictions", type=Path,
        default=ROOT / "reports/invariant_dual_domain_v2/dec_n1_audit/predictions.csv",
    )
    parser.add_argument("--robust-weight", type=float, default=0.60)
    parser.add_argument("--phone-robust-weight", type=float, default=0.0)
    parser.add_argument("--music-weight", type=float, default=0.40)
    parser.add_argument("--file-weight", type=float, default=0.20)
    parser.add_argument("--presence-threshold", type=float, default=0.70)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    for value in (
        args.robust_weight, args.phone_robust_weight, args.music_weight,
        args.file_weight, args.presence_threshold,
    ):
        if not 0 <= value <= 1:
            parser.error("all weights and thresholds must be in [0, 1]")

    average = pd.read_csv(args.average_predictions, dtype={"ID": str})
    robust = pd.read_csv(args.robust_predictions, dtype={"ID": str})
    temporal = pd.read_csv(args.temporal_predictions, dtype={"ID": str})
    fakeprint = pd.read_csv(args.fakeprint_predictions, dtype={"ID": str})
    invariant = pd.read_csv(args.invariant_predictions, dtype={"ID": str})
    rows = []
    combined = []
    for name, specification in AUDITS.items():
        truth, v18 = reconstruct_v18(
            specification, temporal, fakeprint, invariant
        )
        average_score = selected(
            average, specification.temporal_name, v18.index
        )
        robust_score = selected(
            robust, specification.temporal_name, v18.index
        )
        phone_mask = np.zeros(len(v18), dtype=bool)
        if name in ROUTERS:
            route = pd.read_csv(ROUTERS[name], dtype={"ID": str}).set_index("ID")
            phone_mask = route.loc[v18.index, "IS_PHONE"].to_numpy(bool)
        fixed = apply_candidate(
            v18, average_score, robust_score, phone_mask,
            args.robust_weight, args.robust_weight,
            args.music_weight, args.file_weight, args.presence_threshold,
        )
        routed = apply_candidate(
            v18, average_score, robust_score, phone_mask,
            args.robust_weight, args.phone_robust_weight,
            args.music_weight, args.file_weight, args.presence_threshold,
        )
        for method, prediction in (("v18", v18), ("fixed_moe", fixed),
                                   ("phone_routed_moe", routed)):
            metrics = score_frame(truth.join(prediction, how="inner"))
            rows.append({
                "DATASET": name, "METHOD": method,
                "PHONE_COUNT": int(phone_mask.sum()), **metrics,
            })
        if name in {"factorial", "phone"}:
            prefix = name + "_"
            truth_part = truth.copy()
            routed_part = routed.copy()
            truth_part.index = [prefix + value for value in truth_part.index]
            routed_part.index = truth_part.index
            v18_part = v18.copy()
            fixed_part = fixed.copy()
            v18_part.index = truth_part.index
            fixed_part.index = truth_part.index
            combined.append((
                truth_part, v18_part, fixed_part, routed_part,
                int(phone_mask.sum()),
            ))

    if len(combined) == 2:
        truth = pd.concat([item[0] for item in combined])
        phone_count = sum(item[4] for item in combined)
        for method, position in (
            ("v18", 1), ("fixed_moe", 2), ("phone_routed_moe", 3),
        ):
            prediction = pd.concat([item[position] for item in combined])
            rows.append({
                "DATASET": "factorial_plus_phone", "METHOD": method,
                "PHONE_COUNT": phone_count,
                **score_frame(truth.join(prediction, how="inner")),
            })
    result = pd.DataFrame(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)
    print(result[[
        "DATASET", "METHOD", "PHONE_COUNT", "FILE_EER", "VOICE_EER",
        "MUSIC_EER", "ADS",
    ]].to_string(index=False))


if __name__ == "__main__":
    main()
