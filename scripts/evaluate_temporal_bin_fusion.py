#!/usr/bin/env python3
"""Audit SPEAR temporal-bin fusion on the exact reconstructed v27/v28 stack."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from evaluate_diagnostic import score_frame  # noqa: E402
from evaluate_presence_weighted_file_fusion import (  # noqa: E402
    apply_candidate, load, routed_v27,
)
from long_horizon_music_inference import _fuse  # noqa: E402


def common_inputs() -> dict[str, pd.DataFrame]:
    return {
        "temporal_dev": load(ROOT / "reports/temporal_dual_domain_hybrid/ensemble_dev/predictions.csv"),
        "temporal_audit": load(ROOT / "reports/temporal_dual_domain_hybrid/ensemble_audit/predictions.csv"),
        "fakeprint": load(ROOT / "reports/modern_fakeprint_v1/independent_margin_scores.csv"),
        "invariant_dev": load(ROOT / "reports/invariant_dual_domain_v2/dec_n1/dev_predictions.csv"),
        "invariant_audit": load(ROOT / "reports/invariant_dual_domain_v2/dec_n1_audit/predictions.csv"),
        "average_dev": load(ROOT / "reports/long_horizon_music_v2/random_both_all/dev_predictions.csv"),
        "robust_dev": load(ROOT / "reports/long_horizon_music_v2/groupdro_0p10/dev_predictions.csv"),
        "average_audit": load(ROOT / "reports/long_horizon_music_v2/random_both_all_audit/predictions.csv"),
        "robust_audit": load(ROOT / "reports/long_horizon_music_v2/groupdro_0p10_audit/predictions.csv"),
    }


def expert_for(predictions: pd.DataFrame, dataset: str, ids: pd.Index) -> np.ndarray:
    block = predictions[predictions.DATASET.eq(dataset)].set_index("ID")
    missing = set(ids) - set(block.index)
    if missing:
        raise FileNotFoundError(f"temporal expert misses {len(missing)} IDs for {dataset}")
    return block.loc[ids, "TEMPORAL_BIN_MUSIC_PROB"].to_numpy(np.float64)


def apply_expert(
    prediction: pd.DataFrame, expert: np.ndarray,
    music_weight: float, file_weight: float,
    file_consistency_update_weight: float = 1.0,
) -> pd.DataFrame:
    baseline = apply_candidate(prediction, .50)
    result = prediction.copy()
    result["MUSIC_FAKE_PROB"] = _fuse(
        result.MUSIC_FAKE_PROB.to_numpy(), expert, music_weight
    )
    routed = result.MUSIC_PRESENT_PROB.to_numpy() >= .70
    result["FILE_FAKE_PROB"] = _fuse(
        result.FILE_FAKE_PROB.to_numpy(), expert, file_weight * routed
    )
    # Recompute the already-selected v28 logical consistency after Music moves,
    # then retain part of the old v28 File ranking for cross-domain stability.
    updated = apply_candidate(result, .50)
    updated["FILE_FAKE_PROB"] = _fuse(
        baseline.FILE_FAKE_PROB, updated.FILE_FAKE_PROB,
        file_consistency_update_weight,
    )
    return updated


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expert-predictions", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--music-weights", type=float, nargs="+",
                        default=[0, .025, .05, .075, .10, .15, .20, .30])
    parser.add_argument("--file-ratios", type=float, nargs="+", default=[0, .5])
    parser.add_argument("--file-update-weights", type=float, nargs="+",
                        default=[0, .50, .75, 1.0])
    args = parser.parse_args()
    expert_predictions = load(args.expert_predictions)
    common = common_inputs()
    banks = {name: routed_v27(name, **common) for name in ("dev", "factorial", "phone", "yue")}
    dataset_names = {
        "dev": "factorial_eval_1200_v2",
        "factorial": "factorial_eval_1200_v2",
        "phone": "phone_factorial_1200_v1",
        "yue": "yue_cross_component_audit_v1",
    }
    rows, predictions = [], []
    for name, (truth, anchor) in banks.items():
        expert = expert_for(expert_predictions, dataset_names[name], anchor.index)
        for music_weight in args.music_weights:
            for file_ratio in args.file_ratios:
                for update_weight in args.file_update_weights:
                    result = apply_expert(
                        anchor, expert, music_weight, music_weight * file_ratio,
                        update_weight,
                    )
                    rows.append({
                        "DATASET": name, "MUSIC_WEIGHT": music_weight,
                        "FILE_RATIO": file_ratio,
                        "FILE_UPDATE_WEIGHT": update_weight,
                        **score_frame(truth.join(result)),
                    })
                    if (
                        music_weight in (0, .20, .30, .40, .50)
                        and update_weight in (0, .75, 1)
                    ):
                        block = result.reset_index().copy()
                        block.insert(0, "DATASET", name)
                        block.insert(1, "MUSIC_WEIGHT", music_weight)
                        block.insert(2, "FILE_RATIO", file_ratio)
                        block.insert(3, "FILE_UPDATE_WEIGHT", update_weight)
                        predictions.append(block)
    output = pd.DataFrame(rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output_dir / "sweep.csv", index=False)
    pd.concat(predictions, ignore_index=True).to_csv(
        args.output_dir / "selected_predictions.csv", index=False
    )
    print(output[[
        "DATASET", "MUSIC_WEIGHT", "FILE_RATIO", "FILE_UPDATE_WEIGHT",
        "FILE_EER", "VOICE_EER",
        "MUSIC_EER", "ADS",
    ]].to_string(index=False))


if __name__ == "__main__":
    main()
