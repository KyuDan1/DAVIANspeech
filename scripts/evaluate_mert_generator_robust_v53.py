#!/usr/bin/env python3
"""Frozen retrospective audit of the MERT v53 head on codec banks v4-v7."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]

from evaluate_diagnostic import official_eer  # noqa: E402
from mert_generator_robust_music import (  # noqa: E402
    logit_residual, predict_mert_generator_robust_music,
)
from train_mert_generator_robust_music_v53 import load_statistics  # noqa: E402


DATASETS = (
    "codec_mixed_dev_v4", "codec_mixed_blind_v5",
    "codec_mixed_blind_v6", "codec_mixed_blind_v7",
)
ANCHORS = {
    "codec_mixed_dev_v4": ROOT / "reports/v50_candidate_codec_dev_v4/predictions.csv",
    "codec_mixed_blind_v5": ROOT / "reports/v50_candidate_blind_v5/predictions.csv",
    "codec_mixed_blind_v6":
        ROOT / "reports/v50_blind_v6_one_shot/v50_frozen/output/submission.csv",
    "codec_mixed_blind_v7":
        ROOT / "reports/sota_candidate_v52_blind_v7_one_shot/v50_anchor/output/submission.csv",
}
V51 = {
    "codec_mixed_dev_v4": ROOT / "reports/music_expert_v51/v51_frozen_v4/predictions.csv",
    "codec_mixed_blind_v5": ROOT / "reports/music_expert_v51/v51_frozen_v5/predictions.csv",
}


def eer(truth: pd.DataFrame, score: np.ndarray) -> float:
    return float(official_eer(truth.MUSIC_FAKE.astype(int), score))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--statistics", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--residual-weight", type=float, default=.025)
    args = parser.parse_args()
    if args.residual_weight != .025:
        raise ValueError("v53 retrospective protocol freezes residual weight at 0.025")

    datasets, ids, statistics = load_statistics(args.statistics)
    expert = predict_mert_generator_robust_music(
        statistics, args.checkpoint, device=args.device,
    )
    rows, slices, predictions = [], [], []
    for name in DATASETS:
        mask = datasets == name
        bank_ids, bank_expert = ids[mask], expert[mask]
        truth = pd.read_csv(
            ROOT / "data/eval" / name / "truth.csv", dtype={"ID": str},
        ).set_index("ID").loc[bank_ids].reset_index()
        anchor = pd.read_csv(ANCHORS[name], dtype={"ID": str}).set_index("ID")
        anchor = anchor.loc[bank_ids, "MUSIC_FAKE_PROB"].to_numpy(np.float32)
        fused = logit_residual(anchor, bank_expert, args.residual_weight)
        values = {
            "V50": anchor, "EXPERT": bank_expert, "V50_PLUS_025": fused,
        }
        if name in V51:
            v51 = pd.read_csv(V51[name], dtype={"ID": str}).set_index("ID")
            v51 = v51.loc[bank_ids, "MUSIC_FAKE_PROB"].to_numpy(np.float32)
            values["V51"] = v51
            values["V51_PLUS_025"] = logit_residual(
                v51, bank_expert, args.residual_weight,
            )
        for method, score in values.items():
            current = eer(truth, score)
            rows.append({
                "DATASET": name, "METHOD": method,
                "MUSIC_EER": current, "MUSIC_SCORE": 1 - current,
                "ROLE": "retrospective_only_nonselection",
            })
            for column in ("CHANNEL", "MIX_MODE", "VOICE_FAKE"):
                for value, indices in truth.groupby(column).groups.items():
                    subset = truth.loc[indices]
                    if subset.MUSIC_FAKE.nunique() < 2:
                        continue
                    slices.append({
                        "DATASET": name, "METHOD": method, "SLICE": column,
                        "VALUE": value, "N": len(indices),
                        "MUSIC_EER": eer(subset, score[np.asarray(indices)]),
                    })
        for index, item in enumerate(bank_ids):
            record = {
                "DATASET": name, "ID": item,
                "EXPERT_MUSIC_FAKE_PROB": bank_expert[index],
            }
            record.update({f"{key}_MUSIC_FAKE_PROB": value[index]
                           for key, value in values.items()})
            predictions.append(record)

    metrics = pd.DataFrame(rows)
    anchor = metrics.loc[metrics.METHOD.eq("V50")].set_index("DATASET")
    candidate = metrics.loc[metrics.METHOD.eq("V50_PLUS_025")].set_index("DATASET")
    deltas = anchor.MUSIC_EER - candidate.MUSIC_EER
    summary = {
        "residual_weight_frozen_before_retrospective": args.residual_weight,
        "v4_v5_v6_v7_used_for_selection": False,
        "mean_music_eer_reduction": float(deltas.mean()),
        "worst_music_eer_reduction": float(deltas.min()),
        "per_dataset_music_eer_reduction": {
            key: float(value) for key, value in deltas.items()
        },
        "ads_gain_if_other_axes_fixed": float(.3 * deltas.mean()),
        "total_gain_if_other_axes_fixed": float(.27 * deltas.mean()),
        "music_eer_reduction_needed_for_best_to_sota_if_music_only":
            float((.821160 - .7363412698) / .3),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(args.output_dir / "metrics.csv", index=False)
    pd.DataFrame(slices).to_csv(args.output_dir / "slices.csv", index=False)
    pd.DataFrame(predictions).to_csv(args.output_dir / "predictions.csv", index=False)
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8",
    )
    print(metrics.to_string(index=False))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
