#!/usr/bin/env python3
"""Audit independent File-attention and Music-bin residuals on exact v18."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from evaluate_codec_invariant_fusion import AUDITS, fuse, reconstruct_v18  # noqa: E402
from evaluate_diagnostic import score_frame  # noqa: E402
from evaluate_presence_weighted_file_fusion import reconstruct_dev_v18  # noqa: E402


DATASETS = {
    "dev": "factorial_eval_1200_v2",
    "factorial": "factorial_eval_1200_v2",
    "phone": "phone_factorial_1200_v1",
    "yue": "yue_cross_component_audit_v1",
}
ROUTERS = {
    "dev": ROOT / "reports/phone_presence_probe_v1/factorial_eval_1200_v2_router.csv",
    "factorial": ROOT / "reports/phone_presence_probe_v1/factorial_eval_1200_v2_router.csv",
    "phone": ROOT / "reports/phone_presence_probe_v1/phone_factorial_1200_v1_router.csv",
}


def load(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, dtype={"ID": str})


def select(frame: pd.DataFrame, dataset: str, ids: pd.Index) -> pd.DataFrame:
    block = frame.loc[frame.DATASET.eq(dataset)].set_index("ID")
    if block.index.has_duplicates:
        raise ValueError(f"duplicate expert IDs for {dataset}")
    missing = ids.difference(block.index)
    if len(missing):
        raise ValueError(f"expert misses {len(missing)} IDs for {dataset}")
    return block.loc[ids]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--attention-predictions", type=Path,
        default=ROOT / "reports/spear_temporal_attention_v1/attention_channel_s12_audit/predictions.csv",
    )
    parser.add_argument(
        "--music-predictions", type=Path,
        default=ROOT / "reports/spear_temporal_bin_v1/mlp64_seed05_audit/predictions.csv",
    )
    parser.add_argument("--file-weight", type=float, default=.20)
    parser.add_argument("--phone-file-weight", type=float, default=.25)
    parser.add_argument(
        "--music-weights", type=float, nargs="+",
        default=[0, .10, .20, .30, .40, .50, .60],
    )
    parser.add_argument(
        "--phone-music-weights", type=float, nargs="+",
        help="telephone-routed Music weights; defaults to each global weight",
    )
    parser.add_argument(
        "--output", type=Path,
        default=ROOT / "reports/spear_temporal_bin_v1/v18_attention_music/sweep.csv",
    )
    args = parser.parse_args()
    phone_music_weights = (
        args.phone_music_weights
        if args.phone_music_weights is not None else []
    )
    values = [
        args.file_weight, args.phone_file_weight,
        *args.music_weights, *phone_music_weights,
    ]
    if any(not 0 <= value <= 1 for value in values):
        parser.error("all fusion weights must be in [0, 1]")

    temporal_dev = load(ROOT / "reports/temporal_dual_domain_hybrid/ensemble_dev/predictions.csv")
    temporal_audit = load(ROOT / "reports/temporal_dual_domain_hybrid/ensemble_audit/predictions.csv")
    fakeprint = load(ROOT / "reports/modern_fakeprint_v1/independent_margin_scores.csv")
    invariant_dev = load(ROOT / "reports/invariant_dual_domain_v2/dec_n1/dev_predictions.csv")
    invariant_audit = load(ROOT / "reports/invariant_dual_domain_v2/dec_n1_audit/predictions.csv")
    attention = load(args.attention_predictions)
    music = load(args.music_predictions)

    banks = {}
    for name, dataset in DATASETS.items():
        if name == "dev":
            truth, anchor = reconstruct_dev_v18(
                temporal_dev, fakeprint, invariant_dev
            )
        else:
            truth, anchor = reconstruct_v18(
                AUDITS[name], temporal_audit, fakeprint, invariant_audit
            )
        attention_score = select(attention, dataset, anchor.index)
        music_score = select(music, dataset, anchor.index)
        phone = np.zeros(len(anchor), dtype=bool)
        if name in ROUTERS:
            phone = (
                load(ROUTERS[name]).set_index("ID")
                .loc[anchor.index, "IS_PHONE"].to_numpy(bool)
            )
        banks[name] = truth, anchor, attention_score, music_score, phone

    baseline = {
        name: score_frame(truth.join(anchor))["ADS"]
        for name, (truth, anchor, *_rest) in banks.items()
    }
    rows = []
    combinations = (
        [(weight, weight) for weight in args.music_weights]
        if args.phone_music_weights is None
        else [
            (weight, phone_weight)
            for weight in args.music_weights
            for phone_weight in args.phone_music_weights
        ]
    )
    for music_weight, phone_music_weight in combinations:
        record = {
            "MUSIC_WEIGHT": music_weight,
            "PHONE_MUSIC_WEIGHT": phone_music_weight,
        }
        for name, (truth, anchor, attention_score, music_score, phone) in banks.items():
            candidate = anchor.copy()
            file_weight = np.where(
                phone, args.phone_file_weight, args.file_weight
            )
            candidate["FILE_FAKE_PROB"] = fuse(
                anchor.FILE_FAKE_PROB,
                attention_score.FILE_FAKE_PROB,
                file_weight,
            )
            candidate["MUSIC_FAKE_PROB"] = fuse(
                anchor.MUSIC_FAKE_PROB,
                music_score.TEMPORAL_BIN_MUSIC_PROB,
                np.where(phone, phone_music_weight, music_weight),
            )
            metrics = score_frame(truth.join(candidate))
            record[f"{name}_ADS"] = metrics["ADS"]
            record[f"{name}_DELTA"] = metrics["ADS"] - baseline[name]
            record[f"{name}_MUSIC_EER"] = metrics["MUSIC_EER"]
        delta_columns = [f"{name}_DELTA" for name in banks]
        record["MIN_DELTA"] = min(record[column] for column in delta_columns)
        record["MEAN_DELTA"] = np.mean([record[column] for column in delta_columns])
        rows.append(record)

    result = pd.DataFrame(rows).sort_values(
        ["MIN_DELTA", "MEAN_DELTA"], ascending=False
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)
    print("exact_v18_ads", {key: round(value, 8) for key, value in baseline.items()})
    print(result[[
        "MUSIC_WEIGHT", "PHONE_MUSIC_WEIGHT", "dev_DELTA",
        "factorial_DELTA", "phone_DELTA", "yue_DELTA", "MIN_DELTA",
        "MEAN_DELTA",
    ]].to_string(index=False))


if __name__ == "__main__":
    main()
