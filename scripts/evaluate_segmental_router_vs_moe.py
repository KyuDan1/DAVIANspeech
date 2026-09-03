#!/usr/bin/env python3
"""Compare fixed EAT soft MoE with observable file-level expert routers."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from evaluate_diagnostic import official_eer  # noqa: E402


def logit(values) -> np.ndarray:
    values = np.clip(np.asarray(values, dtype=float), 1e-5, 1 - 1e-5)
    return np.log(values) - np.log1p(-values)


def sigmoid(values) -> np.ndarray:
    return np.exp(-np.logaddexp(0, -np.asarray(values, dtype=float)))


def load(path: Path, column: str, output: str) -> pd.DataFrame:
    frame = pd.read_csv(path, dtype={"ID": str})
    if frame.duplicated(["DATASET", "ID"]).any():
        raise ValueError(f"duplicate prediction IDs in {path}")
    return frame.rename(columns={column: output})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--hierarchical-dev", type=Path,
        default=ROOT / "reports/hierarchical_eat_music_v1/ensemble02_dev/predictions.csv",
    )
    parser.add_argument(
        "--segmental-dev", type=Path, nargs=2,
        default=[
            ROOT / "reports/segmental_eat_music_v2/content_seed00/dev_predictions.csv",
            ROOT / "reports/segmental_eat_music_v2/content_seed02/dev_predictions.csv",
        ],
    )
    parser.add_argument(
        "--hierarchical-phone", type=Path,
        default=ROOT / "reports/segmental_eat_music_v2/phone_extra_hierarchical/predictions.csv",
    )
    parser.add_argument(
        "--segmental-phone", type=Path,
        default=ROOT / "reports/segmental_eat_music_v2/phone_extra_content_ensemble/predictions.csv",
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=ROOT / "reports/segmental_eat_music_v2/router_vs_moe",
    )
    args = parser.parse_args()

    hierarchical = load(
        args.hierarchical_dev, "HIERARCHICAL_EAT_MUSIC_PROB", "H"
    )
    segment_members = [
        load(path, "HIERARCHICAL_EAT_MUSIC_PROB", f"S{index}")
        for index, path in enumerate(args.segmental_dev)
    ]
    frame = hierarchical
    for index, member in enumerate(segment_members):
        frame = frame.merge(
            member[["DATASET", "ID", f"S{index}"]],
            on=["DATASET", "ID"], validate="one_to_one",
        )
    segment_logits = np.stack([
        logit(frame[f"S{index}"]) for index in range(len(segment_members))
    ])
    frame["S"] = sigmoid(segment_logits.mean(axis=0))
    frame = frame[[
        "DATASET", "ID", "MUSIC_FAKE", "MUSIC_PRESENT", "H", "S"
    ]]

    phone_h = load(
        args.hierarchical_phone, "HIERARCHICAL_EAT_MUSIC_PROB", "H"
    )
    phone_s = load(args.segmental_phone, "SEGMENTAL_EAT_MUSIC_PROB", "S")
    phone = phone_h.merge(
        phone_s[["DATASET", "ID", "S"]],
        on=["DATASET", "ID"], validate="one_to_one",
    )
    frame = pd.concat([frame, phone[frame.columns]], ignore_index=True)
    frame["IS_PHONE"] = frame.DATASET.str.contains("phone|telephone", case=False)
    hierarchical_logit = logit(frame.H)
    segmental_logit = logit(frame.S)

    fixed_weight = np.full(len(frame), .25)
    confidence_weight = .30 * sigmoid(
        2.0 + .5 * (np.abs(segmental_logit) - np.abs(hierarchical_logit))
    )
    strategies = {
        "hierarchical_only": hierarchical_logit,
        "segmental_only": segmental_logit,
        "phone_hard_expert": np.where(
            frame.IS_PHONE, segmental_logit, hierarchical_logit
        ),
        "confidence_hard_expert": np.where(
            np.abs(segmental_logit) > np.abs(hierarchical_logit),
            segmental_logit, hierarchical_logit,
        ),
        "fixed_soft_moe_025": (
            (1 - fixed_weight) * hierarchical_logit
            + fixed_weight * segmental_logit
        ),
        "phone_soft_router_025_040": (
            (1 - np.where(frame.IS_PHONE, .40, .25)) * hierarchical_logit
            + np.where(frame.IS_PHONE, .40, .25) * segmental_logit
        ),
        "confidence_soft_router": (
            (1 - confidence_weight) * hierarchical_logit
            + confidence_weight * segmental_logit
        ),
    }
    rows = []
    for strategy, scores in strategies.items():
        probabilities = sigmoid(scores)
        for dataset, block in frame.groupby("DATASET", sort=False):
            selected = block.MUSIC_PRESENT.eq(1)
            rows.append({
                "DATASET": dataset,
                "STRATEGY": strategy,
                "N": int(selected.sum()),
                "MUSIC_EER": official_eer(
                    block.loc[selected, "MUSIC_FAKE"],
                    probabilities[block.index[selected]],
                ),
            })
    per_dataset = pd.DataFrame(rows)
    reference = (
        per_dataset.loc[per_dataset.STRATEGY.eq("fixed_soft_moe_025")]
        .set_index("DATASET").MUSIC_EER
    )
    per_dataset["EER_DELTA_VS_FIXED"] = [
        row.MUSIC_EER - reference[row.DATASET]
        for row in per_dataset.itertuples()
    ]
    summary = per_dataset.groupby("STRATEGY", sort=False).agg(
        MEAN_EER=("MUSIC_EER", "mean"),
        WORST_EER=("MUSIC_EER", "max"),
        MAX_REGRESSION_VS_FIXED=("EER_DELTA_VS_FIXED", "max"),
        MEAN_DELTA_VS_FIXED=("EER_DELTA_VS_FIXED", "mean"),
    ).reset_index()
    summary["SELECTION"] = 1 - .5 * (
        summary.MEAN_EER + summary.WORST_EER
    )
    summary = summary.sort_values(
        ["MAX_REGRESSION_VS_FIXED", "SELECTION"], ascending=[True, False]
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    per_dataset.to_csv(args.output_dir / "per_dataset.csv", index=False)
    summary.to_csv(args.output_dir / "summary.csv", index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
