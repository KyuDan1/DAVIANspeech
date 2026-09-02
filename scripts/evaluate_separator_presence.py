#!/usr/bin/env python3
"""Audit whether separator statistics improve fixed presence rankings."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


def logit(values) -> np.ndarray:
    values = np.clip(np.asarray(values, dtype=np.float64), 1e-5, 1 - 1e-5)
    return np.log(values) - np.log1p(-values)


def sigmoid(values) -> np.ndarray:
    return np.exp(-np.logaddexp(0.0, -np.asarray(values, dtype=np.float64)))


def bounded_ratio(values, scale: float = 0.05) -> np.ndarray:
    """Map a non-negative per-file ratio to [0, 1] without cross-file state."""
    values = np.maximum(np.asarray(values, dtype=np.float64), 0.0)
    return values / (values + scale)


def auc(labels, scores) -> float:
    return float(roc_auc_score(np.asarray(labels), np.asarray(scores)))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--truth", type=Path, required=True)
    parser.add_argument("--anchor", type=Path, required=True)
    parser.add_argument("--stats", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--weights", type=float, nargs="+",
                        default=[0, 0.01, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2])
    args = parser.parse_args()

    truth = pd.read_csv(args.truth, dtype={"ID": str})
    anchor = pd.read_csv(args.anchor, dtype={"ID": str})
    stats = pd.concat(pd.read_csv(path, dtype={"ID": str}) for path in args.stats)
    if stats.ID.duplicated().any():
        raise ValueError("Duplicate IDs in separator statistics")
    frame = truth.merge(anchor, on="ID", validate="one_to_one").merge(
        stats, on="ID", validate="one_to_one"
    )

    # The extrema preserve short sequential components. Energy share provides
    # an independent tie-breaker when both stems are active throughout.
    voice_experts = {
        "voice_energy": bounded_ratio(frame.VOICE_ENERGY_RATIO),
        "voice_q90": bounded_ratio(frame.VOICE_FRAME_RATIO_Q90),
        "voice_max": bounded_ratio(frame.VOICE_FRAME_RATIO_Q100),
        "voice_share": frame.VOICE_STEM_SHARE.to_numpy(),
        "voice_composite": (
            bounded_ratio(frame.VOICE_FRAME_RATIO_Q90)
            + bounded_ratio(frame.VOICE_ENERGY_RATIO)
            + frame.VOICE_STEM_SHARE.to_numpy()
        ) / 3,
    }
    music_experts = {
        "music_energy": bounded_ratio(frame.MUSIC_ENERGY_RATIO),
        "music_q90": bounded_ratio(frame.MUSIC_FRAME_RATIO_Q90),
        "music_max": bounded_ratio(frame.MUSIC_FRAME_RATIO_Q100),
        "music_share": frame.MUSIC_STEM_SHARE.to_numpy(),
        "music_composite": (
            bounded_ratio(frame.MUSIC_FRAME_RATIO_Q90)
            + bounded_ratio(frame.MUSIC_ENERGY_RATIO)
            + frame.MUSIC_STEM_SHARE.to_numpy()
        ) / 3,
    }
    groups = [("ALL", "ALL", frame.index)]
    for column in ("CHANNEL", "MIX_MODE", "AUDIO_TYPE"):
        if column in frame:
            groups.extend(
                (column, str(value), group.index)
                for value, group in frame.groupby(column)
            )

    records = []
    for group_name, value, index in groups:
        subset = frame.loc[index]
        for component, label, base_column, experts in (
            ("voice", "VOICE_PRESENT", "VOICE_PRESENT_PROB", voice_experts),
            ("music", "MUSIC_PRESENT", "MUSIC_PRESENT_PROB", music_experts),
        ):
            if subset[label].nunique() != 2:
                continue
            base = frame.loc[index, base_column].to_numpy()
            for expert_name, expert_values in experts.items():
                expert = np.asarray(expert_values)[index]
                for weight in args.weights:
                    candidate = sigmoid(
                        (1 - weight) * logit(base) + weight * logit(expert)
                    )
                    records.append({
                        "GROUP": group_name, "VALUE": value,
                        "COMPONENT": component, "EXPERT": expert_name,
                        "WEIGHT": weight, "N": len(index),
                        "AUC": auc(subset[label], candidate),
                    })
    result = pd.DataFrame(records)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)
    print(result[(result.GROUP == "ALL")].sort_values(
        ["COMPONENT", "AUC"], ascending=[True, False]
    ).groupby("COMPONENT").head(12).to_string(index=False))


if __name__ == "__main__":
    main()
