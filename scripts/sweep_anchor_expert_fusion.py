#!/usr/bin/env python3
"""Sweep low logit-space weights between a proven anchor and one expert."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from evaluate_diagnostic import score_frame  # noqa: E402


PROBABILITIES = ("FILE_FAKE_PROB", "VOICE_FAKE_PROB", "MUSIC_FAKE_PROB")


def logit(values: np.ndarray) -> np.ndarray:
    values = np.clip(values.astype(float), 1e-5, 1 - 1e-5)
    return np.log(values) - np.log1p(-values)


def sigmoid(values: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-values))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--truth", type=Path, required=True)
    parser.add_argument("--anchor", type=Path, required=True)
    parser.add_argument("--expert", type=Path, required=True)
    parser.add_argument("--dataset")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--weights", nargs="+", type=float, default=[0, .025, .05, .075, .1, .15, .2])
    parser.add_argument("--file-scale", type=float, default=1.0)
    parser.add_argument("--voice-scale", type=float, default=1.0)
    parser.add_argument("--music-scale", type=float, default=1.0)
    parser.add_argument("--prediction-output", type=Path)
    parser.add_argument("--prediction-weight", type=float, default=0.05)
    parser.add_argument("--route-column")
    parser.add_argument("--route-values", nargs="+")
    args = parser.parse_args()

    truth = pd.read_csv(args.truth, dtype={"ID": str})
    anchor = pd.read_csv(args.anchor, dtype={"ID": str})
    expert = pd.read_csv(args.expert, dtype={"ID": str})
    if args.dataset and "DATASET" in expert:
        expert = expert[expert.DATASET == args.dataset]
    expert = expert[["ID", *PROBABILITIES]].rename(
        columns={column: f"EXPERT_{column}" for column in PROBABILITIES}
    )
    frame = truth.merge(anchor, on="ID", validate="one_to_one").merge(
        expert, on="ID", validate="one_to_one"
    )
    scales = {
        "FILE_FAKE_PROB": args.file_scale,
        "VOICE_FAKE_PROB": args.voice_scale,
        "MUSIC_FAKE_PROB": args.music_scale,
    }
    if bool(args.route_column) != bool(args.route_values):
        parser.error("--route-column and --route-values must be provided together")

    def routed_fusion(group: pd.DataFrame, column: str, weight: float) -> np.ndarray:
        anchor_values = group[column].to_numpy(dtype=float)
        if not weight:
            return anchor_values
        selected = np.ones(len(group), dtype=bool)
        if args.route_column:
            selected = group[args.route_column].astype(str).isin(args.route_values).to_numpy()
        result = anchor_values.copy()
        result[selected] = sigmoid(
            (1 - weight) * logit(anchor_values[selected])
            + weight * logit(group.loc[selected, f"EXPERT_{column}"].to_numpy())
        )
        return result
    groups = [("ALL", "ALL", frame)]
    for column in ("SOURCE_BANK", "MIX_MODE", "COMPONENT_CASE", "CHANNEL"):
        if column in frame:
            groups.extend((column, str(value), group) for value, group in frame.groupby(column))
    rows = []
    for group_name, value, group in groups:
        for base_weight in args.weights:
            candidate = group.copy()
            for column, scale in scales.items():
                weight = base_weight * scale
                candidate[column] = routed_fusion(group, column, weight)
            try:
                metrics = score_frame(candidate)
            except ValueError:
                continue
            rows.append({
                "GROUP": group_name, "VALUE": value,
                "BASE_WEIGHT": base_weight, **metrics,
            })
    result = pd.DataFrame(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)
    if args.prediction_output is not None:
        candidate = frame.copy()
        for column, scale in scales.items():
            weight = args.prediction_weight * scale
            candidate[column] = routed_fusion(frame, column, weight)
        args.prediction_output.parent.mkdir(parents=True, exist_ok=True)
        presence = [
            column for column in ("VOICE_PRESENT_PROB", "MUSIC_PRESENT_PROB")
            if column in candidate
        ]
        candidate[["ID", *PROBABILITIES, *presence]].to_csv(
            args.prediction_output, index=False
        )
    print(result[result.GROUP == "ALL"].to_string(index=False))


if __name__ == "__main__":
    main()
