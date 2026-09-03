#!/usr/bin/env python3
"""Sweep an invariant head after the conservative v17 music experts."""

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
    clipped = np.clip(np.asarray(values, dtype=np.float64), 1e-5, 1 - 1e-5)
    return np.log(clipped) - np.log1p(-clipped)


def sigmoid(values: np.ndarray) -> np.ndarray:
    return np.exp(-np.logaddexp(0.0, -np.asarray(values, dtype=np.float64)))


def blend(anchor: np.ndarray, expert: np.ndarray, weight: float) -> np.ndarray:
    return sigmoid((1 - weight) * logit(anchor) + weight * logit(expert))


def select_dataset(frame: pd.DataFrame, dataset: str | None) -> pd.DataFrame:
    if dataset and "DATASET" in frame:
        return frame[frame.DATASET == dataset].copy()
    return frame.copy()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--truth", type=Path, required=True)
    parser.add_argument("--anchor", type=Path, required=True)
    parser.add_argument("--mert", type=Path, required=True)
    parser.add_argument("--fakeprint", type=Path, required=True)
    parser.add_argument("--invariant", type=Path, required=True)
    parser.add_argument("--fakeprint-dataset")
    parser.add_argument("--invariant-dataset")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prediction-output", type=Path)
    parser.add_argument(
        "--weights", nargs="+", type=float,
        default=[0, 0.01, 0.025, 0.05, 0.075, 0.1],
    )
    parser.add_argument("--prediction-weight", type=float, default=0.05)
    parser.add_argument("--mert-file-weight", type=float, default=0.025)
    parser.add_argument("--mert-music-weight", type=float, default=0.0125)
    parser.add_argument("--fakeprint-file-weight", type=float, default=0.025)
    parser.add_argument("--fakeprint-music-weight", type=float, default=0.025)
    parser.add_argument("--invariant-file-scale", type=float, default=1.0)
    parser.add_argument("--invariant-voice-scale", type=float, default=1.0)
    parser.add_argument("--invariant-music-scale", type=float, default=1.0)
    args = parser.parse_args()

    truth = pd.read_csv(args.truth, dtype={"ID": str})
    anchor = pd.read_csv(args.anchor, dtype={"ID": str})
    mert = pd.read_csv(args.mert, dtype={"ID": str})[
        ["ID", "SOFIA_MERT_FAKE_PROB"]
    ]
    fakeprint = select_dataset(
        pd.read_csv(args.fakeprint, dtype={"ID": str}), args.fakeprint_dataset
    )[["ID", "MODERN_FAKEPRINT_PROB"]]
    invariant = select_dataset(
        pd.read_csv(args.invariant, dtype={"ID": str}), args.invariant_dataset
    )[["ID", *PROBABILITIES]].rename(
        columns={column: f"INVARIANT_{column}" for column in PROBABILITIES}
    )
    frame = truth.merge(anchor, on="ID", validate="one_to_one")
    frame = frame.merge(mert, on="ID", validate="one_to_one")
    frame = frame.merge(fakeprint, on="ID", validate="one_to_one")
    frame = frame.merge(invariant, on="ID", validate="one_to_one")

    frame["FILE_FAKE_PROB"] = blend(
        frame.FILE_FAKE_PROB, frame.SOFIA_MERT_FAKE_PROB,
        args.mert_file_weight,
    )
    frame["MUSIC_FAKE_PROB"] = blend(
        frame.MUSIC_FAKE_PROB, frame.SOFIA_MERT_FAKE_PROB,
        args.mert_music_weight,
    )
    frame["FILE_FAKE_PROB"] = blend(
        frame.FILE_FAKE_PROB, frame.MODERN_FAKEPRINT_PROB,
        args.fakeprint_file_weight,
    )
    frame["MUSIC_FAKE_PROB"] = blend(
        frame.MUSIC_FAKE_PROB, frame.MODERN_FAKEPRINT_PROB,
        args.fakeprint_music_weight,
    )

    scales = {
        "FILE_FAKE_PROB": args.invariant_file_scale,
        "VOICE_FAKE_PROB": args.invariant_voice_scale,
        "MUSIC_FAKE_PROB": args.invariant_music_scale,
    }
    groups = [("ALL", "ALL", frame)]
    for column in ("CHANNEL", "MIX_MODE", "AUDIO_TYPE", "COMPONENT_CASE"):
        if column in frame:
            groups.extend(
                (column, str(value), group) for value, group in frame.groupby(column)
            )
    records = []
    for group_name, value, group in groups:
        for weight in args.weights:
            candidate = group.copy()
            for column, scale in scales.items():
                candidate[column] = blend(
                    group[column], group[f"INVARIANT_{column}"], weight * scale
                )
            try:
                metrics = score_frame(candidate)
            except ValueError:
                continue
            records.append({
                "GROUP": group_name, "VALUE": value,
                "INVARIANT_WEIGHT": weight, **metrics,
            })
    result = pd.DataFrame(records)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)
    print(result[result.GROUP == "ALL"].to_string(index=False))

    if args.prediction_output is not None:
        candidate = frame.copy()
        for column, scale in scales.items():
            candidate[column] = blend(
                frame[column], frame[f"INVARIANT_{column}"],
                args.prediction_weight * scale,
            )
        presence = [
            column for column in ("VOICE_PRESENT_PROB", "MUSIC_PRESENT_PROB")
            if column in candidate
        ]
        args.prediction_output.parent.mkdir(parents=True, exist_ok=True)
        candidate[["ID", *PROBABILITIES, *presence]].to_csv(
            args.prediction_output, index=False
        )


if __name__ == "__main__":
    main()
