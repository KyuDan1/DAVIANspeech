#!/usr/bin/env python3
"""Recompose File logits from saved anchor-relative component residuals."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def logit(values: np.ndarray) -> np.ndarray:
    values = np.clip(np.asarray(values, np.float64), 1e-6, 1 - 1e-6)
    return np.log(values) - np.log1p(-values)


def sigmoid(values: np.ndarray) -> np.ndarray:
    return np.exp(-np.logaddexp(0.0, -np.asarray(values, np.float64)))


def noisy_or_logit(voice: np.ndarray, music: np.ndarray) -> np.ndarray:
    return np.logaddexp(np.logaddexp(voice, music), voice + music)


def recompose(
    frame: pd.DataFrame, component_weight: float, trained_weight: float = 0.30,
) -> pd.DataFrame:
    if not 0 <= component_weight < 1 or not 0 <= trained_weight < 1:
        raise ValueError("component weights must be in [0,1)")
    required = {
        "FILE_FAKE_PROB", "VOICE_FAKE_PROB", "MUSIC_FAKE_PROB",
        "VOICE_LOGIT_RESIDUAL", "MUSIC_LOGIT_RESIDUAL",
        "DIRECT_FILE_LOGIT_RESIDUAL",
    }
    if missing := required.difference(frame):
        raise ValueError(f"prediction misses columns: {sorted(missing)}")
    output = frame.copy()
    voice = logit(frame.VOICE_FAKE_PROB)
    music = logit(frame.MUSIC_FAKE_PROB)
    voice_residual = frame.VOICE_LOGIT_RESIDUAL.to_numpy(np.float64)
    music_residual = frame.MUSIC_LOGIT_RESIDUAL.to_numpy(np.float64)
    file_residual = frame.DIRECT_FILE_LOGIT_RESIDUAL.to_numpy(np.float64)
    anchor_voice = voice - voice_residual
    anchor_music = music - music_residual
    delta_or = noisy_or_logit(voice, music) - noisy_or_logit(
        anchor_voice, anchor_music
    )
    current_file = logit(frame.FILE_FAKE_PROB)
    anchor_file = current_file - (1 - trained_weight) * file_residual - trained_weight * delta_or
    recomposed_file = (
        anchor_file + (1 - component_weight) * file_residual
        + component_weight * delta_or
    )
    output["FILE_FAKE_PROB"] = sigmoid(recomposed_file)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prediction", type=Path, required=True)
    parser.add_argument("--component-weight", type=float, required=True)
    parser.add_argument("--trained-weight", type=float, default=0.30)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite {args.output_dir}")
    frame = pd.read_csv(args.prediction, dtype={"ID": str})
    result = recompose(frame, args.component_weight, args.trained_weight)
    args.output_dir.mkdir(parents=True)
    result.to_csv(args.output_dir / "predictions.csv", index=False)


if __name__ == "__main__":
    main()
