#!/usr/bin/env python3
"""Scale a saved three-stream residual while exactly reconstructing its anchor."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


RESIDUAL_COLUMNS = (
    "VOICE_LOGIT_RESIDUAL", "MUSIC_LOGIT_RESIDUAL",
    "DIRECT_FILE_LOGIT_RESIDUAL",
)


def logit(values: np.ndarray) -> np.ndarray:
    values = np.clip(np.asarray(values, np.float64), 1e-6, 1 - 1e-6)
    return np.log(values) - np.log1p(-values)


def sigmoid(values: np.ndarray) -> np.ndarray:
    return np.exp(-np.logaddexp(0.0, -np.asarray(values, np.float64)))


def noisy_or_logit(voice: np.ndarray, music: np.ndarray) -> np.ndarray:
    return np.logaddexp(np.logaddexp(voice, music), voice + music)


def scale_residual(
    frame: pd.DataFrame, scale: float, component_weight: float = 0.30,
) -> pd.DataFrame:
    if not 0 <= scale <= 1:
        raise ValueError("residual scale must lie in [0,1]")
    if not 0 <= component_weight < 1:
        raise ValueError("component weight must lie in [0,1)")
    required = {
        "FILE_FAKE_PROB", "VOICE_FAKE_PROB", "MUSIC_FAKE_PROB",
        *RESIDUAL_COLUMNS,
    }
    if missing := required.difference(frame):
        raise ValueError(f"prediction misses columns: {sorted(missing)}")
    output = frame.copy()
    voice = logit(frame.VOICE_FAKE_PROB)
    music = logit(frame.MUSIC_FAKE_PROB)
    file_logit = logit(frame.FILE_FAKE_PROB)
    voice_residual = frame.VOICE_LOGIT_RESIDUAL.to_numpy(np.float64)
    music_residual = frame.MUSIC_LOGIT_RESIDUAL.to_numpy(np.float64)
    file_residual = frame.DIRECT_FILE_LOGIT_RESIDUAL.to_numpy(np.float64)
    anchor_voice = voice - voice_residual
    anchor_music = music - music_residual
    anchor_or = noisy_or_logit(anchor_voice, anchor_music)
    current_or = noisy_or_logit(voice, music)
    direct_weight = 1 - component_weight
    anchor_file = (
        file_logit - direct_weight * file_residual
        - component_weight * (current_or - anchor_or)
    )
    scaled_voice = anchor_voice + scale * voice_residual
    scaled_music = anchor_music + scale * music_residual
    scaled_file = (
        anchor_file + direct_weight * scale * file_residual
        + component_weight * (
            noisy_or_logit(scaled_voice, scaled_music) - anchor_or
        )
    )
    output["VOICE_FAKE_PROB"] = sigmoid(scaled_voice)
    output["MUSIC_FAKE_PROB"] = sigmoid(scaled_music)
    output["FILE_FAKE_PROB"] = sigmoid(scaled_file)
    for column in RESIDUAL_COLUMNS:
        output[column] = scale * frame[column].to_numpy(np.float64)
    return output


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prediction", type=Path, required=True)
    parser.add_argument("--scale", type=float, required=True)
    parser.add_argument("--component-weight", type=float, default=0.30)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite {args.output_dir}")
    frame = pd.read_csv(args.prediction, dtype={"ID": str})
    output = scale_residual(frame, args.scale, args.component_weight)
    args.output_dir.mkdir(parents=True)
    output_path = args.output_dir / "predictions.csv"
    output.to_csv(output_path, index=False)
    (args.output_dir / "provenance.json").write_text(json.dumps({
        "method": "anchor_reconstructed_residual_scale",
        "scale": args.scale,
        "component_weight": args.component_weight,
        "input": str(args.prediction.resolve()),
        "input_sha256": sha256_file(args.prediction),
        "output_sha256": sha256_file(output_path),
    }, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
