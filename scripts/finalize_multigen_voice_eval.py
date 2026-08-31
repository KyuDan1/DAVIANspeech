#!/usr/bin/env python3
"""Combine real and generator manifests into a validated voice source bank."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import soundfile as sf


PREDICTION_COLUMNS = [
    "FILE_FAKE_PROB", "VOICE_FAKE_PROB", "MUSIC_FAKE_PROB",
    "VOICE_PRESENT_PROB", "MUSIC_PRESENT_PROB",
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pool-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--generator-manifest", type=Path, nargs="+", required=True)
    args = parser.parse_args()

    frames = [pd.read_csv(args.pool_dir / "real_truth.csv", dtype=str)]
    frames.extend(pd.read_csv(path, dtype=str) for path in args.generator_manifest)
    truth = pd.concat(frames, ignore_index=True)
    if truth.ID.duplicated().any():
        raise ValueError(f"Duplicate IDs: {truth.loc[truth.ID.duplicated(), 'ID'].tolist()[:5]}")

    source_splits = truth.groupby("SPEAKER").SPLIT.nunique()
    if (source_splits != 1).any():
        raise ValueError("A speaker crosses evaluation splits")
    for sample_id in truth.ID:
        path = args.pool_dir / "audio" / f"{sample_id}.flac"
        if not path.is_file():
            raise FileNotFoundError(path)
        info = sf.info(path)
        if info.samplerate != 16_000 or not 4.0 <= info.duration <= 60.0:
            raise ValueError(f"Invalid audio shape for {sample_id}: {info}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    audio_link = args.output_dir / "audio"
    if not audio_link.exists():
        audio_link.symlink_to((args.pool_dir / "audio").resolve(), target_is_directory=True)
    truth.to_csv(args.output_dir / "truth.csv", index=False)
    sample = pd.DataFrame({"ID": truth.ID})
    for column in PREDICTION_COLUMNS:
        sample[column] = 0.0
    sample.to_csv(args.output_dir / "sample_submission.csv", index=False)
    print(truth.groupby(["SPLIT", "GENERATOR"]).size().unstack(fill_value=0))
    print(f"Finalized {len(truth)} voice files at {args.output_dir}")


if __name__ == "__main__":
    main()
