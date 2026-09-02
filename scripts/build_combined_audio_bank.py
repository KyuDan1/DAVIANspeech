#!/usr/bin/env python3
"""Combine disjoint local banks through symlinks and a merged truth manifest."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    frames = []
    files = {}
    for source in args.input:
        truth = pd.read_csv(source / "truth.csv", dtype={"ID": str})
        truth["SOURCE_BANK"] = source.name
        frames.append(truth)
        for path in (source / "audio").iterdir():
            if path.is_file() or path.is_symlink():
                if path.stem in files:
                    raise ValueError(f"Duplicate audio ID {path.stem}")
                files[path.stem] = path.resolve()
    merged = pd.concat(frames, ignore_index=True)
    if merged.ID.duplicated().any():
        raise ValueError("Duplicate IDs in truth manifests")
    missing = set(merged.ID) - set(files)
    extra = set(files) - set(merged.ID)
    if missing or extra:
        raise ValueError(f"Audio/truth mismatch: missing={len(missing)}, extra={len(extra)}")
    audio_dir = args.output / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    for item in merged.ID:
        source = files[item]
        destination = audio_dir / source.name
        if destination.exists() or destination.is_symlink():
            if destination.resolve() != source:
                raise ValueError(f"Conflicting existing link: {destination}")
        else:
            destination.symlink_to(source)
    args.output.mkdir(parents=True, exist_ok=True)
    merged.to_csv(args.output / "truth.csv", index=False)
    sample = pd.DataFrame({"ID": merged.ID})
    for column in (
        "FILE_FAKE_PROB", "VOICE_FAKE_PROB", "MUSIC_FAKE_PROB",
        "VOICE_PRESENT_PROB", "MUSIC_PRESENT_PROB",
    ):
        sample[column] = 0.5
    sample.to_csv(args.output / "sample_submission.csv", index=False)
    print(f"Built {args.output} with {len(merged)} samples")


if __name__ == "__main__":
    main()
