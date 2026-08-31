"""Expose source components already authorized by the training partitions."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


AUDIO_EXTENSIONS = {".aac", ".flac", ".m4a", ".mp3", ".ogg", ".opus", ".wav"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dataset", type=Path, required=True)
    parser.add_argument("--allow-truth", type=Path, action="append", required=True)
    parser.add_argument("--allow-column", default="VOICE_SOURCE_ID")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    allowed = set()
    for path in args.allow_truth:
        frame = pd.read_csv(path, dtype=str)
        allowed.update(frame[args.allow_column].dropna())
    truth = pd.read_csv(args.source_dataset / "truth.csv", dtype={"ID": str})
    audio_by_id = {
        path.stem: path.resolve() for path in (args.source_dataset / "audio").iterdir()
        if path.is_file() and path.suffix.lower() in AUDIO_EXTENSIONS
    }
    truth = truth[truth.ID.isin(allowed & audio_by_id.keys())].copy()
    truth["ROUTER_TRAIN_SOURCE"] = args.allow_column
    audio_dir = args.output_dir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    for sample_id in truth.ID:
        source = audio_by_id[sample_id]
        destination = audio_dir / f"{sample_id}{source.suffix.lower()}"
        if not destination.exists():
            destination.symlink_to(source)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    truth.to_csv(args.output_dir / "truth.csv", index=False)
    print(f"exposed {len(truth)} authorized components in {args.output_dir}")


if __name__ == "__main__":
    main()
