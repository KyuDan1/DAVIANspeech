"""Register a source-disjoint audio tree as a router-only evaluation set.

The command creates relative symlinks instead of copying audio.  Its output is
compatible with ``extract_telephone_router_features.py`` and deliberately
contains no fake-audio labels because the task is channel routing only.
"""

from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path

import pandas as pd


AUDIO_EXTENSIONS = {".aac", ".flac", ".m4a", ".mp3", ".ogg", ".opus", ".wav"}


def stable_order(path: Path, seed: int) -> str:
    return hashlib.sha256(f"{seed}|{path.as_posix()}".encode()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-name", required=True)
    parser.add_argument("--maximum", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260831)
    args = parser.parse_args()

    source = args.source.resolve()
    paths = sorted(
        (path for path in source.rglob("*")
         if path.is_file() and path.suffix.lower() in AUDIO_EXTENSIONS),
        key=lambda path: stable_order(path.relative_to(source), args.seed),
    )
    if args.maximum:
        paths = paths[:args.maximum]
    if not paths:
        raise ValueError(f"No supported audio found below {source}")

    audio_dir = args.output / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for index, path in enumerate(paths):
        relative = path.relative_to(source)
        digest = hashlib.sha256(relative.as_posix().encode()).hexdigest()[:12]
        item_id = f"external_{index:04d}_{digest}"
        destination = audio_dir / f"{item_id}{path.suffix.lower()}"
        if not destination.exists():
            destination.symlink_to(os.path.relpath(path, destination.parent))
        records.append({
            "ID": item_id,
            "AUDIO_TYPE": "voice",
            "SOURCE": args.source_name,
            "SOURCE_FILE": relative.as_posix(),
            "CHANNEL": "clean",
        })

    truth_path = args.output / "truth.csv"
    pd.DataFrame(records).to_csv(truth_path, index=False)
    print(f"registered {len(records)} source-disjoint files in {truth_path}")


if __name__ == "__main__":
    main()
