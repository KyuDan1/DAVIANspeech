#!/usr/bin/env python3
"""Merge disjoint TTS generation shards and reject duplicate sample IDs."""

from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    frame = pd.concat([pd.read_csv(path, dtype=str) for path in args.inputs], ignore_index=True)
    if frame.ID.duplicated().any():
        raise ValueError("Duplicate IDs in generator shards")
    frame.sort_values("ID").to_csv(args.output, index=False)
    print(f"Merged {len(frame)} rows into {args.output}")


if __name__ == "__main__":
    main()
