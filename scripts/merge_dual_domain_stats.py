#!/usr/bin/env python3
"""Merge sharded EAT/SPEAR statistics into the offline submission format."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_dir", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    paths = sorted(args.input_dir.glob("shard_*.npz"))
    if not paths:
        raise FileNotFoundError(f"No shards in {args.input_dir}")
    shards = [np.load(path, allow_pickle=False) for path in paths]
    ids = np.concatenate([shard["ids"].astype(str) for shard in shards])
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate IDs across statistic shards")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.output,
        ids=ids,
        statistics=np.concatenate([shard["statistics"] for shard in shards]),
        view_mask=np.concatenate([shard["view_mask"] for shard in shards]),
        stream=np.asarray(str(shards[0]["stream"])),
        channel=np.asarray(str(shards[0]["channel"])),
    )
    print(f"Merged {len(ids)} examples into {args.output}")


if __name__ == "__main__":
    main()
