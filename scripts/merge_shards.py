"""Merge per-shard submissions back into sample_submission.csv's row order."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shard-dir", type=Path, required=True)
    parser.add_argument("--sample-submission", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    shards = sorted(args.shard_dir.glob("shard_*.csv"))
    if not shards:
        raise SystemExit(f"No shard_*.csv under {args.shard_dir}")

    merged = pd.concat([pd.read_csv(p) for p in shards], ignore_index=True)
    duplicates = merged["ID"].duplicated().sum()
    if duplicates:
        raise SystemExit(f"{duplicates} IDs appear in more than one shard")

    template = pd.read_csv(args.sample_submission, usecols=["ID"])
    result = template.merge(merged, on="ID", how="left")

    missing = result[result.isna().any(axis=1)]["ID"].tolist()
    if missing:
        raise SystemExit(
            f"{len(missing)} IDs have no prediction, e.g. {missing[:5]}"
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    # pipeline.py writes through csv.DictWriter, whose default dialect uses
    # CRLF. Match it so a sharded run is byte-identical to a single process.
    result.to_csv(args.output, index=False, lineterminator="\r\n")
    print(f"Merged {len(shards)} shards -> {args.output} ({len(result)} rows)")


if __name__ == "__main__":
    main()
