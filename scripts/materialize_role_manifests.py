#!/usr/bin/env python3
"""Materialize deterministic role manifests from mixed-role truth CSVs."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPLITS = ("dev", "holdout", "locked")
SOURCES = {
    "multigen_voice_v2": {"dev": 90, "holdout": 90, "locked": 108},
    "echoes_fma_paired_v3": {"dev": 111, "holdout": 141, "locked": 165},
}


def render_split(columns: list[str], rows: list[dict[str, str]]) -> bytes:
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=columns, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue().encode("utf-8")


def materialize(bank: Path, expected: dict[str, int], check: bool) -> None:
    source = bank / "truth.csv"
    with source.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        columns = list(reader.fieldnames or [])
        rows = list(reader)
    if "ID" not in columns or "SPLIT" not in columns:
        raise ValueError(f"{source}: ID and SPLIT columns are required")
    ids = [row["ID"] for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError(f"{source}: duplicate IDs")
    unknown = sorted({row["SPLIT"] for row in rows} - set(SPLITS))
    if unknown:
        raise ValueError(f"{source}: unsupported SPLIT values: {unknown}")
    for split in SPLITS:
        selected = [row for row in rows if row["SPLIT"] == split]
        if len(selected) != expected[split]:
            raise ValueError(
                f"{source}: expected {expected[split]} {split} rows, "
                f"found {len(selected)}"
            )
        payload = render_split(columns, selected)
        target = bank / f"truth_{split}.csv"
        if check:
            if not target.is_file() or target.read_bytes() != payload:
                raise ValueError(f"stale or missing generated manifest: {target}")
        else:
            temporary = target.with_suffix(".tmp")
            temporary.write_bytes(payload)
            temporary.replace(target)
        digest = hashlib.sha256(payload).hexdigest()
        print(f"{target.relative_to(ROOT)} rows={len(selected)} sha256={digest}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check", action="store_true",
        help="verify generated files byte-for-byte instead of replacing them",
    )
    args = parser.parse_args()
    for name, expected in SOURCES.items():
        materialize(ROOT / "data/eval" / name, expected, args.check)


if __name__ == "__main__":
    main()
