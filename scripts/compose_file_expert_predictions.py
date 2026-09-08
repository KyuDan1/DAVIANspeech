#!/usr/bin/env python3
"""Blend one fixed File expert while preserving all other output columns."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def logit(values: np.ndarray) -> np.ndarray:
    values = np.clip(np.asarray(values, np.float64), 1e-6, 1 - 1e-6)
    return np.log(values) - np.log1p(-values)


def sigmoid(values: np.ndarray) -> np.ndarray:
    return np.exp(-np.logaddexp(0.0, -np.asarray(values, np.float64)))


def compose(base: pd.DataFrame, expert: pd.DataFrame, weight: float) -> pd.DataFrame:
    if not 0 <= weight <= 1:
        raise ValueError("weight must be in [0,1]")
    key = ["DATASET", "ID"]
    base = base.sort_values(key).reset_index(drop=True)
    expert = expert.sort_values(key).reset_index(drop=True)
    if not base[key].equals(expert[key]):
        raise ValueError("base and File-expert IDs differ")
    required_unchanged = (
        "VOICE_FAKE_PROB", "MUSIC_FAKE_PROB",
        "VOICE_PRESENT_PROB", "MUSIC_PRESENT_PROB",
    )
    for column in ("FILE_FAKE_PROB", *required_unchanged):
        if column not in base or column not in expert:
            raise ValueError(f"missing probability column: {column}")
    for column in ("VOICE_PRESENT_PROB", "MUSIC_PRESENT_PROB"):
        if not np.array_equal(base[column], expert[column]):
            raise ValueError(f"presence changed in File expert: {column}")
    output = base.copy()
    output["FILE_FAKE_PROB"] = sigmoid(
        (1 - weight) * logit(base.FILE_FAKE_PROB)
        + weight * logit(expert.FILE_FAKE_PROB)
    )
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--file-expert", type=Path, required=True)
    parser.add_argument("--file-weight", type=float, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite {args.output_dir}")
    base = pd.read_csv(args.base, dtype={"ID": str})
    expert = pd.read_csv(args.file_expert, dtype={"ID": str})
    output = compose(base, expert, args.file_weight)
    args.output_dir.mkdir(parents=True)
    prediction_path = args.output_dir / "predictions.csv"
    output.to_csv(prediction_path, index=False)
    provenance = {
        "method": "fixed_file_expert_logit_blend",
        "file_expert_weight": args.file_weight,
        "base": {"path": str(args.base.resolve()), "sha256": sha256_file(args.base)},
        "file_expert": {
            "path": str(args.file_expert.resolve()),
            "sha256": sha256_file(args.file_expert),
        },
        "prediction_sha256": sha256_file(prediction_path),
    }
    (args.output_dir / "provenance.json").write_text(
        json.dumps(provenance, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
