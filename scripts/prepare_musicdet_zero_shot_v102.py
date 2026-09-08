#!/usr/bin/env python3
"""Prepare real-only MusicDET train and all-class development CSV files."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from data_guard import assert_no_locked_eval_leakage  # noqa: E402


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def select(frame, *, only_real):
    present = pd.to_numeric(frame.MUSIC_PRESENT, errors="coerce").eq(1)
    label = pd.to_numeric(frame.MUSIC_FAKE, errors="coerce")
    keep = present & label.isin([0, 1])
    if only_real:
        keep &= label.eq(0)
    result = frame.loc[keep, ["ID", "PATH", "DATASET"]].copy()
    result["target"] = label.loc[keep].astype(int).to_numpy()
    if result.PATH.isna().any():
        raise ValueError("MusicDET rows require materialized PATH values")
    result["filepath"] = result.PATH.astype(str)
    missing = result.loc[~result.filepath.map(lambda value: Path(value).is_file())]
    if len(missing):
        raise FileNotFoundError(missing.iloc[0].filepath)
    if result.ID.duplicated().any():
        raise ValueError("duplicate IDs")
    return result[["filepath", "target", "ID", "DATASET"]].reset_index(drop=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-manifest", type=Path, required=True)
    parser.add_argument("--development-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if "locked" in " ".join(map(str, vars(args).values())).lower():
        raise ValueError("locked data are forbidden")
    assert_no_locked_eval_leakage(
        args.train_manifest, ROOT / "configs/data_partitions.yaml"
    )
    train_frame = pd.read_csv(args.train_manifest, low_memory=False)
    train = select(train_frame, only_real=True)
    train_all = select(train_frame, only_real=False)
    development = select(
        pd.read_csv(args.development_manifest, low_memory=False), only_real=False
    )
    if set(train.ID) & set(development.ID):
        raise ValueError("train/development ID overlap")
    args.output.mkdir(parents=True)
    train.to_csv(args.output / "train_real.csv", index=False)
    train_all.to_csv(args.output / "train_all.csv", index=False)
    development.to_csv(args.output / "development.csv", index=False)
    report = {
        "schema": "musicdet_zero_shot_v102",
        "train_rows": len(train), "development_rows": len(development),
        "train_all_rows": len(train_all),
        "train_target_counts": train.target.value_counts().sort_index().to_dict(),
        "train_all_target_counts": (
            train_all.target.value_counts().sort_index().to_dict()
        ),
        "development_target_counts": (
            development.target.value_counts().sort_index().to_dict()
        ),
        "train_dataset_counts": train.DATASET.value_counts().to_dict(),
        "train_manifest_sha256": sha256_file(args.train_manifest),
        "development_manifest_sha256": sha256_file(args.development_manifest),
        "locked_scores_read": False,
    }
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
