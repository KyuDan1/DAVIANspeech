#!/usr/bin/env python3
"""Materialize a truth-defined audio subset as symlinks without copying audio."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from data_guard import IDENTITY_COLUMNS, identity_tokens  # noqa: E402


def remove_protected_rows(truth: pd.DataFrame, config_path: Path) -> pd.DataFrame:
    """Exclude rows sharing any tracked identity with a protected data role."""
    config = yaml.safe_load(config_path.read_text("utf-8"))
    config_dir = config_path.resolve().parent
    root = config_dir.parent if config_dir.name == "configs" else config_dir
    protected: set[str] = set()
    for role in ("development", "ood_holdout", "stress_eval", "locked_eval"):
        for relative in config.get(role, []):
            path = root / relative
            if not path.is_file():
                raise FileNotFoundError(f"Protected truth is missing: {path}")
            protected.update(identity_tokens(pd.read_csv(path, dtype=str)))
    overlap = pd.Series(False, index=truth.index)
    for column in IDENTITY_COLUMNS:
        if column in truth:
            overlap |= truth[column].astype(str).isin(protected)
    print(f"Excluded {int(overlap.sum())} protected identity rows")
    return truth.loc[~overlap].copy()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-audio", type=Path, required=True)
    parser.add_argument("--truth", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--exclude-protected-config", type=Path,
        help="Drop rows overlapping any protected role in this partition config.",
    )
    args = parser.parse_args()
    truth = pd.read_csv(args.truth, dtype={"ID": str})
    if args.exclude_protected_config is not None:
        truth = remove_protected_rows(truth, args.exclude_protected_config)
    available = {
        path.stem: path.resolve() for path in args.source_audio.iterdir()
        if path.is_file() or path.is_symlink()
    }
    missing = set(truth.ID) - set(available)
    if missing:
        raise FileNotFoundError(f"Missing {len(missing)} audio files: {sorted(missing)[:5]}")
    audio_dir = args.output / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    for item in truth.ID:
        source = available[item]
        destination = audio_dir / source.name
        if not destination.exists():
            destination.symlink_to(source)
    truth.to_csv(args.output / "truth.csv", index=False)
    sample = pd.DataFrame({"ID": truth.ID})
    for column in (
        "FILE_FAKE_PROB", "VOICE_FAKE_PROB", "MUSIC_FAKE_PROB",
        "VOICE_PRESENT_PROB", "MUSIC_PRESENT_PROB",
    ):
        sample[column] = 0.5
    sample.to_csv(args.output / "sample_submission.csv", index=False)
    print(f"Built {args.output} with {len(truth)} samples")


if __name__ == "__main__":
    main()
