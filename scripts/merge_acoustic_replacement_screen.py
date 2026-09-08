#!/usr/bin/env python3
"""Merge a frozen initial acoustic screen with its replacement-only recheck."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def parse_bool(series: pd.Series) -> pd.Series:
    values = series.astype(str).str.strip().str.lower()
    mapping = {
        "true": True, "1": True, "yes": True, "y": True,
        "false": False, "0": False, "no": False, "n": False,
    }
    if unknown := sorted(set(values).difference(mapping)):
        raise ValueError(f"unknown boolean values: {unknown[:5]}")
    return values.map(mapping).astype(bool)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--initial-dir", type=Path, required=True)
    parser.add_argument("--replacement-dir", type=Path, required=True)
    parser.add_argument("--final-reservation", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    outputs = [
        args.output_dir / "source_scores.csv",
        args.output_dir / "generator_summary.csv",
        args.output_dir / "summary.json",
    ]
    if existing := [path for path in outputs if path.exists()]:
        raise FileExistsError(f"refusing to overwrite: {existing}")

    initial_path = args.initial_dir / "source_scores.csv"
    initial_summary_path = args.initial_dir / "summary.json"
    replacement_path = args.replacement_dir / "source_scores.csv"
    replacement_summary_path = args.replacement_dir / "summary.json"
    initial = pd.read_csv(initial_path)
    replacement = pd.read_csv(replacement_path)
    if list(initial.columns) != list(replacement.columns):
        raise ValueError("initial/replacement score schemas differ")
    initial_pass = parse_bool(initial.ACOUSTIC_SCREEN_PASS)
    replacement_pass = parse_bool(replacement.ACOUSTIC_SCREEN_PASS)
    failed_ids = set(initial.loc[~initial_pass, "MUSIC_SOURCE_ID"])
    replacement_ids = set(replacement.MUSIC_SOURCE_ID)
    if len(failed_ids) != len(replacement_ids) or len(failed_ids) != 1:
        raise ValueError(
            "expected one initial failure and one deterministic replacement"
        )
    if not replacement_pass.all():
        raise ValueError("a replacement failed its frozen acoustic recheck")

    final_reservation = pd.read_csv(args.final_reservation, dtype=str).fillna("")
    final_ids = set(final_reservation.loc[
        final_reservation.MUSIC_FAKE.astype(int).eq(1), "MUSIC_SOURCE_ID"
    ])
    retained = initial.loc[initial_pass].copy()
    merged = pd.concat([retained, replacement], ignore_index=True)
    expected_sources = len(final_ids)
    if len(merged) != expected_sources or set(merged.MUSIC_SOURCE_ID) != final_ids:
        raise ValueError("merged acoustic evidence does not match final reservation")
    if failed_ids.intersection(final_ids):
        raise ValueError("an acoustically failed source survived the final reservation")
    if set(merged.PANNS_THRESHOLD.astype(float)) != {0.20}:
        raise ValueError("PANNs threshold drift")
    if set(merged.DEMUCS_THRESHOLD_DB.astype(float)) != {-1.5}:
        raise ValueError("Demucs threshold drift")
    if not parse_bool(merged.ACOUSTIC_SCREEN_PASS).all():
        raise ValueError("merged screen contains a failure")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    merged.sort_values("MUSIC_SOURCE_ID").to_csv(
        args.output_dir / "source_scores.csv", index=False
    )
    generator = merged.groupby("MUSIC_GENERATOR").agg(
        SOURCES=("MUSIC_SOURCE_ID", "size"),
        PANNS_FLAGS=("PANNS_FLAG", "sum"),
        DEMUCS_FLAGS=("DEMUCS_FLAG", "sum"),
        PASSES=("ACOUSTIC_SCREEN_PASS", "sum"),
        MAX_PANNS_VOICE=("PANNS_ANY_VOICE_MAX", "max"),
    ).reset_index()
    generator.to_csv(args.output_dir / "generator_summary.csv", index=False)
    initial_summary = json.loads(initial_summary_path.read_text("utf-8"))
    replacement_summary = json.loads(replacement_summary_path.read_text("utf-8"))
    summary = {
        "purpose": "final truth semantic evidence; not authenticity-model evaluation",
        "status": "completed_merged_initial_and_replacement",
        "authenticity_detector_scores_read": False,
        "reservation_path": str(args.final_reservation.resolve()),
        "reservation_sha256": sha256_file(args.final_reservation),
        "panns_threshold": 0.20,
        "demucs_threshold_db": -1.5,
        "candidate_sources_scored": expected_sources,
        "panns_flags": 0,
        "demucs_flags": 0,
        "acoustic_screen_passes": expected_sources,
        "acoustic_screen_failures": 0,
        "initial_screen": {
            "path": str(initial_path.resolve()),
            "sha256": sha256_file(initial_path),
            "reservation_sha256": initial_summary["reservation_sha256"],
            "rows": len(initial),
            "failures": int((~initial_pass).sum()),
            "excluded_source_ids": sorted(failed_ids),
        },
        "replacement_recheck": {
            "path": str(replacement_path.resolve()),
            "sha256": sha256_file(replacement_path),
            "reservation_sha256": replacement_summary["reservation_sha256"],
            "rows": len(replacement),
            "passes": int(replacement_pass.sum()),
            "replacement_source_ids": sorted(replacement_ids),
        },
        "selection_rule": (
            f"retain the {len(retained)} initial passes and substitute only the one "
            "deterministic same-generator replacement that passed the "
            "unchanged PANNs 0.20 and Demucs -1.5 dB gates"
        ),
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
