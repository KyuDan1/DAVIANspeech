#!/usr/bin/env python3
"""Validate a built mixed/codec bank before registering or scoring it."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import soundfile as sf
import yaml


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from data_guard import identity_tokens  # noqa: E402
from build_prospective_mixed_phone_v3 import (  # noqa: E402
    CHANNELS, MUSIC_ACOUSTIC_SCREEN_PASS, MUSIC_VOCAL_SCREEN_PASS,
    validate_reservation,
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def boolean_values(series: pd.Series) -> pd.Series:
    mapping = {
        "true": True, "1": True, "yes": True, "y": True,
        "false": False, "0": False, "no": False, "n": False,
    }
    values = series.astype(str).str.strip().str.lower()
    unknown = sorted(set(values).difference(mapping))
    if unknown:
        raise ValueError(f"unknown boolean values: {unknown[:5]}")
    return values.map(mapping).astype(bool)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reservation-dir", type=Path, required=True)
    parser.add_argument("--bank-dir", type=Path, required=True)
    parser.add_argument("--acoustic-dir", type=Path, required=True)
    parser.add_argument("--partition-config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite {args.output_dir}")

    reservation_path = args.reservation_dir / "reservation.csv"
    reservation_provenance_path = args.reservation_dir / "provenance.json"
    truth_path = args.bank_dir / "truth.csv"
    bank_provenance_path = args.bank_dir / "provenance.json"
    source_hash_path = args.bank_dir / "source_hashes.csv"
    audio_hash_path = args.bank_dir / "audio_hashes.csv"
    acoustic_path = args.acoustic_dir / "source_scores.csv"
    paths = (
        reservation_path, reservation_provenance_path, truth_path,
        bank_provenance_path, source_hash_path, audio_hash_path, acoustic_path,
        args.partition_config,
    )
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(path)

    reservation = pd.read_csv(reservation_path, dtype=str).fillna("")
    reservation_provenance = json.loads(
        reservation_provenance_path.read_text("utf-8")
    )
    generators = tuple(
        reservation_provenance["selection"][
            "requested_fake_music_generators"
        ]
    )
    per_cell = int(reservation_provenance["per_cell"])
    validate_reservation(reservation, per_cell, generators)
    if sha256_file(reservation_path) != reservation_provenance["reservation_sha256"]:
        raise ValueError("reservation hash mismatch")

    truth = pd.read_csv(truth_path, dtype=str).fillna("")
    bank_provenance = json.loads(bank_provenance_path.read_text("utf-8"))
    checks: dict[str, bool] = {}
    checks["bank_unscored"] = (
        bank_provenance.get("detector_inference") is False
        and bank_provenance.get("score_computed") is False
    )
    checks["reservation_hash_carried"] = (
        bank_provenance.get("reservation_sha256")
        == reservation_provenance["reservation_sha256"]
    )
    checks["truth_hash"] = sha256_file(truth_path) == bank_provenance["truth_sha256"]
    checks["row_count"] = len(truth) == len(reservation) * len(CHANNELS)
    checks["unique_ids"] = truth.ID.nunique() == len(truth)
    checks["paired_channels"] = (
        set(truth.CHANNEL) == set(CHANNELS)
        and truth.groupby("BASE_ID").CHANNEL.nunique().eq(len(CHANNELS)).all()
    )
    common = [column for column in reservation if column != "BASE_ID"]
    merged = truth.merge(
        reservation, on="BASE_ID", how="left", suffixes=("_truth", "_reserved"),
        validate="many_to_one",
    )
    numeric_recipe_columns = {
        "FILE_FAKE", "VOICE_FAKE", "MUSIC_FAKE", "VOICE_PRESENT",
        "MUSIC_PRESENT", "SNR_DB", "OVERLAP_FRACTION", "GAP_SECONDS",
    }
    recipe_checks = []
    for column in common:
        truth_values = merged[f"{column}_truth"]
        reserved_values = merged[f"{column}_reserved"]
        if column in numeric_recipe_columns:
            recipe_checks.append(np.allclose(
                pd.to_numeric(truth_values, errors="coerce"),
                pd.to_numeric(reserved_values, errors="coerce"),
                equal_nan=True,
            ))
        else:
            recipe_checks.append(truth_values.equals(reserved_values))
    checks["recipe_metadata"] = all(recipe_checks)

    audio_dir = args.bank_dir / "audio"
    audio = sorted(audio_dir.glob("*.flac"))
    checks["audio_id_alignment"] = {path.stem for path in audio} == set(truth.ID)
    properties = [sf.info(path) for path in audio]
    checks["audio_format"] = bool(properties) and all(
        item.samplerate == 16_000
        and item.channels in (1, 2)
        and 4.0 <= item.duration <= 60.0
        for item in properties
    )
    audio_hashes = pd.read_csv(audio_hash_path, dtype=str).fillna("")
    recorded_audio = dict(zip(audio_hashes.ID, audio_hashes.SHA256))
    checks["audio_hash_rows"] = set(recorded_audio) == set(truth.ID)
    checks["audio_hash_values"] = checks["audio_hash_rows"] and all(
        recorded_audio[path.stem] == sha256_file(path) for path in audio
    )

    source_hashes = pd.read_csv(source_hash_path, dtype=str).fillna("")
    checks["source_count"] = (
        len(source_hashes) == 2 * len(reservation)
        and source_hashes.SOURCE_ID.nunique() == len(source_hashes)
    )
    required_sources = set(reservation.VOICE_SOURCE_ID) | set(
        reservation.MUSIC_SOURCE_ID
    )
    checks["source_id_alignment"] = set(source_hashes.SOURCE_ID) == required_sources

    acoustic = pd.read_csv(acoustic_path, dtype=str).fillna("")
    fake_music = set(
        reservation.loc[
            reservation.MUSIC_FAKE.astype(int).eq(1), "MUSIC_SOURCE_ID"
        ]
    )
    checks["acoustic_id_alignment"] = set(acoustic.MUSIC_SOURCE_ID) == fake_music
    checks["acoustic_all_pass"] = boolean_values(
        acoustic.ACOUSTIC_SCREEN_PASS
    ).all()
    checks["acoustic_thresholds"] = (
        set(acoustic.PANNS_THRESHOLD.astype(float)) == {0.20}
        and set(acoustic.DEMUCS_THRESHOLD_DB.astype(float)) == {-1.5}
        and set(reservation.loc[
            reservation.MUSIC_FAKE.astype(int).eq(1), "MUSIC_VOCAL_SCREEN"
        ]).issubset({MUSIC_VOCAL_SCREEN_PASS, MUSIC_ACOUSTIC_SCREEN_PASS})
    )
    generator_counts = reservation.loc[
        reservation.MUSIC_FAKE.astype(int).eq(1), "MUSIC_GENERATOR"
    ].value_counts()
    checks["generator_balance"] = (
        set(generator_counts.index) == set(generators)
        and generator_counts.nunique() == 1
    )

    config = yaml.safe_load(args.partition_config.read_text("utf-8")) or {}
    root = args.partition_config.resolve().parent.parent
    candidate_tokens = identity_tokens(reservation)
    overlap_rows = []
    for role, relatives in config.items():
        if not isinstance(relatives, list):
            continue
        for relative in relatives:
            path = root / relative
            overlap = sorted(
                candidate_tokens & identity_tokens(pd.read_csv(path, dtype=str))
            )
            overlap_rows.append({
                "ROLE": role, "PATH": relative, "OVERLAP": len(overlap),
                "EXAMPLES": "|".join(overlap[:5]),
            })
    overlap_frame = pd.DataFrame(overlap_rows)
    checks["registered_identity_disjoint"] = overlap_frame.OVERLAP.eq(0).all()

    checks = {name: bool(passed) for name, passed in checks.items()}
    result = {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "checks_passed": int(sum(checks.values())),
        "checks_total": len(checks),
        "base_rows": len(reservation),
        "rendered_rows": len(truth),
        "fake_music_sources": len(fake_music),
        "fake_music_by_generator": {
            key: int(value) for key, value in generator_counts.items()
        },
        "truth_sha256": sha256_file(truth_path),
        "reservation_sha256": sha256_file(reservation_path),
        "authenticity_detector_scores_read": False,
    }
    args.output_dir.mkdir(parents=True)
    overlap_frame.to_csv(args.output_dir / "role_overlap.csv", index=False)
    (args.output_dir / "validation.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    if result["status"] != "PASS":
        failed = [name for name, passed in checks.items() if not passed]
        raise RuntimeError(f"validation failed: {failed}")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
