#!/usr/bin/env python3
"""Validate an unscored codec-mixed bank and write a provenance bundle."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
import numpy as np
import soundfile as sf
import yaml


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from build_prospective_mixed_phone_v3 import (  # noqa: E402
    CHANNELS,
    FAKEMUSICCAPS_GENERATORS,
    IDENTITY_COLUMNS,
    MUSIC_IDENTITY_COLUMNS,
    MUSIC_VOCAL_SCREEN_PASS,
    MODES,
    PREDICTION_COLUMNS,
    canonical_music_group,
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def bool_series(frame: pd.DataFrame, column: str) -> pd.Series:
    values = frame[column].astype(str).str.strip().str.lower()
    mapping = {
        "true": True, "1": True, "yes": True, "y": True,
        "false": False, "0": False, "no": False, "n": False,
    }
    unknown = sorted(set(values).difference(mapping))
    if unknown:
        raise ValueError(f"unknown boolean values in {column}: {unknown[:5]}")
    return values.map(mapping).astype(bool)


def path_hash(path: Path) -> dict[str, object]:
    return {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reservation-dir", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--bank-dir", type=Path, required=True)
    parser.add_argument("--acoustic-dir", type=Path, required=True)
    parser.add_argument("--archive-dir", type=Path, required=True)
    parser.add_argument("--archive-volumes", type=Path, required=True)
    parser.add_argument("--partition-config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seven-zip", type=Path, required=True)
    parser.add_argument("--ffmpeg", type=Path, required=True)
    parser.add_argument("--panns-model", type=Path, required=True)
    parser.add_argument("--demucs-model", type=Path, required=True)
    args = parser.parse_args()

    outputs = [
        args.output_dir / "validation.json",
        args.output_dir / "VALIDATION.md",
        args.output_dir / "archive_volume_validation.csv",
        args.output_dir / "role_overlap_by_column.csv",
        args.output_dir / "full_provenance.json",
    ]
    if existing := [path for path in outputs if path.exists()]:
        raise FileExistsError(f"refusing to overwrite: {existing}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    reservation_path = args.reservation_dir / "reservation.csv"
    reservation_provenance_path = args.reservation_dir / "provenance.json"
    truth_path = args.bank_dir / "truth.csv"
    source_hash_path = args.bank_dir / "source_hashes.csv"
    audio_hash_path = args.bank_dir / "audio_hashes.csv"
    submission_path = args.bank_dir / "sample_submission.csv"
    bank_provenance_path = args.bank_dir / "provenance.json"
    acoustic_score_path = args.acoustic_dir / "source_scores.csv"
    acoustic_summary_path = args.acoustic_dir / "summary.json"
    acoustic_generator_path = args.acoustic_dir / "generator_summary.csv"
    requirements_path = args.reservation_dir / "source_requirements.csv"
    targets_path = args.reservation_dir / "archive_member_targets.txt"
    archive_requirements_path = (
        args.reservation_dir / "archive_requirements_provenance.json"
    )
    required_files = [
        reservation_path, reservation_provenance_path, truth_path,
        source_hash_path, audio_hash_path, submission_path,
        bank_provenance_path, acoustic_score_path, acoustic_summary_path,
        acoustic_generator_path, requirements_path, targets_path,
        archive_requirements_path, args.archive_volumes, args.partition_config,
        args.seven_zip, args.ffmpeg, args.panns_model, args.demucs_model,
    ]
    if missing := [str(path) for path in required_files if not path.is_file()]:
        raise FileNotFoundError(f"missing required files: {missing}")

    reservation = pd.read_csv(reservation_path, dtype=str).fillna("")
    truth = pd.read_csv(truth_path, dtype=str).fillna("")
    source_hashes = pd.read_csv(source_hash_path, dtype=str).fillna("")
    audio_hashes = pd.read_csv(audio_hash_path, dtype=str).fillna("")
    submission = pd.read_csv(submission_path, dtype=str).fillna("")
    scores = pd.read_csv(acoustic_score_path, dtype=str).fillna("")
    requirements = pd.read_csv(requirements_path, dtype=str).fillna("")
    reservation_provenance = json.loads(
        reservation_provenance_path.read_text("utf-8")
    )
    bank_provenance = json.loads(bank_provenance_path.read_text("utf-8"))
    acoustic_summary = json.loads(acoustic_summary_path.read_text("utf-8"))

    base_count = len(reservation)
    rendered_count = base_count * len(CHANNELS)
    per_cell = int(reservation_provenance["per_cell"])
    checks: dict[str, bool] = {}
    checks["reservation_hash"] = (
        sha256_file(reservation_path)
        == reservation_provenance["reservation_sha256"]
    )
    checks["reservation_rows"] = base_count == 12 * per_cell == 120
    base_cells = reservation.groupby(["MIX_MODE", "COMPONENT_CASE"]).size()
    checks["reservation_cell_balance"] = (
        len(base_cells) == 12 and set(base_cells.astype(int)) == {per_cell}
    )
    checks["reservation_unique_ids"] = (
        reservation.BASE_ID.nunique() == base_count
        and reservation.VOICE_SOURCE_ID.nunique() == base_count
        and reservation.MUSIC_GROUP_ID.nunique() == base_count
    )
    checks["reservation_labels"] = bool((
        reservation.FILE_FAKE.astype(int)
        == reservation[["VOICE_FAKE", "MUSIC_FAKE"]].astype(int).max(axis=1)
    ).all())
    selected_fake = reservation.loc[reservation.MUSIC_FAKE.astype(int).eq(1)]
    generator_counts = selected_fake.groupby(
        "MUSIC_GENERATOR"
    ).MUSIC_SOURCE_ID.nunique()
    checks["five_balanced_fake_music_generators"] = (
        set(generator_counts.index) == set(FAKEMUSICCAPS_GENERATORS)
        and set(generator_counts.astype(int)) == {12}
    )
    checks["instrumental_text_screen"] = (
        set(selected_fake.MUSIC_VOCAL_SCREEN) == {MUSIC_VOCAL_SCREEN_PASS}
    )

    checks["truth_rows_and_ids"] = (
        len(truth) == rendered_count and truth.ID.nunique() == rendered_count
    )
    rendered_cells = truth.groupby(["MIX_MODE", "COMPONENT_CASE"]).size()
    checks["rendered_cell_balance"] = (
        len(rendered_cells) == 12
        and set(rendered_cells.astype(int)) == {per_cell * len(CHANNELS)}
    )
    channel_counts = truth.groupby("CHANNEL").size()
    parent_counts = truth.groupby("BASE_ID").size()
    checks["paired_channels"] = (
        set(channel_counts.index) == set(CHANNELS)
        and set(channel_counts.astype(int)) == {base_count}
        and set(parent_counts.astype(int)) == {len(CHANNELS)}
    )
    numeric_base_columns = {
        "FILE_FAKE", "VOICE_FAKE", "MUSIC_FAKE", "VOICE_PRESENT",
        "MUSIC_PRESENT", "SNR_DB", "OVERLAP_FRACTION", "GAP_SECONDS",
    }
    base_truth = truth.groupby("BASE_ID").first().loc[reservation.BASE_ID]
    base_reservation = reservation.set_index("BASE_ID").loc[reservation.BASE_ID]
    base_matches = []
    for column in reservation.columns:
        if column == "BASE_ID":
            continue
        single_valued = truth.groupby("BASE_ID")[column].nunique().max() == 1
        if column in numeric_base_columns:
            left = pd.to_numeric(base_truth[column], errors="coerce").to_numpy()
            right = pd.to_numeric(
                base_reservation[column], errors="coerce"
            ).to_numpy()
            equal = (
                np.array_equal(np.isnan(left), np.isnan(right))
                and np.allclose(left, right, rtol=0.0, atol=0.0, equal_nan=True)
            )
        else:
            equal = bool(
                (base_truth[column].to_numpy() ==
                 base_reservation[column].to_numpy()).all()
            )
        base_matches.append(single_valued and equal)
    checks["truth_matches_reservation"] = all(base_matches)

    expected_members = set(requirements.ARCHIVE_MEMBER)
    actual_members = {
        str(path.relative_to(args.source_root))
        for path in args.source_root.rglob("*") if path.is_file()
    }
    target_lines = [
        line for line in targets_path.read_text("utf-8").splitlines() if line
    ]
    checks["source_requirement_rows"] = (
        len(requirements) == 240
        and requirements.ARCHIVE_TARGET.nunique() == 240
    )
    checks["exact_archive_target_list"] = (
        len(target_lines) == 240
        and set(target_lines) == set(requirements.ARCHIVE_TARGET)
    )
    checks["exact_extracted_source_tree"] = (
        actual_members == expected_members and len(actual_members) == 240
    )

    recomputed_source_hashes = []
    for row in source_hashes.itertuples(index=False):
        path = Path(row.LOCAL_PATH)
        recomputed_source_hashes.append(sha256_file(path))
    checks["selected_source_hashes"] = (
        len(source_hashes) == 240
        and source_hashes.SOURCE_ID.nunique() == 240
        and recomputed_source_hashes == list(source_hashes.SHA256)
    )
    score_pass = bool_series(scores, "ACOUSTIC_SCREEN_PASS")
    panns_flags = bool_series(scores, "PANNS_FLAG")
    demucs_flags = bool_series(scores, "DEMUCS_FLAG")
    checks["acoustic_source_alignment"] = (
        len(scores) == 60
        and set(scores.MUSIC_SOURCE_ID) == set(selected_fake.MUSIC_SOURCE_ID)
    )
    checks["frozen_acoustic_thresholds"] = (
        set(scores.PANNS_THRESHOLD.astype(float)) == {0.20}
        and set(scores.DEMUCS_THRESHOLD_DB.astype(float)) == {-1.5}
    )
    checks["acoustic_screen_all_pass"] = (
        bool(score_pass.all())
        and not bool(panns_flags.any())
        and not bool(demucs_flags.any())
        and acoustic_summary["acoustic_screen_passes"] == 60
        and acoustic_summary["acoustic_screen_failures"] == 0
    )
    semantic_exclusions = reservation_provenance["semantic_exclusions"]
    checks["one_deterministic_acoustic_replacement"] = (
        semantic_exclusions["applied"] is True
        and semantic_exclusions["selection_rule"] == "ACOUSTIC_SCREEN_PASS=false"
        and semantic_exclusions["exclusion_unique_ids"] == 1
        and semantic_exclusions["selected_replacement_count"] == 1
        and len(semantic_exclusions["replacements"]) == 1
        and acoustic_summary["status"]
        == "completed_merged_initial_and_replacement"
        and acoustic_summary["initial_screen"]["failures"] == 1
        and acoustic_summary["replacement_recheck"]["passes"] == 1
    )

    audio_dir = args.bank_dir / "audio"
    actual_audio = sorted(audio_dir.glob("*.flac"))
    expected_audio_ids = set(truth.ID)
    checks["exact_rendered_audio_tree"] = (
        len(actual_audio) == rendered_count
        and {path.stem for path in actual_audio} == expected_audio_ids
    )
    properties = []
    recomputed_audio_hashes: dict[str, str] = {}
    for path in actual_audio:
        info = sf.info(path)
        properties.append((info.samplerate, info.channels, info.frames / info.samplerate))
        recomputed_audio_hashes[path.stem] = sha256_file(path)
    checks["audio_format_and_duration"] = all(
        rate == 16_000 and channels == 1 and 4.0 <= duration <= 60.0
        for rate, channels, duration in properties
    )
    manifest_audio_hashes = dict(zip(audio_hashes.ID, audio_hashes.SHA256))
    checks["rendered_audio_hashes"] = (
        len(audio_hashes) == rendered_count
        and manifest_audio_hashes == recomputed_audio_hashes
    )
    checks["submission_template"] = (
        list(submission.columns) == ["ID", *PREDICTION_COLUMNS]
        and set(submission.ID) == expected_audio_ids
        and len(submission) == rendered_count
        and all(set(submission[column].astype(float)) == {0.5}
                for column in PREDICTION_COLUMNS)
    )
    checks["built_provenance_hashes"] = (
        bank_provenance["stage"] == "built_unscored"
        and bank_provenance["detector_inference"] is False
        and bank_provenance["score_computed"] is False
        and bank_provenance["truth_sha256"] == sha256_file(truth_path)
        and bank_provenance["source_hashes_sha256"] == sha256_file(source_hash_path)
        and bank_provenance["audio_hashes_sha256"] == sha256_file(audio_hash_path)
    )
    checks["config_unchanged_since_reservation"] = (
        sha256_file(args.partition_config)
        == reservation_provenance["inputs"]["partition_config"]["sha256"]
    )

    config = yaml.safe_load(args.partition_config.read_text("utf-8")) or {}
    bank_truth_relative = str(truth_path.resolve().relative_to(ROOT.resolve()))
    configured = {
        str(value): role
        for role, values in config.items() if isinstance(values, list)
        for value in values
    }
    checks["no_config_role_added"] = bank_truth_relative not in configured
    v2_relative = "data/eval/prospective_mixed_phone_v3_instrumental_v2/truth.csv"
    checks["v2_retired_role"] = configured.get(v2_relative) == "retrospective_diagnostic"

    overlap_rows = []
    reservation_views = [
        "VOICE_SOURCE_ID", "VOICE_ARCHIVE_ID", "VOICE_ARCHIVE_MEMBER",
        "MUSIC_SOURCE_ID", "MUSIC_ARCHIVE_ID", "MUSIC_ARCHIVE_MEMBER",
        "MUSIC_GROUP_ID",
    ]
    for relative, role in sorted(configured.items()):
        path = ROOT / relative
        other = pd.read_csv(path, dtype=str).fillna("")
        for column in reservation_views:
            if column in other:
                left = set(reservation[column]).difference({""})
                right = set(other[column]).difference({""})
                overlap_rows.append({
                    "ROLE": role, "TRUTH": relative, "VIEW": column,
                    "INTERSECTION": len(left.intersection(right)),
                })
        music_groups: set[str] = set()
        for column in MUSIC_IDENTITY_COLUMNS:
            if column in other:
                music_groups.update(
                    canonical_music_group(value)
                    for value in other[column] if value
                )
        overlap_rows.append({
            "ROLE": role, "TRUTH": relative,
            "VIEW": "CANONICAL_MUSIC_GROUP",
            "INTERSECTION": len(set(reservation.MUSIC_GROUP_ID).intersection(music_groups)),
        })
    overlaps = pd.DataFrame(overlap_rows)
    overlaps.to_csv(args.output_dir / "role_overlap_by_column.csv", index=False)
    checks["all_configured_identity_overlap_zero"] = bool(
        (overlaps.INTERSECTION.astype(int) == 0).all()
    )
    v2_overlaps = overlaps.loc[overlaps.TRUTH.eq(v2_relative)]
    checks["all_v2_retired_identity_overlap_zero"] = (
        len(v2_overlaps) == 8
        and bool((v2_overlaps.INTERSECTION.astype(int) == 0).all())
    )

    volume_manifest = pd.read_csv(args.archive_volumes, dtype=str).fillna("")
    def validate_volume(row: object) -> dict[str, object]:
        path = args.archive_dir / str(row.VOLUME)
        actual = sha256_file(path) if path.is_file() else ""
        return {
            "VOLUME": row.VOLUME,
            "EXPECTED_BYTES": int(row.BYTES),
            "ACTUAL_BYTES": path.stat().st_size if path.is_file() else -1,
            "EXPECTED_SHA256": row.EXPECTED_SHA256,
            "ACTUAL_SHA256": actual,
            "PASS": (
                path.is_file()
                and path.stat().st_size == int(row.BYTES)
                and actual == row.EXPECTED_SHA256
            ),
        }
    with ThreadPoolExecutor(max_workers=8) as executor:
        volume_results = list(executor.map(
            validate_volume, volume_manifest.itertuples(index=False)
        ))
    volume_validation = pd.DataFrame(volume_results).sort_values("VOLUME")
    volume_validation.to_csv(
        args.output_dir / "archive_volume_validation.csv", index=False
    )
    checks["archive_volume_integrity"] = (
        len(volume_validation) == 67
        and bool(volume_validation.PASS.all())
    )

    result = {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks_passed": sum(checks.values()),
        "checks_total": len(checks),
        "checks": checks,
        "counts": {
            "base_rows": base_count,
            "rendered_rows": len(truth),
            "rendered_audio_files": len(actual_audio),
            "selected_source_files": len(actual_members),
            "fake_music_sources_screened": len(scores),
            "archive_volumes": len(volume_validation),
            "registered_truth_manifests_checked": len(configured),
            "identity_overlap_checks": len(overlaps),
        },
        "duration_seconds": {
            "minimum": min(value[2] for value in properties),
            "maximum": max(value[2] for value in properties),
        },
        "fake_music_by_generator": {
            key: int(value) for key, value in generator_counts.items()
        },
        "acoustic_flags": {
            "panns": int(panns_flags.sum()),
            "demucs": int(demucs_flags.sum()),
        },
    }
    (args.output_dir / "validation.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    if result["status"] != "PASS":
        failed = [name for name, passed in checks.items() if not passed]
        raise RuntimeError(f"validation failed: {failed}")

    artifacts = {
        "partition_config": path_hash(args.partition_config),
        "reservation": path_hash(reservation_path),
        "reservation_provenance": path_hash(reservation_provenance_path),
        "source_requirements": path_hash(requirements_path),
        "archive_member_targets": path_hash(targets_path),
        "archive_volumes": path_hash(args.archive_volumes),
        "archive_requirements_provenance": path_hash(archive_requirements_path),
        "acoustic_source_scores": path_hash(acoustic_score_path),
        "acoustic_generator_summary": path_hash(acoustic_generator_path),
        "acoustic_summary": path_hash(acoustic_summary_path),
        "initial_reservation": path_hash(
            ROOT / "reports/codec_mixed_dev_v4_design/reservation_initial/reservation.csv"
        ),
        "identity_corrected_reservation": path_hash(
            ROOT / "reports/codec_mixed_dev_v4_design/reservation_identity_corrected/reservation.csv"
        ),
        "initial_corrected_acoustic_source_scores": path_hash(
            ROOT / "reports/codec_mixed_dev_v4_design/acoustic_screen_identity_corrected/source_scores.csv"
        ),
        "replacement_acoustic_source_scores": path_hash(
            ROOT / "reports/codec_mixed_dev_v4_design/acoustic_recheck_replacement/source_scores.csv"
        ),
        "truth": path_hash(truth_path),
        "sample_submission": path_hash(submission_path),
        "selected_source_hashes": path_hash(source_hash_path),
        "rendered_audio_hashes": path_hash(audio_hash_path),
        "bank_provenance": path_hash(bank_provenance_path),
        "builder": path_hash(ROOT / "scripts/build_prospective_mixed_phone_v3.py"),
        "acoustic_auditor": path_hash(ROOT / "scripts/audit_prospective_music_vocals.py"),
        "source_reporter": path_hash(ROOT / "scripts/report_mixfake_reservation_sources.py"),
        "acoustic_evidence_merger": path_hash(
            ROOT / "scripts/merge_acoustic_replacement_screen.py"
        ),
        "validator": path_hash(Path(__file__)),
        "artifactnet_detector_preserved": path_hash(
            ROOT / "src/artifactnet_detector.py"
        ),
        "panns_model": path_hash(args.panns_model),
        "demucs_model": path_hash(args.demucs_model),
        "seven_zip": path_hash(args.seven_zip),
        "ffmpeg": path_hash(args.ffmpeg),
        "validation": path_hash(args.output_dir / "validation.json"),
        "archive_volume_validation": path_hash(
            args.output_dir / "archive_volume_validation.csv"
        ),
        "role_overlap_by_column": path_hash(
            args.output_dir / "role_overlap_by_column.csv"
        ),
    }
    archive_requirements = json.loads(archive_requirements_path.read_text("utf-8"))
    full_provenance = {
        "schema_version": 1,
        "purpose": "reusable codec-invariant development/selection bank",
        "status": "built_unscored_validated_not_registered",
        "seed": int(reservation_provenance["seed"]),
        "per_cell": per_cell,
        "reservation_history": {
            "requested_initial": (
                "reports/codec_mixed_dev_v4_design/reservation_initial"
            ),
            "identity_corrected": (
                "reports/codec_mixed_dev_v4_design/"
                "reservation_identity_corrected"
            ),
            "final_one_time_acoustic_replacement": (
                "reports/codec_mixed_dev_v4_design/reservation_replacement"
            ),
            "replacement": semantic_exclusions["replacements"][0],
        },
        "archive": {
            "repository": archive_requirements["repository"],
            "revision": archive_requirements["revision"],
            "volumes": 67,
            "total_bytes": int(volume_validation.ACTUAL_BYTES.sum()),
            "all_sha256_verified": True,
        },
        "source_extraction": {
            "source_root": str(args.source_root.resolve()),
            "seven_zip_version": subprocess.run(
                [str(args.seven_zip), "i"], check=True,
                capture_output=True, text=True,
            ).stdout.splitlines()[1].strip(),
            "requested_members": len(target_lines),
            "extracted_files": len(actual_members),
            "missing_members": 0,
            "extra_members": 0,
        },
        "acoustic_semantic_screen": {
            "musiccaps_text_screen": MUSIC_VOCAL_SCREEN_PASS,
            "panns_threshold": 0.20,
            "demucs_vocal_to_mix_threshold_db": -1.5,
            "screened_fake_music_sources": len(scores),
            "passes": int(score_pass.sum()),
            "failures": int((~score_pass).sum()),
            "replacement_reservation_created": True,
            "replacement_count": 1,
            "authenticity_detector": False,
        },
        "bank": {
            "directory": str(args.bank_dir.resolve()),
            "base_rows": base_count,
            "rendered_rows": len(truth),
            "channels": list(CHANNELS),
            "mix_modes": list(MODES),
            "config_role_added": False,
        },
        "leakage_controls": {
            "partition_roles_protected": list(config),
            "v2_role": configured[v2_relative],
            "v2_identity_overlap": 0,
            "all_registered_identity_overlap": 0,
            "prospective_v2_predictions_or_scores_read": False,
            "authenticity_detector_run": False,
            "authenticity_score_computed": False,
        },
        "artifacts": artifacts,
    }
    (args.output_dir / "full_provenance.json").write_text(
        json.dumps(full_provenance, indent=2) + "\n", encoding="utf-8"
    )

    validation_lines = [
        "# Codec mixed development v4 validation",
        "",
        "## Result",
        "",
        f"**{result['checks_passed']}/{result['checks_total']} checks PASS.**",
        "The bank is built, unscored, validated, and intentionally absent from ",
        "`configs/data_partitions.yaml` pending an explicit later registration decision.",
        "No authenticity detector was run and no prospective-v2 prediction or score was read.",
        "",
        "## Counts",
        "",
        f"- {base_count} unique bases and {len(truth)} paired codec renders.",
        f"- 12 cells with {per_cell} bases / {per_cell * len(CHANNELS)} renders each.",
        f"- {len(actual_members)} exact extracted sources; no missing or extra members.",
        f"- {len(scores)}/{len(scores)} fake-music sources pass PANNs 0.20 and Demucs -1.5 dB.",
        "- Five FakeMusicCaps generators contribute exactly 12 fake-music sources each.",
        f"- {len(volume_validation)}/{len(volume_validation)} archive volume hashes match the fixed repository revision.",
        f"- {len(overlaps)} identity-view comparisons across {len(configured)} configured manifests have zero overlap.",
        "",
        "## Core hashes",
        "",
        "| Artifact | SHA-256 |",
        "| --- | --- |",
        f"| Reservation | `{artifacts['reservation']['sha256']}` |",
        f"| Acoustic source scores | `{artifacts['acoustic_source_scores']['sha256']}` |",
        f"| Truth | `{artifacts['truth']['sha256']}` |",
        f"| Selected-source hash manifest | `{artifacts['selected_source_hashes']['sha256']}` |",
        f"| Rendered-audio hash manifest | `{artifacts['rendered_audio_hashes']['sha256']}` |",
        f"| Builder | `{artifacts['builder']['sha256']}` |",
        f"| Validation JSON | `{artifacts['validation']['sha256']}` |",
        "",
        "Full paths, tool/model hashes, archive revision and validation evidence are in ",
        "`full_provenance.json`, `archive_volume_validation.csv`, and ",
        "`role_overlap_by_column.csv`.",
    ]
    (args.output_dir / "VALIDATION.md").write_text(
        "\n".join(validation_lines) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
