#!/usr/bin/env python3
"""Build the acoustic-gated FakeMusicCaps reservation used by blind v6.

The normal prospective-bank builder requires positive instrumental wording in
MusicCaps.  That conservative pool is exhausted for two requested generators
after protecting every registered role.  This utility keeps the positive-vocal
text rejection, admits otherwise neutral captions as *pending*, and only
promotes them after an independent PANNs/Demucs semantic screen passes.

It never imports or reads an authenticity detector or authenticity score.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import sys

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from build_prospective_mixed_phone_v3 import (  # noqa: E402
    MUSIC_VOCAL_SCREEN_PASS,
    canonical_music_group,
    load_musiccaps_metadata,
    music_identity,
    musiccaps_vocal_screen,
    parse_mixfake_protocol,
    protected_identities,
    sha256_file,
    stable_rank,
)


GENERATORS = ("MusicGen_medium", "audioldm2", "stable_audio_open")
PENDING = "musiccaps_no_positive_voice_signal_acoustic_pending_v1"
ACOUSTIC_PASS = "musiccaps_no_positive_voice_text_panns_demucs_pass_v1"


def parse_bool(series: pd.Series) -> pd.Series:
    mapping = {
        "true": True, "1": True, "yes": True, "y": True,
        "false": False, "0": False, "no": False, "n": False,
    }
    values = series.astype(str).str.strip().str.lower()
    unknown = sorted(set(values).difference(mapping))
    if unknown:
        raise ValueError(f"unknown boolean values: {unknown[:5]}")
    return values.map(mapping).astype(bool)


def candidate_pool(
    unmixed_details_path: Path,
    background_protocol_path: Path,
    musiccaps_metadata_path: Path,
    partition_config: Path,
) -> tuple[list[dict[str, str]], dict[str, int]]:
    details = pd.read_csv(unmixed_details_path, dtype=str).fillna("")
    protocol = parse_mixfake_protocol(background_protocol_path)
    metadata = load_musiccaps_metadata(musiccaps_metadata_path)
    protected_exact, protected_music = protected_identities(partition_config)
    rows: list[dict[str, str]] = []
    counts: Counter[str] = Counter()
    selected = details.loc[
        details["split"].eq("eval")
        & details.major_type.eq("Background")
        & details.sub_type.eq("Music")
        & details.authenticity.eq("spoof")
        & details.sub_dataset.eq("FakeMusicCaps")
    ]
    for row in selected.itertuples(index=False):
        archive_id = Path(row.file_path).stem
        source_id, group, generator = music_identity(archive_id, row.sub_dataset)
        if generator not in GENERATORS:
            continue
        counts[f"{generator}:raw"] += 1
        passed, reason = musiccaps_vocal_screen(
            metadata.get(group.removeprefix("fmc:"))
        )
        if reason == "excluded_positive_voice_signal":
            counts[f"{generator}:positive_vocal_rejected"] += 1
            continue
        if source_id in protected_exact or group in protected_music:
            counts[f"{generator}:registered_identity_rejected"] += 1
            continue
        protocol_row = protocol.get(archive_id)
        if protocol_row is None or protocol_row["label"] != "spoof":
            raise ValueError(f"missing/conflicting protocol row: {archive_id}")
        screen = MUSIC_VOCAL_SCREEN_PASS if passed else PENDING
        counts[f"{generator}:eligible_{'text_pass' if passed else 'acoustic_pending'}"] += 1
        rows.append({
            "source_id": source_id,
            "group": group,
            "generator": generator,
            "archive_id": archive_id,
            "archive_member": protocol_row["member"],
            "vocal_screen": screen,
        })
    return rows, dict(sorted(counts.items()))


def select_initial(
    candidates: list[dict[str, str]], per_generator: int, seed: int,
) -> list[dict[str, str]]:
    buckets: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in candidates:
        buckets[row["generator"]].append(row)
    for generator in GENERATORS:
        buckets[generator].sort(key=lambda row: (
            row["vocal_screen"] != MUSIC_VOCAL_SCREEN_PASS,
            stable_rank(seed, generator, row["group"], row["source_id"]),
        ))
    selected: list[dict[str, str]] = []
    used_groups: set[str] = set()
    # Generator order is fixed and rotated by rank so no archive order enters.
    order = sorted(GENERATORS, key=lambda value: stable_rank(seed, "generator", value))
    for generator in order:
        for row in buckets[generator]:
            if row["group"] in used_groups:
                continue
            selected.append(row)
            used_groups.add(row["group"])
            if sum(item["generator"] == generator for item in selected) == per_generator:
                break
        count = sum(item["generator"] == generator for item in selected)
        if count != per_generator:
            raise ValueError(f"need {per_generator} {generator} sources, found {count}")
    return selected


def replace_failures(
    frame: pd.DataFrame,
    candidates: list[dict[str, str]],
    scores: pd.DataFrame,
    seed: int,
) -> tuple[pd.DataFrame, list[dict[str, str]]]:
    passed = parse_bool(scores.ACOUSTIC_SCREEN_PASS)
    failures = set(scores.loc[~passed, "MUSIC_SOURCE_ID"].astype(str))
    selected_ids = set(frame.MUSIC_SOURCE_ID.astype(str))
    used_groups = set(frame.MUSIC_GROUP_ID.astype(str))
    replacements: list[dict[str, str]] = []
    result = frame.copy()
    for source_id in sorted(failures):
        matches = result.index[result.MUSIC_SOURCE_ID.eq(source_id)]
        if len(matches) != 1:
            raise ValueError(f"failed source must occur once: {source_id}")
        index = int(matches[0])
        generator = str(result.at[index, "MUSIC_GENERATOR"])
        old_group = str(result.at[index, "MUSIC_GROUP_ID"])
        used_groups.remove(old_group)
        available = [
            row for row in candidates
            if row["generator"] == generator
            and row["source_id"] not in selected_ids
            and row["source_id"] not in failures
            and row["group"] not in used_groups
        ]
        available.sort(key=lambda row: (
            row["vocal_screen"] != MUSIC_VOCAL_SCREEN_PASS,
            stable_rank(seed, "replacement", generator, row["group"], row["source_id"]),
        ))
        if not available:
            raise ValueError(f"no same-generator replacement for {source_id}")
        replacement = available[0]
        selected_ids.add(replacement["source_id"])
        used_groups.add(replacement["group"])
        for column, key in (
            ("MUSIC_SOURCE_ID", "source_id"),
            ("MUSIC_GROUP_ID", "group"),
            ("MUSIC_GENERATOR", "generator"),
            ("MUSIC_VOCAL_SCREEN", "vocal_screen"),
            ("MUSIC_ARCHIVE_ID", "archive_id"),
            ("MUSIC_ARCHIVE_MEMBER", "archive_member"),
        ):
            result.at[index, column] = replacement[key]
        replacements.append({
            "excluded_source_id": source_id,
            "excluded_group": old_group,
            "replacement_source_id": replacement["source_id"],
            "replacement_group": replacement["group"],
            "generator": generator,
        })
    return result, replacements


def validate_structure(frame: pd.DataFrame, per_generator: int) -> None:
    if len(frame) != 48 or frame.BASE_ID.nunique() != 48:
        raise ValueError("reservation must contain 48 unique bases")
    cells = frame.groupby(["MIX_MODE", "COMPONENT_CASE"]).size()
    if len(cells) != 12 or set(cells.astype(int)) != {4}:
        raise ValueError(f"unbalanced layout/case cells: {cells.to_dict()}")
    for column in ("VOICE_SOURCE_ID", "MUSIC_SOURCE_ID", "MUSIC_GROUP_ID"):
        if frame[column].nunique() != len(frame):
            raise ValueError(f"source reuse in {column}")
    fake = frame.loc[frame.MUSIC_FAKE.astype(int).eq(1)]
    counts = Counter(fake.MUSIC_GENERATOR)
    if counts != Counter({generator: per_generator for generator in GENERATORS}):
        raise ValueError(f"fake-generator imbalance: {counts}")
    if set(fake.MUSIC_VOCAL_SCREEN).difference(
        {MUSIC_VOCAL_SCREEN_PASS, PENDING, ACOUSTIC_PASS}
    ):
        raise ValueError("fake music has an unsupported semantic status")


def write_output(
    output_dir: Path,
    frame: pd.DataFrame,
    provenance: dict[str, object],
) -> None:
    if output_dir.exists():
        raise FileExistsError(output_dir)
    output_dir.mkdir(parents=True)
    reservation_path = output_dir / "reservation.csv"
    frame.to_csv(reservation_path, index=False)
    provenance["reservation_sha256"] = sha256_file(reservation_path)
    provenance["builder_sha256"] = sha256_file(Path(__file__))
    (output_dir / "provenance.json").write_text(
        json.dumps(provenance, indent=2) + "\n", "utf-8"
    )
    requirements = pd.DataFrame({
        "KIND": "music",
        "CANONICAL_GROUP": frame.MUSIC_GROUP_ID,
        "SOURCE_ID": frame.MUSIC_SOURCE_ID,
        "ARCHIVE_ID": frame.MUSIC_ARCHIVE_ID,
        "ARCHIVE_MEMBER": frame.MUSIC_ARCHIVE_MEMBER,
        "ARCHIVE_TARGET": "MixFake/" + frame.MUSIC_ARCHIVE_MEMBER.astype(str),
    }).drop_duplicates("SOURCE_ID").sort_values("ARCHIVE_TARGET")
    requirements.to_csv(output_dir / "source_requirements.csv", index=False)
    (output_dir / "archive_member_targets.txt").write_text(
        "\n".join(requirements.ARCHIVE_TARGET) + "\n", "utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("initial", "replace", "finalize"))
    parser.add_argument("--base-reservation-dir", type=Path, required=True)
    parser.add_argument("--partition-config", type=Path, required=True)
    parser.add_argument("--unmixed-details", type=Path, required=True)
    parser.add_argument("--background-protocol", type=Path, required=True)
    parser.add_argument("--musiccaps-metadata", type=Path, required=True)
    parser.add_argument("--acoustic-scores", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260912)
    args = parser.parse_args()

    base_path = args.base_reservation_dir / "reservation.csv"
    base_provenance_path = args.base_reservation_dir / "provenance.json"
    frame = pd.read_csv(base_path, dtype=str).fillna("")
    base_provenance = json.loads(base_provenance_path.read_text("utf-8"))
    candidates, counts = candidate_pool(
        args.unmixed_details, args.background_protocol,
        args.musiccaps_metadata, args.partition_config,
    )
    replacements: list[dict[str, str]] = []
    if args.mode == "initial":
        selected = select_initial(candidates, 8, args.seed + 21)
        target_indices = list(frame.index[frame.MUSIC_FAKE.astype(int).eq(1)])
        selected.sort(key=lambda row: stable_rank(args.seed, "assignment", row["source_id"]))
        for index, row in zip(target_indices, selected, strict=True):
            for column, key in (
                ("MUSIC_SOURCE_ID", "source_id"),
                ("MUSIC_GROUP_ID", "group"),
                ("MUSIC_GENERATOR", "generator"),
                ("MUSIC_VOCAL_SCREEN", "vocal_screen"),
                ("MUSIC_ARCHIVE_ID", "archive_id"),
                ("MUSIC_ARCHIVE_MEMBER", "archive_member"),
            ):
                frame.at[index, column] = row[key]
    else:
        if args.acoustic_scores is None:
            raise ValueError(f"{args.mode} requires --acoustic-scores")
        scores = pd.read_csv(args.acoustic_scores, dtype=str).fillna("")
        if args.mode == "replace":
            frame, replacements = replace_failures(
                frame, candidates, scores, args.seed + 21,
            )
        else:
            fake_ids = set(frame.loc[
                frame.MUSIC_FAKE.astype(int).eq(1), "MUSIC_SOURCE_ID"
            ])
            if set(scores.MUSIC_SOURCE_ID) != fake_ids or not parse_bool(
                scores.ACOUSTIC_SCREEN_PASS
            ).all():
                raise ValueError("final acoustic evidence is incomplete or failed")
            pending = frame.MUSIC_VOCAL_SCREEN.eq(PENDING)
            frame.loc[pending, "MUSIC_VOCAL_SCREEN"] = ACOUSTIC_PASS
    validate_structure(frame, 8)
    provenance = {
        **base_provenance,
        "stage": f"reserved_blind_v6_{args.mode}",
        "seed": args.seed,
        "id_prefix": "cmbv6",
        "base_reservation": {
            "path": str(base_path.resolve()), "sha256": sha256_file(base_path),
        },
        "actual_partition_config": {
            "path": str(args.partition_config.resolve()),
            "sha256": sha256_file(args.partition_config),
        },
        "semantic_policy": {
            "positive_vocal_text_rejection": True,
            "neutral_text_status_before_audio_screen": PENDING,
            "neutral_text_status_after_audio_screen": ACOUSTIC_PASS,
            "panns_max": 0.20,
            "demucs_vocals_to_mixture_db_max": -1.5,
        },
        "selection": {
            "protected_roles": "all roles in actual partition_config",
            "canonical_music_group_exclusion": True,
            "requested_fake_music_generators": list(GENERATORS),
            "fake_music_sources_per_generator": 8,
            "generator_ood_claim": False,
            "within_holdout_source_reuse": "paired channel variants only",
        },
        "candidate_counts": counts,
        "replacements": replacements,
        "acoustic_scores": None if args.acoustic_scores is None else {
            "path": str(args.acoustic_scores.resolve()),
            "sha256": sha256_file(args.acoustic_scores),
        },
        "authenticity_detector_inference": False,
        "authenticity_score_computed": False,
    }
    write_output(args.output_dir, frame, provenance)
    fake = frame.loc[frame.MUSIC_FAKE.astype(int).eq(1)]
    print(fake.groupby(["MUSIC_GENERATOR", "MUSIC_VOCAL_SCREEN"]).size())
    print(f"wrote {len(frame)} bases to {args.output_dir}")


if __name__ == "__main__":
    main()
