#!/usr/bin/env python3
"""Build the source/speaker/song-disjoint blind mixed-codec v7 bank.

This is a truth-construction utility, not an authenticity experiment.  Its
inputs are source metadata/audio, MixFake protocols, and an optional semantic
voice-in-music audit produced by PANNs/Demucs.  It deliberately has no
prediction, detector checkpoint, probability, or scoring argument.

Lifecycle (each stage is immutable)::

    reserve -> materialize-voices/extract music -> semantic-screen
            -> revise (repeat until all pass) -> validate-sources
            -> render -> register the frozen truth under locked_eval

The reservation snapshots *every* pre-existing ``data/**/truth*.csv`` and
source-hash manifest, rather than trusting the current role registry alone.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass
from io import BytesIO
import hashlib
import json
import math
from pathlib import Path
import re
import shutil
import subprocess
import sys
from typing import Iterable

import numpy as np
import pandas as pd
import soundfile as sf
from scipy.signal import resample_poly


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))
from build_prospective_mixed_phone_v3 import (  # noqa: E402
    CHANNELS,
    CELLS,
    MODES,
    MUSIC_ACOUSTIC_SCREEN_PASS,
    MUSIC_VOCAL_SCREEN_PASS,
    PREDICTION_COLUMNS,
    canonical_music_group,
    load_audio,
    load_musiccaps_metadata,
    mix_layout,
    music_identity,
    musiccaps_vocal_screen,
    parse_mixfake_protocol,
    sha256_file,
    stable_rank,
)
from telephone_channel import apply_channel  # noqa: E402


SR = 16_000
DEFAULT_SEED = 20260914
DEFAULT_PER_CELL = 6
ID_PREFIX = "cmbv7"
FAKE_MUSIC_GENERATORS = (
    "MusicGen_medium", "audioldm2", "musicldm", "mustango",
    "stable_audio_open",
)
ECHO_LABELS = {"bonafide": 0, "fake": 1}
PENDING_SEMANTIC = "panns_demucs_semantic_screen_pending_v7"
SEMANTIC_PASS = MUSIC_ACOUSTIC_SCREEN_PASS
REAL_MUSIC_GENERATOR = "FMA"
FORBIDDEN_PREDICTION_COLUMNS = {
    "FILE_FAKE_PROB", "VOICE_FAKE_PROB", "MUSIC_FAKE_PROB",
    "VOICE_PRESENT_PROB", "MUSIC_PRESENT_PROB", "EER", "ADS", "CPS",
}

# Includes historical schemas not covered by src/data_guard.py.  Values are
# protected raw and by filename stem; speakers and reference speakers are
# included explicitly.
EXACT_IDENTITY_COLUMNS = (
    "ID", "BASE_ID", "MIXTURE_ID", "PAIR_ID", "GROUP_ID", "SOURCE",
    "SOURCE_ID", "SPEAKER_ID", "SPEAKER",
    "VOICE_SOURCE_ID", "VOICE_SPEAKER", "VOICE_REFERENCE_ID",
    "VOICE_REFERENCE_SPEAKER", "VOICE_CONTENT_ID", "VOICE_ARCHIVE_ID",
    "MUSIC_SOURCE_ID", "MUSIC_ARCHIVE_ID", "VOICE_GROUP", "SOURCE_FILE",
    "FIRST_SOURCE_ID", "SECOND_SOURCE_ID", "FIRST_GROUP", "SECOND_GROUP",
    "PAIR_GROUP", "PARENT_ID", "FMA_TRACK_ID", "MUSIC_GROUP",
    "MUSIC_GROUP_ID",
)
MUSIC_IDENTITY_COLUMNS = (
    "MUSIC_SOURCE_ID", "MUSIC_ARCHIVE_ID", "MUSIC_GROUP",
    "MUSIC_GROUP_ID", "FMA_TRACK_ID",
)


def clean(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    return str(value).strip()


def identity_variants(value: object) -> set[str]:
    raw = clean(value).replace("\\", "/")
    if not raw:
        return set()
    variants = {raw, raw.lower()}
    stem = Path(raw).stem
    if stem:
        variants.update({stem, stem.lower()})
    return variants


def bool_series(series: pd.Series) -> pd.Series:
    mapping = {
        "true": True, "1": True, "yes": True, "y": True,
        "false": False, "0": False, "no": False, "n": False,
    }
    values = series.astype(str).str.strip().str.lower()
    unknown = sorted(set(values).difference(mapping))
    if unknown:
        raise ValueError(f"unknown boolean values: {unknown[:5]}")
    return values.map(mapping).astype(bool)


def atomic_output_dir(destination: Path) -> tuple[Path, callable]:
    """Return a sibling staging directory and an atomic publish callback."""
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite {destination}")
    staging = destination.with_name(destination.name + ".partial")
    if staging.exists():
        raise FileExistsError(f"stale staging directory: {staging}")
    staging.mkdir(parents=True)

    def publish() -> None:
        staging.rename(destination)

    return staging, publish


def discover_protected_files() -> tuple[list[Path], list[Path]]:
    truths = sorted(path for path in (ROOT / "data").glob("**/truth*.csv") if path.is_file())
    hashes = sorted(path for path in (ROOT / "data").glob("**/source_hashes.csv") if path.is_file())
    if not truths:
        raise ValueError("no pre-existing truth manifests found")
    return truths, hashes


@dataclass(frozen=True)
class Protection:
    exact: set[str]
    music_groups: set[str]
    audio_hashes: set[str]
    snapshot: pd.DataFrame


def build_protection(
    truth_paths: Iterable[Path], source_hash_paths: Iterable[Path],
) -> Protection:
    exact: set[str] = set()
    music_groups: set[str] = set()
    audio_hashes: set[str] = set()
    records: list[dict[str, object]] = []
    for kind, paths in (("truth", truth_paths), ("source_hashes", source_hash_paths)):
        for path in paths:
            resolved = path.resolve()
            try:
                relative = resolved.relative_to(ROOT.resolve())
            except ValueError as exc:
                raise ValueError(f"protected path escapes repository: {resolved}") from exc
            frame = pd.read_csv(resolved, dtype=str).fillna("")
            records.append({
                "KIND": kind, "PATH": str(relative), "ROWS": len(frame),
                "SHA256": sha256_file(resolved),
            })
            if kind == "source_hashes":
                for column in ("SHA256", "AUDIO_SHA256"):
                    if column in frame:
                        audio_hashes.update(
                            clean(value).lower() for value in frame[column] if clean(value)
                        )
                continue
            for column in EXACT_IDENTITY_COLUMNS:
                if column in frame:
                    for value in frame[column]:
                        exact.update(identity_variants(value))
            for column in MUSIC_IDENTITY_COLUMNS:
                if column in frame:
                    for value in frame[column]:
                        if clean(value):
                            music_groups.add(canonical_music_group(clean(value)))
            # SOURCE_FILE/SOURCE_ID are only music aliases on music-only rows.
            if "AUDIO_TYPE" in frame:
                is_music = frame.AUDIO_TYPE.astype(str).str.lower().eq("music")
                for column in ("SOURCE_FILE", "SOURCE_ID", "GROUP_ID"):
                    if column in frame:
                        for value in frame.loc[is_music, column]:
                            if clean(value):
                                music_groups.add(canonical_music_group(clean(value)))
            for column in ("AUDIO_SHA256", "SOURCE_SHA256"):
                if column in frame:
                    audio_hashes.update(
                        clean(value).lower() for value in frame[column] if clean(value)
                    )
    snapshot = pd.DataFrame(records).sort_values(["KIND", "PATH"]).reset_index(drop=True)
    return Protection(exact, music_groups, audio_hashes, snapshot)


def verify_snapshot(snapshot: pd.DataFrame) -> None:
    required = {"KIND", "PATH", "ROWS", "SHA256"}
    if missing := required.difference(snapshot):
        raise ValueError(f"protection snapshot lacks {sorted(missing)}")
    for row in snapshot.itertuples(index=False):
        path = (ROOT / str(row.PATH)).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"protected input disappeared: {path}")
        if sha256_file(path) != str(row.SHA256):
            raise ValueError(f"protected input changed after reservation: {path}")
        if len(pd.read_csv(path, dtype=str)) != int(row.ROWS):
            raise ValueError(f"protected input row count changed: {path}")


def echo_metadata(paths: list[Path], protection: Protection) -> tuple[dict[int, list[dict]], dict]:
    pools: dict[int, list[dict]] = {0: [], 1: []}
    counts: Counter[str] = Counter()
    required = [
        "utt_id", "label", "source", "source_text", "source_speaker_id",
        "synthesis_details",
    ]
    for parquet_path in paths:
        frame = pd.read_parquet(parquet_path, columns=required)
        shard = parquet_path.name
        for row in frame.itertuples(index=False):
            label_name = clean(row.label)
            if label_name not in ECHO_LABELS:
                counts[f"excluded_label:{label_name}"] += 1
                continue
            label = ECHO_LABELS[label_name]
            details = row.synthesis_details
            if not isinstance(details, dict):
                details = {}
            model = clean(details.get("model")) if label else "bonafide"
            if label and not model:
                counts["excluded_fake_missing_model"] += 1
                continue
            source = clean(row.source)
            speaker = clean(row.source_speaker_id)
            reference = clean(details.get("reference"))
            reference_speaker = clean(details.get("reference_speaker_id"))
            identities = {
                clean(row.utt_id), source, speaker, reference, reference_speaker,
            }
            variants = set().union(*(identity_variants(value) for value in identities))
            if not source or not speaker or (label and (not reference or not reference_speaker)):
                counts[f"excluded_incomplete:{label_name}"] += 1
                continue
            if variants & protection.exact:
                counts[f"excluded_protected:{label_name}"] += 1
                continue
            pools[label].append({
                "label": label,
                "label_name": label_name,
                "source_id": f"echofake:{row.utt_id}",
                "utt_id": clean(row.utt_id),
                "speaker": speaker,
                "content_id": source,
                "reference_id": reference,
                "reference_speaker": reference_speaker,
                "generator": model,
                "parquet": str(parquet_path.resolve()),
                "parquet_shard": shard,
                "archive_member": f"echofake_voice/{row.utt_id}.flac",
            })
            counts[f"eligible:{label_name}:{model}"] += 1
    return pools, dict(sorted(counts.items()))


def fake_music_text_status(metadata: dict[str, str] | None) -> tuple[bool, str]:
    passed, reason = musiccaps_vocal_screen(metadata)
    if passed:
        return True, MUSIC_VOCAL_SCREEN_PASS
    if reason == "excluded_no_explicit_instrumental_or_no_vocal":
        # Neutral captions are not accepted as truth yet; they proceed only to
        # the frozen PANNs+Demucs semantic gate.
        return True, PENDING_SEMANTIC
    return False, reason


def music_metadata(
    unmixed_details_path: Path, background_protocol_path: Path,
    musiccaps_metadata_path: Path, protection: Protection,
) -> tuple[dict[int, list[dict]], dict]:
    details = pd.read_csv(unmixed_details_path, dtype=str).fillna("")
    protocol = parse_mixfake_protocol(background_protocol_path)
    metadata = load_musiccaps_metadata(musiccaps_metadata_path)
    pools: dict[int, list[dict]] = {0: [], 1: []}
    counts: Counter[str] = Counter()
    selected = details.loc[
        details.split.eq("eval")
        & details.major_type.eq("Background")
        & details.sub_type.eq("Music")
    ]
    seen: set[tuple[int, str]] = set()
    for row in selected.itertuples(index=False):
        label = int(clean(row.authenticity) == "spoof")
        archive_id = Path(clean(row.file_path)).stem
        protocol_row = protocol.get(archive_id)
        expected_label = "spoof" if label else "bonafide"
        if protocol_row is None or protocol_row["label"] != expected_label:
            raise ValueError(f"missing/conflicting music protocol: {archive_id}")
        source_id, group, generator = music_identity(archive_id, clean(row.sub_dataset))
        if label:
            if clean(row.sub_dataset) == "SONICS":
                counts["excluded_sonics_suno_udio"] += 1
                continue
            if clean(row.sub_dataset) != "FakeMusicCaps" or generator not in FAKE_MUSIC_GENERATORS:
                counts[f"excluded_fake_generator:{generator}"] += 1
                continue
            accepted, status = fake_music_text_status(metadata.get(group.removeprefix("fmc:")))
            if not accepted:
                counts[status] += 1
                continue
        else:
            if clean(row.sub_dataset) != "FMA":
                counts[f"excluded_real_dataset:{row.sub_dataset}"] += 1
                continue
            status = PENDING_SEMANTIC
            generator = REAL_MUSIC_GENERATOR
        variants = identity_variants(source_id) | identity_variants(archive_id)
        if variants & protection.exact or group in protection.music_groups:
            counts[f"excluded_protected:{generator}"] += 1
            continue
        key = (label, group)
        # Retain different fake-generator renditions of a MusicCaps song.  The
        # selected reservation itself enforces canonical group uniqueness.
        if not label and key in seen:
            continue
        seen.add(key)
        pools[label].append({
            "label": label, "source_id": source_id, "group": group,
            "generator": generator, "archive_id": archive_id,
            "archive_member": protocol_row["member"], "semantic_status": status,
        })
        counts[f"eligible:{label}:{generator}:{status}"] += 1
    return pools, dict(sorted(counts.items()))


def balanced_voice_take(rows: list[dict], count: int, seed: int) -> list[dict]:
    """Round-robin generators while enforcing new target/reference speakers."""
    buckets: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        buckets[str(row["generator"])].append(row)
    for generator, items in buckets.items():
        items.sort(key=lambda row: stable_rank(seed, generator, row["utt_id"]))
    order = sorted(buckets, key=lambda value: stable_rank(seed, "generator", value))
    selected: list[dict] = []
    used_speakers: set[str] = set()
    used_content: set[str] = set()
    while len(selected) < count:
        progressed = False
        for generator in order:
            choice = None
            for row in buckets[generator]:
                speakers = {row["speaker"], row["reference_speaker"]} - {""}
                contents = {row["content_id"], row["reference_id"]} - {""}
                if speakers.isdisjoint(used_speakers) and contents.isdisjoint(used_content):
                    choice = row
                    break
            if choice is None:
                continue
            buckets[generator].remove(choice)
            selected.append(choice)
            used_speakers.update({choice["speaker"], choice["reference_speaker"]} - {""})
            used_content.update({choice["content_id"], choice["reference_id"]} - {""})
            progressed = True
            if len(selected) == count:
                break
        if not progressed:
            raise ValueError(f"need {count} speaker/content-disjoint voices, found {len(selected)}")
    return selected


def balanced_music_take(rows: list[dict], count: int, seed: int) -> list[dict]:
    buckets: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        buckets[str(row["generator"])].append(row)
    order = sorted(buckets, key=lambda value: stable_rank(seed, "generator", value))
    for generator in order:
        buckets[generator].sort(key=lambda row: (
            row["semantic_status"] != MUSIC_VOCAL_SCREEN_PASS,
            stable_rank(seed, generator, row["group"], row["source_id"]),
        ))
    selected: list[dict] = []
    used_groups: set[str] = set()
    while len(selected) < count:
        progressed = False
        for generator in order:
            choice = next((row for row in buckets[generator] if row["group"] not in used_groups), None)
            if choice is None:
                continue
            buckets[generator].remove(choice)
            selected.append(choice)
            used_groups.add(choice["group"])
            progressed = True
            if len(selected) == count:
                break
        if not progressed:
            raise ValueError(f"need {count} canonical-song-disjoint music sources, found {len(selected)}")
    return selected


def validate_reservation(frame: pd.DataFrame, per_cell: int = DEFAULT_PER_CELL,
                         require_semantic_pass: bool = False) -> None:
    expected = len(MODES) * len(CELLS) * per_cell
    if len(frame) != expected or frame.BASE_ID.nunique() != expected:
        raise ValueError(f"expected {expected} unique bases, got {len(frame)}")
    cell_counts = frame.groupby(["MIX_MODE", "VOICE_FAKE", "MUSIC_FAKE"]).size()
    if len(cell_counts) != 12 or set(cell_counts.astype(int)) != {per_cell}:
        raise ValueError(f"unbalanced layout/case cells: {cell_counts.to_dict()}")
    for column in ("VOICE_SOURCE_ID", "MUSIC_SOURCE_ID", "MUSIC_GROUP_ID"):
        if frame[column].duplicated().any():
            raise ValueError(f"within-bank source reuse: {column}")
    if frame.VOICE_SPEAKER.duplicated().any():
        raise ValueError("within-bank target-speaker reuse")
    references = set(frame.VOICE_REFERENCE_SPEAKER) - {""}
    targets = set(frame.VOICE_SPEAKER)
    if len(references) != int(frame.VOICE_REFERENCE_SPEAKER.ne("").sum()):
        raise ValueError("within-bank reference-speaker reuse")
    if references & targets:
        raise ValueError("target/reference speaker crossover")
    file_truth = frame[["VOICE_FAKE", "MUSIC_FAKE"]].astype(int).max(axis=1)
    if not np.array_equal(file_truth, frame.FILE_FAKE.astype(int)):
        raise ValueError("FILE_FAKE is not component OR")
    fake_music = frame.loc[frame.MUSIC_FAKE.astype(int).eq(1)]
    counts = fake_music.MUSIC_GENERATOR.value_counts()
    if set(counts.index) != set(FAKE_MUSIC_GENERATORS) or counts.max() - counts.min() > 1:
        raise ValueError(f"fake-music generator imbalance: {counts.to_dict()}")
    if frame.VOICE_FAKE.astype(int).eq(1).sum() != frame.VOICE_FAKE.astype(int).eq(0).sum():
        raise ValueError("voice labels are not balanced")
    fake_voice_counts = frame.loc[frame.VOICE_FAKE.astype(int).eq(1), "VOICE_GENERATOR"].value_counts()
    if len(fake_voice_counts) < 8 or fake_voice_counts.max() - fake_voice_counts.min() > 1:
        raise ValueError(f"fake-voice generator diversity/imbalance failure: {fake_voice_counts.to_dict()}")
    if frame.MUSIC_GENERATOR.astype(str).str.lower().isin({"suno", "udio"}).any():
        raise ValueError("vocal SONICS/Suno/Udio music entered the reservation")
    if require_semantic_pass and set(frame.MUSIC_VOCAL_SCREEN) != {SEMANTIC_PASS}:
        raise ValueError("not every music source passed the frozen semantic gate")


def write_source_requirements(directory: Path, reservation: pd.DataFrame) -> None:
    music = reservation[[
        "MUSIC_SOURCE_ID", "MUSIC_GROUP_ID", "MUSIC_GENERATOR",
        "MUSIC_ARCHIVE_ID", "MUSIC_ARCHIVE_MEMBER",
    ]].copy()
    music.columns = ["SOURCE_ID", "CANONICAL_GROUP", "GENERATOR", "ARCHIVE_ID", "ARCHIVE_MEMBER"]
    music["KIND"] = "music"
    music["ARCHIVE_TARGET"] = "MixFake/" + music.ARCHIVE_MEMBER.astype(str)
    music.sort_values("ARCHIVE_TARGET").to_csv(directory / "source_requirements.csv", index=False)
    (directory / "archive_member_targets.txt").write_text(
        "\n".join(music.ARCHIVE_TARGET.astype(str)) + "\n", "utf-8"
    )


def write_reservation(directory: Path, frame: pd.DataFrame, provenance: dict,
                      candidate_pool: pd.DataFrame, snapshot: pd.DataFrame) -> None:
    staging, publish = atomic_output_dir(directory)
    try:
        frame.to_csv(staging / "reservation.csv", index=False)
        candidate_pool.to_csv(staging / "candidate_pool.csv", index=False)
        snapshot.to_csv(staging / "protected_inputs.csv", index=False)
        write_source_requirements(staging, frame)
        payload = dict(provenance)
        payload.update({
            "reservation_sha256": sha256_file(staging / "reservation.csv"),
            "candidate_pool_sha256": sha256_file(staging / "candidate_pool.csv"),
            "protected_inputs_sha256": sha256_file(staging / "protected_inputs.csv"),
            "builder_sha256": sha256_file(Path(__file__)),
            "authenticity_detector_inference": False,
            "authenticity_score_computed": False,
        })
        (staging / "provenance.json").write_text(
            json.dumps(payload, indent=2, ensure_ascii=False) + "\n", "utf-8"
        )
        publish()
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def reserve(args: argparse.Namespace) -> None:
    truth_paths, source_hash_paths = discover_protected_files()
    protection = build_protection(truth_paths, source_hash_paths)
    voices, voice_counts = echo_metadata(args.echo_parquet, protection)
    music, music_counts = music_metadata(
        args.unmixed_details, args.background_protocol, args.musiccaps_metadata,
        protection,
    )
    needed = len(MODES) * len(CELLS) * args.per_cell // 2
    voice_selected = {
        label: balanced_voice_take(voices[label], needed, args.seed + 10 + label)
        for label in (0, 1)
    }
    # Ensure real and fake selections do not cross speaker/content identities.
    real_voice_identities = set().union(*(
        {row["speaker"], row["reference_speaker"], row["content_id"], row["reference_id"]} - {""}
        for row in voice_selected[0]
    ))
    fake_candidates = [
        row for row in voices[1]
        if ({row["speaker"], row["reference_speaker"], row["content_id"], row["reference_id"]} - {""}).isdisjoint(real_voice_identities)
    ]
    voice_selected[1] = balanced_voice_take(fake_candidates, needed, args.seed + 11)
    music_selected = {
        label: balanced_music_take(music[label], needed, args.seed + 20 + label)
        for label in (0, 1)
    }
    offsets_voice = {0: 0, 1: 0}
    offsets_music = {0: 0, 1: 0}
    records: list[dict[str, object]] = []
    index = 0
    for mode in MODES:
        for voice_fake, music_fake in CELLS:
            for repeat in range(args.per_cell):
                voice = voice_selected[voice_fake][offsets_voice[voice_fake]]
                music_item = music_selected[music_fake][offsets_music[music_fake]]
                offsets_voice[voice_fake] += 1
                offsets_music[music_fake] += 1
                key = f"{args.seed}|{mode}|{voice_fake}|{music_fake}|{repeat}"
                snr = (-10, -5, 0, 5, 10)[int(stable_rank(args.seed, key, "snr")[:8], 16) % 5]
                overlap = (0.25, 0.50, 0.75)[int(stable_rank(args.seed, key, "overlap")[:8], 16) % 3]
                order = "voice_first" if int(stable_rank(args.seed, key, "order")[:8], 16) % 2 == 0 else "music_first"
                gap = (0.0, 0.2, 0.5)[int(stable_rank(args.seed, key, "gap")[:8], 16) % 3]
                cell = f"{'F' if voice_fake else 'R'}{'F' if music_fake else 'R'}"
                records.append({
                    "BASE_ID": f"{ID_PREFIX}_{index:04d}",
                    "FILE_FAKE": int(voice_fake or music_fake),
                    "VOICE_FAKE": voice_fake, "MUSIC_FAKE": music_fake,
                    "VOICE_PRESENT": 1, "MUSIC_PRESENT": 1,
                    "AUDIO_TYPE": "mixed", "MIX_MODE": mode,
                    "COMPONENT_CASE": cell, "EVAL_CELL": f"{mode}__{cell}",
                    "VOICE_SOURCE_ID": voice["source_id"],
                    "VOICE_SPEAKER": voice["speaker"],
                    "VOICE_CONTENT_ID": voice["content_id"],
                    "VOICE_REFERENCE_ID": voice["reference_id"],
                    "VOICE_REFERENCE_SPEAKER": voice["reference_speaker"],
                    "VOICE_GENERATOR": voice["generator"],
                    "VOICE_ARCHIVE_ID": voice["utt_id"],
                    "VOICE_ARCHIVE_MEMBER": voice["archive_member"],
                    "VOICE_PARQUET_SHARD": voice["parquet_shard"],
                    "MUSIC_SOURCE_ID": music_item["source_id"],
                    "MUSIC_GROUP_ID": music_item["group"],
                    "MUSIC_GENERATOR": music_item["generator"],
                    "MUSIC_VOCAL_SCREEN": music_item["semantic_status"],
                    "MUSIC_ARCHIVE_ID": music_item["archive_id"],
                    "MUSIC_ARCHIVE_MEMBER": music_item["archive_member"],
                    "SNR_DB": snr if mode != "sequential" else "",
                    "OVERLAP_FRACTION": overlap if mode == "partial_overlap" else (1.0 if mode == "concurrent" else 0.0),
                    "ORDER": order,
                    "GAP_SECONDS": gap if mode == "sequential" else 0.0,
                    "SOURCE_DATASET": "EchoFake+MixFake_official_eval",
                    "PROVENANCE_MODE": "all_existing_truth_source_speaker_song_disjoint",
                })
                index += 1
    frame = pd.DataFrame(records)
    validate_reservation(frame, args.per_cell, require_semantic_pass=False)
    candidate_rows: list[dict[str, object]] = []
    for label in (0, 1):
        candidate_rows.extend({"KIND": "voice", **row} for row in voices[label])
        candidate_rows.extend({"KIND": "music", **row} for row in music[label])
    candidate_pool = pd.DataFrame(candidate_rows).fillna("")
    input_paths = [
        *args.echo_parquet, args.unmixed_details, args.background_protocol,
        args.musiccaps_metadata,
    ]
    provenance = {
        "schema_version": "codec_mixed_blind_v7",
        "stage": "reserved_pending_semantic_screen",
        "seed": args.seed, "id_prefix": ID_PREFIX, "per_cell": args.per_cell,
        "base_rows": len(frame), "rendered_rows_planned": len(frame) * len(CHANNELS),
        "channels": list(CHANNELS), "provenance_mode": "strict_v7",
        "inputs": {
            path.name: {"path": str(path.resolve()), "sha256": sha256_file(path)}
            for path in input_paths
        },
        "protection": {
            "truth_manifests": len(truth_paths),
            "source_hash_manifests": len(source_hash_paths),
            "exact_identity_variants": len(protection.exact),
            "canonical_music_groups": len(protection.music_groups),
            "known_source_audio_hashes": len(protection.audio_hashes),
            "scope": "every pre-existing data/**/truth*.csv and source_hashes.csv",
        },
        "voice_candidate_counts": voice_counts,
        "music_candidate_counts": music_counts,
        "selection": {
            "requested_fake_music_generators": list(FAKE_MUSIC_GENERATORS),
            "fake_voice_generators": sorted(set(frame.loc[frame.VOICE_FAKE.eq(1), "VOICE_GENERATOR"])),
            "speaker_reuse": False, "canonical_music_group_reuse": False,
            "suno_udio_selected": 0,
        },
        "semantic_policy": {
            "positive_vocal_text_rejection": True,
            "every_selected_music_requires_frozen_panns_demucs_pass": True,
            "pending_value": PENDING_SEMANTIC, "pass_value": SEMANTIC_PASS,
        },
    }
    write_reservation(args.output_dir, frame, provenance, candidate_pool, protection.snapshot)
    print(json.dumps({
        "status": "reserved", "base_rows": len(frame), "rendered_rows": len(frame) * len(CHANNELS),
        "fake_voice_generators": frame.loc[frame.VOICE_FAKE.eq(1), "VOICE_GENERATOR"].value_counts().to_dict(),
        "fake_music_generators": frame.loc[frame.MUSIC_FAKE.eq(1), "MUSIC_GENERATOR"].value_counts().to_dict(),
        "protected_truths": len(truth_paths), "authenticity_scored": False,
    }, indent=2))


def load_reservation_dir(directory: Path) -> tuple[pd.DataFrame, dict, pd.DataFrame, pd.DataFrame]:
    paths = {
        "reservation": directory / "reservation.csv",
        "provenance": directory / "provenance.json",
        "candidates": directory / "candidate_pool.csv",
        "snapshot": directory / "protected_inputs.csv",
    }
    for path in paths.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    frame = pd.read_csv(paths["reservation"], dtype=str).fillna("")
    provenance = json.loads(paths["provenance"].read_text("utf-8"))
    candidates = pd.read_csv(paths["candidates"], dtype=str).fillna("")
    snapshot = pd.read_csv(paths["snapshot"], dtype=str).fillna("")
    for key, filename in (("reservation_sha256", "reservation.csv"),
                          ("candidate_pool_sha256", "candidate_pool.csv"),
                          ("protected_inputs_sha256", "protected_inputs.csv")):
        if sha256_file(directory / filename) != provenance[key]:
            raise ValueError(f"{filename} hash mismatch")
    return frame, provenance, candidates, snapshot


def materialize_voices(args: argparse.Namespace) -> None:
    reservation, provenance, _, snapshot = load_reservation_dir(args.reservation_dir)
    verify_snapshot(snapshot)
    wanted = set(reservation.VOICE_ARCHIVE_ID)
    if len(wanted) != len(reservation):
        raise ValueError("voice archive IDs are reused")
    staging, publish = atomic_output_dir(args.output_dir)
    try:
        found: set[str] = set()
        records: list[dict[str, object]] = []
        voice_dir = staging / "echofake_voice"
        voice_dir.mkdir()
        for parquet_path in args.echo_parquet:
            table = pd.read_parquet(parquet_path, columns=["utt_id", "path"])
            selected = table.loc[table.utt_id.astype(str).isin(wanted - found)]
            for row in selected.itertuples(index=False):
                payload = row.path
                if not isinstance(payload, dict) or not payload.get("bytes"):
                    raise ValueError(f"missing embedded audio bytes for {row.utt_id}")
                audio, rate = sf.read(BytesIO(payload["bytes"]), dtype="float32", always_2d=True)
                mono = np.nan_to_num(audio.mean(axis=1).astype(np.float32))
                if rate != SR:
                    divisor = math.gcd(int(rate), SR)
                    mono = resample_poly(mono, SR // divisor, int(rate) // divisor).astype(np.float32)
                if not mono.size or len(mono) > 60 * SR:
                    raise ValueError(f"invalid EchoFake duration: {row.utt_id} {len(mono) / SR:.3f}s")
                destination = voice_dir / f"{row.utt_id}.flac"
                sf.write(destination, mono, SR, format="FLAC", subtype="PCM_16")
                found.add(str(row.utt_id))
                records.append({
                    "VOICE_ARCHIVE_ID": row.utt_id,
                    "SOURCE_PARQUET": str(parquet_path.resolve()),
                    "SOURCE_CONTAINER_NAME": clean(payload.get("path")),
                    "DURATION": len(mono) / SR, "BYTES": destination.stat().st_size,
                    "SHA256": sha256_file(destination),
                })
        if found != wanted:
            raise ValueError(f"missing EchoFake rows: {sorted(wanted - found)[:5]}")
        pd.DataFrame(records).sort_values("VOICE_ARCHIVE_ID").to_csv(
            staging / "voice_manifest.csv", index=False
        )
        (staging / "provenance.json").write_text(json.dumps({
            "stage": "voice_sources_materialized", "reservation_sha256": provenance["reservation_sha256"],
            "voice_rows": len(records), "voice_manifest_sha256": sha256_file(staging / "voice_manifest.csv"),
            "decoder": "soundfile_to_16k_mono_pcm16_flac", "authenticity_detector_inference": False,
            "authenticity_score_computed": False,
        }, indent=2) + "\n", "utf-8")
        publish()
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    print(json.dumps({"status": "materialized", "voice_sources": len(wanted)}, indent=2))


def extract_music(args: argparse.Namespace) -> None:
    """Extract only frozen reservation members from the split MixFake 7z."""
    import multivolumefile
    import py7zr

    reservation, provenance, _, snapshot = load_reservation_dir(args.reservation_dir)
    verify_snapshot(snapshot)
    if not args.source_root.is_dir():
        raise FileNotFoundError(args.source_root)
    destination = args.source_root / "MixFake"
    volume_paths = sorted(args.archive_base.parent.glob(args.archive_base.name + ".*"))
    if not volume_paths:
        raise FileNotFoundError(f"no split archive volumes for {args.archive_base}")
    targets = sorted({"MixFake/" + str(value).replace("\\", "/").lstrip("/")
                      for value in reservation.MUSIC_ARCHIVE_MEMBER})
    if any(".." in Path(value).parts for value in targets):
        raise ValueError("unsafe archive target")
    existing = {target for target in targets if (args.source_root / target).is_file()}
    targets_to_extract = sorted(set(targets) - existing)
    if not targets_to_extract:
        print(json.dumps({
            "stage": "music_sources_already_extracted",
            "reservation_sha256": provenance["reservation_sha256"],
            "targets": len(targets), "new_targets": 0,
            "authenticity_detector_inference": False,
            "authenticity_score_computed": False,
        }, indent=2))
        return
    staging = args.source_root / ".mixfake_extract.partial"
    if staging.exists():
        raise FileExistsError(f"stale extraction staging directory: {staging}")
    staging.mkdir()
    try:
        with multivolumefile.open(args.archive_base, "rb") as multivolume:
            with py7zr.SevenZipFile(multivolume, "r") as archive:
                names = set(archive.getnames())
                missing = sorted(set(targets_to_extract) - names)
                if missing:
                    raise ValueError(f"reserved archive targets missing: {missing[:5]}")
                archive.extract(path=staging, targets=targets_to_extract)
        extracted = staging / "MixFake"
        actual = sorted(path.relative_to(staging).as_posix() for path in extracted.rglob("*") if path.is_file())
        if actual != targets_to_extract:
            raise ValueError(f"extracted target mismatch: expected={len(targets_to_extract)} actual={len(actual)}")
        if not destination.exists():
            extracted.rename(destination)
            staging.rmdir()
        else:
            for target in targets_to_extract:
                relative = Path(target).relative_to("MixFake")
                source = staging / target
                output = destination / relative
                if output.exists():
                    raise FileExistsError(f"refusing to overwrite extracted member: {output}")
                output.parent.mkdir(parents=True, exist_ok=True)
                source.rename(output)
            shutil.rmtree(staging)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    report = {
        "stage": "music_sources_extracted", "reservation_sha256": provenance["reservation_sha256"],
        "archive_base": str(args.archive_base.resolve()), "archive_volumes": len(volume_paths),
        "archive_total_bytes": sum(path.stat().st_size for path in volume_paths),
        "targets": len(targets), "new_targets": len(targets_to_extract),
        "preexisting_targets": len(existing), "authenticity_detector_inference": False,
        "authenticity_score_computed": False,
    }
    report_path = args.source_root / f"music_extraction_{provenance['reservation_sha256'][:12]}.json"
    if report_path.exists():
        raise FileExistsError(f"refusing to overwrite extraction report: {report_path}")
    report_path.write_text(json.dumps(report, indent=2) + "\n", "utf-8")
    print(json.dumps(report, indent=2))


def locate_source(roots: list[Path], member: str) -> Path:
    relative = Path(str(member).replace("\\", "/"))
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"unsafe member path: {member}")
    matches: set[Path] = set()
    for root in roots:
        for candidate in (root / relative, root / relative.name,
                          root / Path(*relative.parts[1:]) if relative.parts and relative.parts[0] == "unmixed_dataset" else root / relative):
            if candidate.is_file():
                matches.add(candidate.resolve())
    if len(matches) != 1:
        raise FileNotFoundError(f"expected exactly one {member} across roots; got {sorted(map(str, matches))}")
    return next(iter(matches))


def semantic_screen(args: argparse.Namespace) -> None:
    # Semantic presence models only; no authenticity model is imported here.
    from audit_prospective_music_vocals import (
        IndependentPannsVocalScreen, demucs_energy, sha256_file as audit_sha256,
    )
    from separation import HTDemucsSeparator

    reservation, provenance, _, snapshot = load_reservation_dir(args.reservation_dir)
    verify_snapshot(snapshot)
    selected = reservation.copy()
    if args.only_source_id:
        requested = set(args.only_source_id)
        selected = selected.loc[selected.MUSIC_SOURCE_ID.isin(requested)]
        if set(selected.MUSIC_SOURCE_ID) != requested:
            raise ValueError("semantic source filter is not contained in reservation")
    selected = selected.drop_duplicates("MUSIC_SOURCE_ID")
    staging, publish = atomic_output_dir(args.output_dir)
    try:
        panns = IndependentPannsVocalScreen(args.panns_dir, args.device)
        separator = HTDemucsSeparator(device=args.device, repo=args.demucs_repo, shifts=0, overlap=0.25)
        records: list[dict[str, object]] = []
        for index, row in enumerate(selected.itertuples(index=False), 1):
            path = locate_source(args.source_root, row.MUSIC_ARCHIVE_MEMBER)
            audio = load_audio(path)
            scores = panns.score(audio)
            scores.update(demucs_energy(separator, path))
            panns_flag = scores["PANNS_ANY_VOICE_MAX"] > args.panns_threshold
            demucs_flag = scores["DEMUCS_VOCAL_TO_MIX_DB"] > args.demucs_threshold_db
            records.append({
                "MUSIC_SOURCE_ID": row.MUSIC_SOURCE_ID,
                "MUSIC_GROUP_ID": row.MUSIC_GROUP_ID,
                "MUSIC_GENERATOR": row.MUSIC_GENERATOR,
                "MUSIC_LABEL": row.MUSIC_FAKE,
                "MUSIC_ARCHIVE_MEMBER": row.MUSIC_ARCHIVE_MEMBER,
                "AUDIO_SHA256": audit_sha256(path), **scores,
                "PANNS_THRESHOLD": args.panns_threshold, "PANNS_FLAG": panns_flag,
                "DEMUCS_THRESHOLD_DB": args.demucs_threshold_db, "DEMUCS_FLAG": demucs_flag,
                "ACOUSTIC_SCREEN_PASS": not (panns_flag or demucs_flag),
            })
            if index % 10 == 0:
                print(f"semantic screen {index}/{len(selected)}", flush=True)
        result = pd.DataFrame(records)
        result.to_csv(staging / "source_scores.csv", index=False)
        (staging / "summary.json").write_text(json.dumps({
            "purpose": "instrumental semantic screen only; not authenticity evaluation",
            "reservation_sha256": provenance["reservation_sha256"],
            "sources": len(result), "passes": int(bool_series(result.ACOUSTIC_SCREEN_PASS).sum()),
            "failures": int((~bool_series(result.ACOUSTIC_SCREEN_PASS)).sum()),
            "panns_threshold": args.panns_threshold,
            "demucs_threshold_db": args.demucs_threshold_db,
            "authenticity_detector_inference": False, "authenticity_score_computed": False,
        }, indent=2) + "\n", "utf-8")
        publish()
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def merge_semantic_scores(paths: list[Path]) -> pd.DataFrame:
    frames = []
    for path in paths:
        frame = pd.read_csv(path, dtype=str).fillna("")
        if FORBIDDEN_PREDICTION_COLUMNS & set(frame):
            raise ValueError(f"authenticity/scoring columns forbidden in semantic input: {path}")
        required = {"MUSIC_SOURCE_ID", "ACOUSTIC_SCREEN_PASS", "PANNS_THRESHOLD", "DEMUCS_THRESHOLD_DB"}
        if missing := required.difference(frame):
            raise ValueError(f"semantic scores lack {sorted(missing)}: {path}")
        frames.append(frame)
    result = pd.concat(frames, ignore_index=True)
    if result.MUSIC_SOURCE_ID.duplicated().any():
        duplicates = result.loc[result.MUSIC_SOURCE_ID.duplicated(False), "MUSIC_SOURCE_ID"].unique()
        raise ValueError(f"duplicate semantic decisions: {duplicates[:5]}")
    if set(result.PANNS_THRESHOLD.astype(float)) != {0.20} or set(result.DEMUCS_THRESHOLD_DB.astype(float)) != {-1.5}:
        raise ValueError("semantic thresholds differ from frozen 0.20/-1.5 policy")
    return result


def revise(args: argparse.Namespace) -> None:
    reservation, provenance, candidates, snapshot = load_reservation_dir(args.reservation_dir)
    verify_snapshot(snapshot)
    scores = merge_semantic_scores(args.semantic_scores)
    score_pass = dict(zip(scores.MUSIC_SOURCE_ID, bool_series(scores.ACOUSTIC_SCREEN_PASS)))
    missing = set(reservation.MUSIC_SOURCE_ID) - set(score_pass)
    if missing:
        raise ValueError(f"current music sources lack semantic decisions: {sorted(missing)[:5]}")
    failed = [source for source in reservation.MUSIC_SOURCE_ID if not score_pass[source]]
    selected_ids = set(reservation.MUSIC_SOURCE_ID)
    selected_groups = set(reservation.MUSIC_GROUP_ID)
    replacements: list[dict[str, str]] = []
    result = reservation.copy()
    music_candidates = candidates.loc[candidates.KIND.eq("music")].copy()
    for source_id in failed:
        row_index = result.index[result.MUSIC_SOURCE_ID.eq(source_id)]
        if len(row_index) != 1:
            raise ValueError(f"failed source does not map to one base: {source_id}")
        index = int(row_index[0])
        label = str(result.at[index, "MUSIC_FAKE"])
        generator = str(result.at[index, "MUSIC_GENERATOR"])
        old_group = str(result.at[index, "MUSIC_GROUP_ID"])
        available = music_candidates.loc[
            music_candidates.label.astype(str).eq(label)
            & music_candidates.generator.eq(generator)
            & ~music_candidates.source_id.isin(selected_ids | set(score_pass))
            & ~music_candidates.group.isin(selected_groups)
        ].copy()
        if available.empty:
            raise ValueError(f"no unseen same-label/generator replacement for {source_id}")
        available["_rank"] = [stable_rank(int(provenance["seed"]), "semantic-replacement", generator, value)
                              for value in available.source_id]
        choice = available.sort_values("_rank").iloc[0]
        selected_ids.add(str(choice.source_id))
        selected_groups.remove(old_group)
        selected_groups.add(str(choice.group))
        mapping = {
            "MUSIC_SOURCE_ID": "source_id", "MUSIC_GROUP_ID": "group",
            "MUSIC_GENERATOR": "generator", "MUSIC_VOCAL_SCREEN": "semantic_status",
            "MUSIC_ARCHIVE_ID": "archive_id", "MUSIC_ARCHIVE_MEMBER": "archive_member",
        }
        for column, source_column in mapping.items():
            result.at[index, column] = choice[source_column]
        replacements.append({
            "excluded_source_id": source_id, "excluded_group": old_group,
            "replacement_source_id": str(choice.source_id),
            "replacement_group": str(choice.group), "generator": generator,
        })
    if not failed:
        result["MUSIC_VOCAL_SCREEN"] = SEMANTIC_PASS
    validate_reservation(result, int(provenance["per_cell"]), require_semantic_pass=not failed)
    history = list(provenance.get("semantic_replacement_history", [])) + replacements
    new_provenance = dict(provenance)
    new_provenance.update({
        "stage": "reserved_pending_semantic_screen" if failed else "source_semantics_validated",
        "parent_reservation_sha256": provenance["reservation_sha256"],
        "semantic_score_inputs": [{"path": str(path.resolve()), "sha256": sha256_file(path)} for path in args.semantic_scores],
        "semantic_replacement_history": history,
        "semantic_failures_this_revision": len(failed),
    })
    write_reservation(args.output_dir, result, new_provenance, candidates, snapshot)
    print(json.dumps({
        "status": new_provenance["stage"], "failures": len(failed),
        "replacements": replacements,
    }, indent=2))


def protected_from_snapshot(snapshot: pd.DataFrame) -> Protection:
    verify_snapshot(snapshot)
    truths = [ROOT / path for path in snapshot.loc[snapshot.KIND.eq("truth"), "PATH"]]
    hashes = [ROOT / path for path in snapshot.loc[snapshot.KIND.eq("source_hashes"), "PATH"]]
    return build_protection(truths, hashes)


def validate_sources(args: argparse.Namespace) -> None:
    reservation, provenance, _, snapshot = load_reservation_dir(args.reservation_dir)
    protection = protected_from_snapshot(snapshot)
    validate_reservation(reservation, int(provenance["per_cell"]), require_semantic_pass=True)
    candidate_values: set[str] = set()
    for column in (
        "VOICE_SOURCE_ID", "VOICE_SPEAKER", "VOICE_CONTENT_ID", "VOICE_REFERENCE_ID",
        "VOICE_REFERENCE_SPEAKER", "VOICE_ARCHIVE_ID", "MUSIC_SOURCE_ID", "MUSIC_ARCHIVE_ID",
    ):
        for value in reservation[column]:
            candidate_values.update(identity_variants(value))
    identity_overlap = sorted(candidate_values & protection.exact)
    music_overlap = sorted(set(reservation.MUSIC_GROUP_ID) & protection.music_groups)
    records: list[dict[str, object]] = []
    source_hash_overlap: list[str] = []
    for row in reservation.itertuples(index=False):
        for kind, source_id, member in (
            ("voice", row.VOICE_SOURCE_ID, row.VOICE_ARCHIVE_MEMBER),
            ("music", row.MUSIC_SOURCE_ID, row.MUSIC_ARCHIVE_MEMBER),
        ):
            path = locate_source(args.source_root, member)
            digest = sha256_file(path)
            if digest.lower() in protection.audio_hashes:
                source_hash_overlap.append(f"{kind}:{source_id}")
            records.append({
                "KIND": kind, "SOURCE_ID": source_id, "ARCHIVE_MEMBER": member,
                "LOCAL_PATH": str(path), "BYTES": path.stat().st_size, "SHA256": digest,
            })
    source_hashes = pd.DataFrame(records).drop_duplicates(["KIND", "SOURCE_ID"])
    checks = {
        "protected_snapshot_unchanged": True,
        "reservation_source_identity_disjoint": not identity_overlap,
        "reservation_canonical_music_disjoint": not music_overlap,
        "source_audio_hash_disjoint": not source_hash_overlap,
        "all_sources_resolve_once": len(source_hashes) == len(reservation) * 2,
        "all_music_semantic_pass": set(reservation.MUSIC_VOCAL_SCREEN) == {SEMANTIC_PASS},
        "suno_udio_sources_zero": not reservation.MUSIC_GENERATOR.str.lower().isin({"suno", "udio"}).any(),
        "authenticity_inference_absent": provenance.get("authenticity_detector_inference") is False,
        "authenticity_scoring_absent": provenance.get("authenticity_score_computed") is False,
    }
    staging, publish = atomic_output_dir(args.output_dir)
    try:
        source_hashes.to_csv(staging / "source_hashes.csv", index=False)
        report = {
            "status": "PASS" if all(checks.values()) else "FAIL",
            "checks": {key: bool(value) for key, value in checks.items()},
            "identity_overlap": identity_overlap[:20], "music_group_overlap": music_overlap[:20],
            "source_hash_overlap": source_hash_overlap[:20],
            "reservation_sha256": provenance["reservation_sha256"],
            "source_hashes_sha256": sha256_file(staging / "source_hashes.csv"),
            "protected_truth_manifests": int(snapshot.KIND.eq("truth").sum()),
            "protected_source_hash_manifests": int(snapshot.KIND.eq("source_hashes").sum()),
            "authenticity_detector_scores_read": False,
        }
        (staging / "validation.json").write_text(json.dumps(report, indent=2) + "\n", "utf-8")
        if report["status"] != "PASS":
            raise RuntimeError(f"source validation failed: {[key for key, value in checks.items() if not value]}")
        publish()
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    print(json.dumps(report, indent=2))


def render(args: argparse.Namespace) -> None:
    reservation, provenance, _, snapshot = load_reservation_dir(args.reservation_dir)
    verify_snapshot(snapshot)
    validate_reservation(reservation, int(provenance["per_cell"]), require_semantic_pass=True)
    validation_path = args.source_validation / "validation.json"
    if not validation_path.is_file():
        raise FileNotFoundError(validation_path)
    source_validation = json.loads(validation_path.read_text("utf-8"))
    if source_validation.get("status") != "PASS" or source_validation.get("reservation_sha256") != provenance["reservation_sha256"]:
        raise ValueError("source validation is not PASS for this exact reservation")
    if shutil.which(str(args.ffmpeg)) is None and not args.ffmpeg.is_file():
        raise FileNotFoundError(args.ffmpeg)
    staging, publish = atomic_output_dir(args.output_dir)
    try:
        audio_dir = staging / "audio"
        audio_dir.mkdir()
        truths: list[dict[str, object]] = []
        hashes: list[dict[str, object]] = []
        for base_index, row in enumerate(reservation.itertuples(index=False), 1):
            voice_path = locate_source(args.source_root, row.VOICE_ARCHIVE_MEMBER)
            music_path = locate_source(args.source_root, row.MUSIC_ARCHIVE_MEMBER)
            mixed = mix_layout(load_audio(voice_path), load_audio(music_path), pd.Series(row._asdict()))
            if not 4 * SR <= len(mixed) <= 60 * SR:
                raise ValueError(f"rendered base duration out of range: {row.BASE_ID}")
            for channel in CHANNELS:
                key = int(stable_rank(0, row.BASE_ID, channel)[:16], 16) % (2**32)
                rendered = apply_channel(mixed, channel, ffmpeg=None if channel == "clean" else args.ffmpeg, key=key)
                sample_id = f"{row.BASE_ID}__{channel}"
                destination = audio_dir / f"{sample_id}.flac"
                sf.write(destination, rendered, SR, format="FLAC", subtype="PCM_16")
                hashes.append({"ID": sample_id, "BYTES": destination.stat().st_size, "SHA256": sha256_file(destination)})
                record = row._asdict()
                record.update({"ID": sample_id, "PARENT_ID": row.BASE_ID, "CHANNEL": channel, "DURATION": len(rendered) / SR})
                truths.append(record)
            if base_index % 10 == 0:
                print(f"render {base_index}/{len(reservation)} bases", flush=True)
        truth = pd.DataFrame(truths)
        expected = len(reservation) * len(CHANNELS)
        if len(truth) != expected or truth.ID.nunique() != expected:
            raise AssertionError("render row/ID count mismatch")
        truth.to_csv(staging / "truth.csv", index=False)
        sample = pd.DataFrame({"ID": truth.ID})
        for column in PREDICTION_COLUMNS:
            sample[column] = 0.5
        sample.to_csv(staging / "sample_submission.csv", index=False)
        pd.DataFrame(hashes).to_csv(staging / "audio_hashes.csv", index=False)
        shutil.copy2(args.source_validation / "source_hashes.csv", staging / "source_hashes.csv")
        bank_provenance = dict(provenance)
        bank_provenance.update({
            "stage": "built_unscored", "rendered_rows": len(truth),
            "truth_sha256": sha256_file(staging / "truth.csv"),
            "audio_hashes_sha256": sha256_file(staging / "audio_hashes.csv"),
            "source_hashes_sha256": sha256_file(staging / "source_hashes.csv"),
            "source_validation_sha256": sha256_file(validation_path),
            "ffmpeg": subprocess.run([str(args.ffmpeg), "-version"], check=True, capture_output=True, text=True).stdout.splitlines()[0],
            "authenticity_detector_inference": False, "authenticity_score_computed": False,
        })
        (staging / "provenance.json").write_text(json.dumps(bank_provenance, indent=2) + "\n", "utf-8")
        publish()
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    print(json.dumps({"status": "built_unscored", "rows": expected, "truth_sha256": bank_provenance["truth_sha256"]}, indent=2))


def validate_bank(args: argparse.Namespace) -> None:
    reservation, reservation_provenance, _, snapshot = load_reservation_dir(args.reservation_dir)
    verify_snapshot(snapshot)
    validate_reservation(
        reservation, int(reservation_provenance["per_cell"]), require_semantic_pass=True,
    )
    required_paths = {
        "truth": args.bank_dir / "truth.csv",
        "sample": args.bank_dir / "sample_submission.csv",
        "audio_hashes": args.bank_dir / "audio_hashes.csv",
        "source_hashes": args.bank_dir / "source_hashes.csv",
        "provenance": args.bank_dir / "provenance.json",
        "source_validation": args.source_validation / "validation.json",
    }
    for path in required_paths.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    truth = pd.read_csv(required_paths["truth"], dtype=str).fillna("")
    bank_provenance = json.loads(required_paths["provenance"].read_text("utf-8"))
    source_validation = json.loads(required_paths["source_validation"].read_text("utf-8"))
    audio_paths = sorted((args.bank_dir / "audio").glob("*.flac"))
    audio_hashes = pd.read_csv(required_paths["audio_hashes"], dtype=str).fillna("")
    recorded_hashes = dict(zip(audio_hashes.ID, audio_hashes.SHA256))
    sample = pd.read_csv(required_paths["sample"], dtype=str).fillna("")
    expected_rows = len(reservation) * len(CHANNELS)
    properties = [sf.info(path) for path in audio_paths]
    recipe = truth.merge(
        reservation, on="BASE_ID", how="left", suffixes=("_truth", "_reserved"),
        validate="many_to_one",
    )
    recipe_columns = [column for column in reservation.columns if column != "BASE_ID"]
    recipe_equal = True
    for column in recipe_columns:
        left = recipe[f"{column}_truth"].astype(str)
        right = recipe[f"{column}_reserved"].astype(str)
        if not left.equals(right):
            recipe_equal = False
            break
    checks = {
        "bank_provenance_unscored": (
            bank_provenance.get("authenticity_detector_inference") is False
            and bank_provenance.get("authenticity_score_computed") is False
        ),
        "source_validation_pass": source_validation.get("status") == "PASS",
        "reservation_hash_carried": (
            bank_provenance.get("reservation_sha256") == reservation_provenance["reservation_sha256"]
            == source_validation.get("reservation_sha256")
        ),
        "truth_hash": sha256_file(required_paths["truth"]) == bank_provenance.get("truth_sha256"),
        "source_hashes_hash": sha256_file(required_paths["source_hashes"]) == bank_provenance.get("source_hashes_sha256"),
        "audio_hash_manifest_hash": sha256_file(required_paths["audio_hashes"]) == bank_provenance.get("audio_hashes_sha256"),
        "rendered_row_count": len(truth) == expected_rows,
        "unique_rendered_ids": truth.ID.nunique() == expected_rows,
        "five_paired_channels": (
            set(truth.CHANNEL) == set(CHANNELS)
            and truth.groupby("BASE_ID").CHANNEL.nunique().eq(len(CHANNELS)).all()
            and truth.groupby("CHANNEL").size().eq(len(reservation)).all()
        ),
        "recipe_metadata_exact": recipe_equal,
        "audio_id_alignment": {path.stem for path in audio_paths} == set(truth.ID),
        "audio_format_and_duration": bool(properties) and all(
            item.samplerate == SR and item.channels == 1 and 4.0 <= item.duration <= 60.0
            for item in properties
        ),
        "audio_hash_id_alignment": set(recorded_hashes) == set(truth.ID),
        "audio_hash_values": len(recorded_hashes) == expected_rows and all(
            recorded_hashes.get(path.stem) == sha256_file(path) for path in audio_paths
        ),
        "sample_is_neutral_template_only": (
            set(sample.ID) == set(truth.ID)
            and set(PREDICTION_COLUMNS).issubset(sample.columns)
            and all(set(pd.to_numeric(sample[column])) == {0.5} for column in PREDICTION_COLUMNS)
        ),
        "no_prediction_artifacts": not any(
            path.name in {"predictions.csv", "scores.csv", "metrics.json"}
            for path in args.bank_dir.rglob("*") if path.is_file()
        ),
    }
    report = {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": {key: bool(value) for key, value in checks.items()},
        "checks_passed": int(sum(bool(value) for value in checks.values())),
        "checks_total": len(checks), "base_rows": len(reservation),
        "rendered_rows": len(truth), "truth_sha256": sha256_file(required_paths["truth"]),
        "reservation_sha256": reservation_provenance["reservation_sha256"],
        "authenticity_detector_scores_read": False,
    }
    staging, publish = atomic_output_dir(args.output_dir)
    try:
        (staging / "validation.json").write_text(json.dumps(report, indent=2) + "\n", "utf-8")
        if report["status"] != "PASS":
            raise RuntimeError(f"bank validation failed: {[key for key, value in checks.items() if not value]}")
        publish()
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    print(json.dumps(report, indent=2))


def default(relative: str) -> Path:
    return ROOT / relative


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    reserve_parser = sub.add_parser("reserve")
    reserve_parser.add_argument("--echo-parquet", type=Path, action="append", required=True)
    reserve_parser.add_argument("--unmixed-details", type=Path, default=default("data/external/mixfake/MixFake/protocols/unmixed_details.csv"))
    reserve_parser.add_argument("--background-protocol", type=Path, default=default("data/external/mixfake/MixFake/protocols/Mixed_and_Back_BackLabel.txt"))
    reserve_parser.add_argument("--musiccaps-metadata", type=Path, default=default("data/sources/musiccaps_metadata/musiccaps-public.csv"))
    reserve_parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    reserve_parser.add_argument("--per-cell", type=int, default=DEFAULT_PER_CELL)
    reserve_parser.add_argument("--output-dir", type=Path, required=True)
    reserve_parser.set_defaults(func=reserve)

    materialize = sub.add_parser("materialize-voices")
    materialize.add_argument("--reservation-dir", type=Path, required=True)
    materialize.add_argument("--echo-parquet", type=Path, action="append", required=True)
    materialize.add_argument("--output-dir", type=Path, required=True)
    materialize.set_defaults(func=materialize_voices)

    extract = sub.add_parser("extract-music")
    extract.add_argument("--reservation-dir", type=Path, required=True)
    extract.add_argument("--archive-base", type=Path, default=default("data/external/mixfake_archive/MixFake.7z"))
    extract.add_argument("--source-root", type=Path, required=True)
    extract.set_defaults(func=extract_music)

    screen = sub.add_parser("semantic-screen")
    screen.add_argument("--reservation-dir", type=Path, required=True)
    screen.add_argument("--source-root", type=Path, action="append", required=True)
    screen.add_argument("--panns-dir", type=Path, default=default("models/panns"))
    screen.add_argument("--demucs-repo", type=Path, default=default("models/htdemucs"))
    screen.add_argument("--device", default="cuda:0")
    screen.add_argument("--panns-threshold", type=float, default=0.20)
    screen.add_argument("--demucs-threshold-db", type=float, default=-1.5)
    screen.add_argument("--only-source-id", action="append")
    screen.add_argument("--output-dir", type=Path, required=True)
    screen.set_defaults(func=semantic_screen)

    revise_parser = sub.add_parser("revise")
    revise_parser.add_argument("--reservation-dir", type=Path, required=True)
    revise_parser.add_argument("--semantic-scores", type=Path, action="append", required=True)
    revise_parser.add_argument("--output-dir", type=Path, required=True)
    revise_parser.set_defaults(func=revise)

    validate_parser = sub.add_parser("validate-sources")
    validate_parser.add_argument("--reservation-dir", type=Path, required=True)
    validate_parser.add_argument("--source-root", type=Path, action="append", required=True)
    validate_parser.add_argument("--output-dir", type=Path, required=True)
    validate_parser.set_defaults(func=validate_sources)

    render_parser = sub.add_parser("render")
    render_parser.add_argument("--reservation-dir", type=Path, required=True)
    render_parser.add_argument("--source-validation", type=Path, required=True)
    render_parser.add_argument("--source-root", type=Path, action="append", required=True)
    render_parser.add_argument("--ffmpeg", type=Path, default=Path("/usr/bin/ffmpeg"))
    render_parser.add_argument("--output-dir", type=Path, required=True)
    render_parser.set_defaults(func=render)

    bank_validate = sub.add_parser("validate-bank")
    bank_validate.add_argument("--reservation-dir", type=Path, required=True)
    bank_validate.add_argument("--source-validation", type=Path, required=True)
    bank_validate.add_argument("--bank-dir", type=Path, required=True)
    bank_validate.add_argument("--output-dir", type=Path, required=True)
    bank_validate.set_defaults(func=validate_bank)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
