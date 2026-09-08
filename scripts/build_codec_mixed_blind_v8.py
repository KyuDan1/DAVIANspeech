#!/usr/bin/env python3
"""Build blind v8 without reading any authenticity prediction or score.

v8 adds a joint AI-song condition: SONICS Suno/Udio songs are separated with
frozen HTDemucs into their vocal and accompaniment components and are used only
in FF cells.  They can therefore never contaminate RF with an AI vocal label.
RF uses voice-free FakeMusicCaps; FR uses EchoFake replay attacks; RR is real.
All component identities are protected against every pre-existing truth file.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from io import BytesIO
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import pandas as pd
import soundfile as sf
from scipy.signal import resample_poly


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))
import build_codec_mixed_blind_v7 as v7  # noqa: E402
from build_prospective_mixed_phone_v3 import (  # noqa: E402
    CHANNELS, CELLS, MODES, PREDICTION_COLUMNS, load_audio, mix_layout,
    music_identity, parse_mixfake_protocol, sha256_file, stable_rank,
)
from telephone_channel import apply_channel  # noqa: E402


SR = 16_000
SEED = 20260915
PER_CELL = 6
ID_PREFIX = "cmbv8"
SEMANTIC_PENDING = "panns_demucs_joint_semantic_pending_v8"
SEMANTIC_PASS = "panns_demucs_joint_semantic_pass_v8"
SONICS_PREFIX = "sonics:"


def echo_candidates(paths: list[Path], protection: v7.Protection) -> tuple[dict[int, list[dict]], dict]:
    pools: dict[int, list[dict]] = {0: [], 1: []}
    counts: Counter[str] = Counter()
    columns = [
        "utt_id", "label", "source", "source_speaker_id",
        "synthesis_details", "replay_details",
    ]
    for parquet_path in paths:
        frame = pd.read_parquet(parquet_path, columns=columns)
        for row in frame.itertuples(index=False):
            label_name = v7.clean(row.label)
            if label_name not in {"bonafide", "replay_fake"}:
                continue
            label = int(label_name == "replay_fake")
            synthesis = row.synthesis_details if isinstance(row.synthesis_details, dict) else {}
            replay = row.replay_details if isinstance(row.replay_details, dict) else {}
            model = v7.clean(synthesis.get("model")) if label else "bonafide"
            generator = f"replay:{model}" if label else "bonafide"
            source = v7.clean(row.source)
            speaker = v7.clean(row.source_speaker_id)
            reference = v7.clean(synthesis.get("reference"))
            reference_speaker = v7.clean(synthesis.get("reference_speaker_id"))
            identities = {v7.clean(row.utt_id), source, speaker, reference, reference_speaker}
            variants = set().union(*(v7.identity_variants(value) for value in identities))
            if not source or not speaker or (label and (not model or not reference or not reference_speaker)):
                counts[f"excluded_incomplete:{label_name}"] += 1
                continue
            if variants & protection.exact:
                counts[f"excluded_protected:{label_name}"] += 1
                continue
            pools[label].append({
                "KIND": "voice", "LABEL": label,
                "VOICE_SOURCE_ID": f"echofake:{row.utt_id}",
                "VOICE_SPEAKER": speaker, "VOICE_CONTENT_ID": source,
                "VOICE_REFERENCE_ID": reference,
                "VOICE_REFERENCE_SPEAKER": reference_speaker,
                "VOICE_GENERATOR": generator,
                "VOICE_ARCHIVE_ID": v7.clean(row.utt_id),
                "VOICE_ARCHIVE_MEMBER": f"echofake_voice/{row.utt_id}.flac",
                "VOICE_PARQUET_SHARD": parquet_path.name,
                "REPLAY_PLAYER": v7.clean(replay.get("player")),
                "REPLAY_RECORDER": v7.clean(replay.get("recorder")),
                "REPLAY_DISTANCE": v7.clean(replay.get("distance")),
            })
            counts[f"eligible:{label_name}:{generator}"] += 1
    return pools, dict(sorted(counts.items()))


def standard_music_candidates(
    details_path: Path, protocol_path: Path, metadata_path: Path,
    protection: v7.Protection,
) -> tuple[dict[int, list[dict]], dict]:
    pools, counts = v7.music_metadata(
        details_path, protocol_path, metadata_path, protection,
    )
    converted: dict[int, list[dict]] = {0: [], 1: []}
    for label in (0, 1):
        for row in pools[label]:
            converted[label].append({
                "KIND": "music", "LABEL": label,
                "MUSIC_SOURCE_ID": row["source_id"],
                "MUSIC_GROUP_ID": row["group"],
                "MUSIC_GENERATOR": row["generator"],
                "MUSIC_PLATFORM": "FMA" if not label else "FakeMusicCaps",
                "MUSIC_ARCHIVE_ID": row["archive_id"],
                "MUSIC_ARCHIVE_MEMBER": row["archive_member"],
                "SONICS_ORIGINAL_ARCHIVE_MEMBER": "",
            })
    return converted, counts


def sonics_candidates(
    details_path: Path, protocol_path: Path, metadata_path: Path,
    protection: v7.Protection,
) -> tuple[list[dict], dict]:
    details = pd.read_csv(details_path, dtype=str).fillna("")
    protocol = parse_mixfake_protocol(protocol_path)
    metadata = pd.read_csv(metadata_path, dtype=str).fillna("")
    metadata = metadata.loc[metadata.target.astype(str).eq("1")].copy()
    metadata_by_name = {
        str(row.filename): row for row in metadata.itertuples(index=False)
    }
    counts: Counter[str] = Counter()
    rows: list[dict] = []
    selected = details.loc[
        details.split.eq("eval") & details.major_type.eq("Background")
        & details.sub_type.eq("Music") & details.sub_dataset.eq("SONICS")
        & details.authenticity.eq("spoof")
    ]
    for row in selected.itertuples(index=False):
        archive_id = Path(row.file_path).stem
        protocol_row = protocol.get(archive_id)
        if protocol_row is None or protocol_row["label"] != "spoof":
            raise ValueError(f"missing SONICS protocol: {archive_id}")
        source_id, group, platform = music_identity(archive_id, "SONICS")
        original_name = archive_id.removeprefix("FM_fake_songs_wav_")
        meta = metadata_by_name.get(original_name)
        if meta is None:
            counts["excluded_missing_sonics_metadata"] += 1
            continue
        algorithm = v7.clean(meta.algorithm)
        metadata_platform = v7.clean(meta.source)
        if metadata_platform != platform or not algorithm:
            raise ValueError(f"SONICS metadata conflict: {archive_id}")
        variants = v7.identity_variants(source_id) | v7.identity_variants(archive_id)
        if variants & protection.exact or group in protection.music_groups:
            counts[f"excluded_protected:{algorithm}"] += 1
            continue
        stem = f"sonics_stems/{archive_id}"
        rows.append({
            "KIND": "joint", "LABEL": 1,
            "VOICE_SOURCE_ID": f"sonics-vocal:{source_id}",
            "VOICE_SPEAKER": f"sonics-vocal:{group}",
            "VOICE_CONTENT_ID": group, "VOICE_REFERENCE_ID": "",
            "VOICE_REFERENCE_SPEAKER": "",
            "VOICE_GENERATOR": f"{SONICS_PREFIX}{algorithm}:vocal",
            "VOICE_ARCHIVE_ID": archive_id,
            "VOICE_ARCHIVE_MEMBER": stem + "__vocals.flac",
            "VOICE_PARQUET_SHARD": "",
            "REPLAY_PLAYER": "", "REPLAY_RECORDER": "", "REPLAY_DISTANCE": "",
            "MUSIC_SOURCE_ID": f"sonics-music:{source_id}",
            "MUSIC_GROUP_ID": group,
            "MUSIC_GENERATOR": f"{SONICS_PREFIX}{algorithm}",
            "MUSIC_PLATFORM": platform,
            "MUSIC_ARCHIVE_ID": archive_id,
            "MUSIC_ARCHIVE_MEMBER": stem + "__accompaniment.flac",
            "SONICS_ORIGINAL_ARCHIVE_MEMBER": protocol_row["member"],
        })
        counts[f"eligible:{platform}:{algorithm}"] += 1
    return rows, dict(sorted(counts.items()))


def balanced_voice(rows: list[dict], count: int, seed: int,
                   forbidden: set[str] | None = None) -> list[dict]:
    forbidden = set() if forbidden is None else set(forbidden)
    buckets: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        buckets[row["VOICE_GENERATOR"]].append(row)
    for generator, values in buckets.items():
        values.sort(key=lambda row: stable_rank(seed, generator, row["VOICE_ARCHIVE_ID"]))
    order = sorted(buckets, key=lambda value: stable_rank(seed, "generator", value))
    selected: list[dict] = []
    used = set(forbidden)
    while len(selected) < count:
        progressed = False
        for generator in order:
            choice = None
            for row in buckets[generator]:
                identities = {
                    row["VOICE_SPEAKER"], row["VOICE_CONTENT_ID"],
                    row["VOICE_REFERENCE_ID"], row["VOICE_REFERENCE_SPEAKER"],
                } - {""}
                if identities.isdisjoint(used):
                    choice = row
                    break
            if choice is None:
                continue
            buckets[generator].remove(choice)
            selected.append(choice)
            used.update({
                choice["VOICE_SPEAKER"], choice["VOICE_CONTENT_ID"],
                choice["VOICE_REFERENCE_ID"], choice["VOICE_REFERENCE_SPEAKER"],
            } - {""})
            progressed = True
            if len(selected) == count:
                break
        if not progressed:
            raise ValueError(f"need {count} voice candidates, found {len(selected)}")
    return selected


def balanced_music(rows: list[dict], count: int, seed: int) -> list[dict]:
    buckets: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        buckets[row["MUSIC_GENERATOR"]].append(row)
    for generator, values in buckets.items():
        values.sort(key=lambda row: stable_rank(seed, generator, row["MUSIC_GROUP_ID"], row["MUSIC_SOURCE_ID"]))
    order = sorted(buckets, key=lambda value: stable_rank(seed, "generator", value))
    selected: list[dict] = []
    used: set[str] = set()
    while len(selected) < count:
        progressed = False
        for generator in order:
            choice = next((row for row in buckets[generator] if row["MUSIC_GROUP_ID"] not in used), None)
            if choice is None:
                continue
            buckets[generator].remove(choice)
            selected.append(choice)
            used.add(choice["MUSIC_GROUP_ID"])
            progressed = True
            if len(selected) == count:
                break
        if not progressed:
            raise ValueError(f"need {count} music candidates, found {len(selected)}")
    return selected


def validate_reservation(frame: pd.DataFrame, require_semantic: bool = False) -> None:
    expected = len(MODES) * len(CELLS) * PER_CELL
    if len(frame) != expected or frame.BASE_ID.nunique() != expected:
        raise ValueError(f"expected {expected} unique bases")
    counts = frame.groupby(["MIX_MODE", "COMPONENT_CASE"]).size()
    if len(counts) != 12 or set(counts.astype(int)) != {PER_CELL}:
        raise ValueError(f"layout/case imbalance: {counts.to_dict()}")
    file_label = frame[["VOICE_FAKE", "MUSIC_FAKE"]].astype(int).max(axis=1)
    if not np.array_equal(file_label, frame.FILE_FAKE.astype(int)):
        raise ValueError("FILE_FAKE != component OR")
    for column in ("VOICE_SOURCE_ID", "MUSIC_SOURCE_ID", "MUSIC_GROUP_ID"):
        if frame[column].duplicated().any():
            raise ValueError(f"source/group reuse: {column}")
    if frame.VOICE_SPEAKER.duplicated().any():
        raise ValueError("target speaker/source reuse")
    references = frame.VOICE_REFERENCE_SPEAKER.loc[
        frame.VOICE_REFERENCE_SPEAKER.ne("")
    ]
    if references.duplicated().any() or set(references) & set(frame.VOICE_SPEAKER):
        raise ValueError("reference speaker reuse/crossover")
    sonics = frame.MUSIC_SOURCE_KIND.eq("sonics_joint")
    if not (sonics == frame.COMPONENT_CASE.eq("FF")).all():
        raise ValueError("SONICS joint sources must occur in every and only FF cell")
    if not frame.loc[sonics, "VOICE_SOURCE_KIND"].eq("sonics_joint").all():
        raise ValueError("SONICS music is not paired to its vocal component")
    if not frame.loc[sonics, "VOICE_FAKE"].astype(int).eq(1).all():
        raise ValueError("SONICS vocal was not labelled fake")
    if frame.loc[frame.COMPONENT_CASE.eq("RF"), "MUSIC_PLATFORM"].isin({"suno", "udio"}).any():
        raise ValueError("Suno/Udio vocal song contaminated RF")
    replay_counts = frame.loc[frame.VOICE_SOURCE_KIND.eq("echofake_replay"), "VOICE_GENERATOR"].value_counts()
    if len(replay_counts) < 8 or replay_counts.max() - replay_counts.min() > 1:
        raise ValueError(f"replay TTS imbalance: {replay_counts.to_dict()}")
    sonics_counts = frame.loc[sonics, "MUSIC_GENERATOR"].value_counts()
    if len(sonics_counts) != 5 or sonics_counts.max() - sonics_counts.min() > 1:
        raise ValueError(f"SONICS algorithm imbalance: {sonics_counts.to_dict()}")
    if set(frame.loc[sonics, "MUSIC_PLATFORM"]) != {"suno", "udio"}:
        raise ValueError("both Suno and Udio must be represented")
    if require_semantic and set(frame.MUSIC_VOCAL_SCREEN) != {SEMANTIC_PASS}:
        raise ValueError("semantic screen is incomplete")


def reserve(args: argparse.Namespace) -> None:
    truth_paths, source_hash_paths = v7.discover_protected_files()
    protection = v7.build_protection(truth_paths, source_hash_paths)
    voice_pools, voice_counts = echo_candidates(args.echo_parquet, protection)
    standard_music, music_counts = standard_music_candidates(
        args.unmixed_details, args.background_protocol, args.musiccaps_metadata, protection,
    )
    joint_pool, joint_counts = sonics_candidates(
        args.unmixed_details, args.background_protocol, args.sonics_metadata, protection,
    )
    real_voice = balanced_voice(voice_pools[0], 36, args.seed + 10)
    real_identity = set().union(*({
        row["VOICE_SPEAKER"], row["VOICE_CONTENT_ID"], row["VOICE_REFERENCE_ID"],
        row["VOICE_REFERENCE_SPEAKER"],
    } - {""} for row in real_voice))
    replay_voice = balanced_voice(voice_pools[1], 18, args.seed + 11, real_identity)
    real_music = balanced_music(standard_music[0], 36, args.seed + 20)
    fake_instrumental = balanced_music(standard_music[1], 18, args.seed + 21)
    joint = balanced_music(joint_pool, 18, args.seed + 22)
    voice_offsets = {"real": 0, "replay": 0}
    music_offsets = {"real": 0, "fake": 0, "joint": 0}
    records: list[dict] = []
    for mode in MODES:
        for voice_fake, music_fake in CELLS:
            for repeat in range(PER_CELL):
                cell = f"{'F' if voice_fake else 'R'}{'F' if music_fake else 'R'}"
                if cell == "FF":
                    voice = music = joint[music_offsets["joint"]]
                    music_offsets["joint"] += 1
                    voice_kind = music_kind = "sonics_joint"
                else:
                    if voice_fake:
                        voice = replay_voice[voice_offsets["replay"]]
                        voice_offsets["replay"] += 1
                        voice_kind = "echofake_replay"
                    else:
                        voice = real_voice[voice_offsets["real"]]
                        voice_offsets["real"] += 1
                        voice_kind = "echofake_real"
                    if music_fake:
                        music = fake_instrumental[music_offsets["fake"]]
                        music_offsets["fake"] += 1
                        music_kind = "fakemusiccaps_instrumental"
                    else:
                        music = real_music[music_offsets["real"]]
                        music_offsets["real"] += 1
                        music_kind = "fma_instrumental"
                key = f"{args.seed}|{mode}|{cell}|{repeat}"
                snr = (-10, -5, 0, 5, 10)[int(stable_rank(args.seed, key, "snr")[:8], 16) % 5]
                overlap = (0.25, 0.50, 0.75)[int(stable_rank(args.seed, key, "overlap")[:8], 16) % 3]
                order = "voice_first" if int(stable_rank(args.seed, key, "order")[:8], 16) % 2 == 0 else "music_first"
                gap = (0.0, 0.2, 0.5)[int(stable_rank(args.seed, key, "gap")[:8], 16) % 3]
                records.append({
                    "BASE_ID": f"{ID_PREFIX}_{len(records):04d}",
                    "FILE_FAKE": int(voice_fake or music_fake),
                    "VOICE_FAKE": voice_fake, "MUSIC_FAKE": music_fake,
                    "VOICE_PRESENT": 1, "MUSIC_PRESENT": 1, "AUDIO_TYPE": "mixed",
                    "MIX_MODE": mode, "COMPONENT_CASE": cell,
                    "EVAL_CELL": f"{mode}__{cell}",
                    "VOICE_SOURCE_KIND": voice_kind,
                    **{column: voice[column] for column in (
                        "VOICE_SOURCE_ID", "VOICE_SPEAKER", "VOICE_CONTENT_ID",
                        "VOICE_REFERENCE_ID", "VOICE_REFERENCE_SPEAKER", "VOICE_GENERATOR",
                        "VOICE_ARCHIVE_ID", "VOICE_ARCHIVE_MEMBER", "VOICE_PARQUET_SHARD",
                        "REPLAY_PLAYER", "REPLAY_RECORDER", "REPLAY_DISTANCE",
                    )},
                    "MUSIC_SOURCE_KIND": music_kind,
                    **{column: music[column] for column in (
                        "MUSIC_SOURCE_ID", "MUSIC_GROUP_ID", "MUSIC_GENERATOR",
                        "MUSIC_PLATFORM", "MUSIC_ARCHIVE_ID", "MUSIC_ARCHIVE_MEMBER",
                        "SONICS_ORIGINAL_ARCHIVE_MEMBER",
                    )},
                    "MUSIC_VOCAL_SCREEN": SEMANTIC_PENDING,
                    "SNR_DB": snr if mode != "sequential" else "",
                    "OVERLAP_FRACTION": overlap if mode == "partial_overlap" else (1.0 if mode == "concurrent" else 0.0),
                    "ORDER": order, "GAP_SECONDS": gap if mode == "sequential" else 0.0,
                    "SOURCE_DATASET": "EchoFake_replay+MixFake_FMC_FMA_SONICS",
                    "PROVENANCE_MODE": "all_existing_truth_source_speaker_song_hash_disjoint",
                })
    reservation = pd.DataFrame(records)
    validate_reservation(reservation)
    candidate_rows = [*voice_pools[0], *voice_pools[1], *standard_music[0], *standard_music[1], *joint_pool]
    candidates = pd.DataFrame(candidate_rows).fillna("")
    provenance = {
        "schema_version": "codec_mixed_blind_v8", "stage": "reserved_pending_semantic_screen",
        "seed": args.seed, "per_cell": PER_CELL, "base_rows": len(reservation),
        "rendered_rows_planned": len(reservation) * len(CHANNELS), "channels": list(CHANNELS),
        "protected_truth_manifests": len(truth_paths),
        "protected_source_hash_manifests": len(source_hash_paths),
        "voice_candidate_counts": voice_counts, "music_candidate_counts": music_counts,
        "sonics_candidate_counts": joint_counts,
        "selection": {
            "replay_voice_generators": sorted(set(reservation.loc[reservation.VOICE_SOURCE_KIND.eq("echofake_replay"), "VOICE_GENERATOR"])),
            "instrumental_fake_music_generators": sorted(set(reservation.loc[reservation.MUSIC_SOURCE_KIND.eq("fakemusiccaps_instrumental"), "MUSIC_GENERATOR"])),
            "sonics_joint_generators": sorted(set(reservation.loc[reservation.MUSIC_SOURCE_KIND.eq("sonics_joint"), "MUSIC_GENERATOR"])),
            "suno_udio_only_in_ff": True,
        },
        "authenticity_detector_inference": False, "authenticity_score_computed": False,
    }
    write_reservation(args.output_dir, reservation, candidates, protection.snapshot, provenance)
    print(json.dumps({
        "status": "reserved", "bases": len(reservation),
        "replay_voice": reservation.loc[reservation.VOICE_SOURCE_KIND.eq("echofake_replay"), "VOICE_GENERATOR"].value_counts().to_dict(),
        "sonics_joint": reservation.loc[reservation.MUSIC_SOURCE_KIND.eq("sonics_joint"), "MUSIC_GENERATOR"].value_counts().to_dict(),
        "authenticity_scored": False,
    }, indent=2))


def write_requirements(directory: Path, frame: pd.DataFrame) -> None:
    members = []
    for row in frame.itertuples(index=False):
        member = row.SONICS_ORIGINAL_ARCHIVE_MEMBER or row.MUSIC_ARCHIVE_MEMBER
        members.append({
            "SOURCE_ID": row.MUSIC_SOURCE_ID, "CANONICAL_GROUP": row.MUSIC_GROUP_ID,
            "GENERATOR": row.MUSIC_GENERATOR, "ARCHIVE_MEMBER": member,
            "ARCHIVE_TARGET": "MixFake/" + member,
        })
    requirements = pd.DataFrame(members).drop_duplicates("SOURCE_ID").sort_values("ARCHIVE_TARGET")
    requirements.to_csv(directory / "source_requirements.csv", index=False)
    (directory / "archive_member_targets.txt").write_text("\n".join(requirements.ARCHIVE_TARGET) + "\n", "utf-8")


def write_reservation(directory: Path, frame: pd.DataFrame, candidates: pd.DataFrame,
                      snapshot: pd.DataFrame, provenance: dict) -> None:
    staging, publish = v7.atomic_output_dir(directory)
    try:
        frame.to_csv(staging / "reservation.csv", index=False)
        candidates.to_csv(staging / "candidate_pool.csv", index=False)
        snapshot.to_csv(staging / "protected_inputs.csv", index=False)
        write_requirements(staging, frame)
        payload = dict(provenance)
        payload.update({
            "reservation_sha256": sha256_file(staging / "reservation.csv"),
            "candidate_pool_sha256": sha256_file(staging / "candidate_pool.csv"),
            "protected_inputs_sha256": sha256_file(staging / "protected_inputs.csv"),
            "builder_sha256": sha256_file(Path(__file__)),
            "authenticity_detector_inference": False, "authenticity_score_computed": False,
        })
        (staging / "provenance.json").write_text(json.dumps(payload, indent=2) + "\n", "utf-8")
        publish()
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def load_reservation(directory: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    frame = pd.read_csv(directory / "reservation.csv", dtype=str).fillna("")
    candidates = pd.read_csv(directory / "candidate_pool.csv", dtype=str).fillna("")
    snapshot = pd.read_csv(directory / "protected_inputs.csv", dtype=str).fillna("")
    provenance = json.loads((directory / "provenance.json").read_text("utf-8"))
    for key, name in (("reservation_sha256", "reservation.csv"), ("candidate_pool_sha256", "candidate_pool.csv"), ("protected_inputs_sha256", "protected_inputs.csv")):
        if sha256_file(directory / name) != provenance[key]:
            raise ValueError(f"{name} hash mismatch")
    v7.verify_snapshot(snapshot)
    return frame, candidates, snapshot, provenance


def extract_music(args: argparse.Namespace) -> None:
    import multivolumefile
    import py7zr

    frame, _, _, provenance = load_reservation(args.reservation_dir)
    members = {
        str(row.SONICS_ORIGINAL_ARCHIVE_MEMBER or row.MUSIC_ARCHIVE_MEMBER).replace("\\", "/")
        for row in frame.itertuples(index=False)
    }
    targets = sorted("MixFake/" + member.lstrip("/") for member in members)
    existing = {target for target in targets if (args.source_root / target).is_file()}
    todo = sorted(set(targets) - existing)
    if not todo:
        print(json.dumps({"status": "already_extracted", "targets": len(targets)}, indent=2)); return
    staging = args.source_root / ".mixfake_extract_v8.partial"
    if staging.exists():
        raise FileExistsError(staging)
    staging.mkdir(parents=True)
    try:
        with multivolumefile.open(args.archive_base, "rb") as volume:
            with py7zr.SevenZipFile(volume, "r") as archive:
                missing = sorted(set(todo) - set(archive.getnames()))
                if missing:
                    raise ValueError(f"archive members missing: {missing[:5]}")
                archive.extract(path=staging, targets=todo)
        actual = sorted(path.relative_to(staging).as_posix() for path in staging.rglob("*") if path.is_file())
        if actual != todo:
            raise ValueError(f"extraction mismatch {len(actual)} != {len(todo)}")
        for target in todo:
            source = staging / target
            destination = args.source_root / target
            destination.parent.mkdir(parents=True, exist_ok=True)
            if destination.exists():
                raise FileExistsError(destination)
            source.rename(destination)
        shutil.rmtree(staging)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    report = {
        "stage": "archive_sources_extracted", "targets": len(targets), "new_targets": len(todo),
        "reservation_sha256": provenance["reservation_sha256"],
        "authenticity_detector_inference": False, "authenticity_score_computed": False,
    }
    report_path = args.source_root / f"music_extraction_{provenance['reservation_sha256'][:12]}.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", "utf-8")
    print(json.dumps(report, indent=2))


def materialize_echo(args: argparse.Namespace) -> None:
    frame, _, _, provenance = load_reservation(args.reservation_dir)
    wanted = set(frame.loc[frame.VOICE_SOURCE_KIND.str.startswith("echofake"), "VOICE_ARCHIVE_ID"])
    output = args.source_root / "echofake_voice"
    output.mkdir(parents=True, exist_ok=True)
    found = {path.stem for path in output.glob("*.flac") if path.stem in wanted}
    records = []
    for parquet_path in args.echo_parquet:
        table = pd.read_parquet(parquet_path, columns=["utt_id", "path"])
        selected = table.loc[table.utt_id.astype(str).isin(wanted - found)]
        for row in selected.itertuples(index=False):
            payload = row.path
            audio, rate = sf.read(BytesIO(payload["bytes"]), dtype="float32", always_2d=True)
            mono = np.nan_to_num(audio.mean(axis=1).astype(np.float32))
            if rate != SR:
                divisor = math.gcd(int(rate), SR)
                mono = resample_poly(mono, SR // divisor, int(rate) // divisor).astype(np.float32)
            destination = output / f"{row.utt_id}.flac"
            sf.write(destination, mono, SR, format="FLAC", subtype="PCM_16")
            found.add(str(row.utt_id))
            records.append({"ID": row.utt_id, "SHA256": sha256_file(destination), "DURATION": len(mono) / SR})
    if found != wanted:
        raise ValueError(f"missing EchoFake sources: {sorted(wanted - found)[:5]}")
    report = args.source_root / f"echo_materialization_{provenance['reservation_sha256'][:12]}.csv"
    pd.DataFrame(records).to_csv(report, index=False)
    print(json.dumps({"status": "echo_materialized", "sources": len(wanted)}, indent=2))


def strongest_vocal_crop(vocals: np.ndarray, music: np.ndarray, seconds: int = 12) -> tuple[np.ndarray, np.ndarray, int]:
    count = min(len(vocals), len(music))
    vocals, music = vocals[:count], music[:count]
    length = seconds * SR
    if count < length:
        repeats = int(np.ceil(length / max(count, 1)))
        return np.tile(vocals, repeats)[:length], np.tile(music, repeats)[:length], 0
    squared = vocals.astype(np.float64) ** 2
    cumulative = np.concatenate(([0.0], np.cumsum(squared)))
    # Evaluate one-second boundaries using an O(n) cumulative-energy table.
    positions = np.arange(0, count - length + 1, SR, dtype=int)
    energy = cumulative[positions + length] - cumulative[positions]
    start = int(positions[np.argmax(energy)])
    return vocals[start:start + length], music[start:start + length], start


def materialize_sonics(args: argparse.Namespace) -> None:
    from separation import HTDemucsSeparator

    frame, _, _, provenance = load_reservation(args.reservation_dir)
    selected = frame.loc[frame.MUSIC_SOURCE_KIND.eq("sonics_joint")].drop_duplicates("MUSIC_SOURCE_ID")
    output = args.source_root / "sonics_stems"
    output.mkdir(parents=True, exist_ok=True)
    separator = HTDemucsSeparator(device=args.device, repo=args.demucs_repo, shifts=0, overlap=0.25)
    records = []
    for index, row in enumerate(selected.itertuples(index=False), 1):
        vocal_path = args.source_root / row.VOICE_ARCHIVE_MEMBER
        music_path = args.source_root / row.MUSIC_ARCHIVE_MEMBER
        if vocal_path.is_file() and music_path.is_file():
            continue
        original = v7.locate_source([args.source_root / "MixFake"], row.SONICS_ORIGINAL_ARCHIVE_MEMBER)
        vocals, music = separator.separate(original)
        vocals, music, start = strongest_vocal_crop(vocals, music)
        peak = max(float(np.max(np.abs(vocals))), float(np.max(np.abs(music))), 1e-8)
        scale = min(1.0, 0.98 / peak)
        vocal_path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(vocal_path, vocals * scale, SR, format="FLAC", subtype="PCM_16")
        sf.write(music_path, music * scale, SR, format="FLAC", subtype="PCM_16")
        records.append({
            "MUSIC_SOURCE_ID": row.MUSIC_SOURCE_ID, "ORIGINAL": str(original),
            "ORIGINAL_SHA256": sha256_file(original), "CROP_START_SECONDS": start / SR,
            "VOCALS_SHA256": sha256_file(vocal_path), "MUSIC_SHA256": sha256_file(music_path),
        })
        print(f"SONICS stems {index}/{len(selected)}", flush=True)
    report = args.source_root / f"sonics_stems_{provenance['reservation_sha256'][:12]}.csv"
    pd.DataFrame(records).to_csv(report, index=False)


def semantic_screen(args: argparse.Namespace) -> None:
    from audit_prospective_music_vocals import IndependentPannsVocalScreen, demucs_energy
    from separation import HTDemucsSeparator

    frame, _, _, provenance = load_reservation(args.reservation_dir)
    panns = IndependentPannsVocalScreen(args.panns_dir, args.device)
    separator = HTDemucsSeparator(device=args.device, repo=args.demucs_repo, shifts=0, overlap=0.25)
    records = []
    for index, row in enumerate(frame.itertuples(index=False), 1):
        music_path = v7.locate_source(args.source_root, row.MUSIC_ARCHIVE_MEMBER)
        music_audio = load_audio(music_path)
        music_scores = panns.score(music_audio)
        music_scores.update(demucs_energy(separator, music_path))
        music_pass = (
            music_scores["PANNS_ANY_VOICE_MAX"] <= args.panns_threshold
            and music_scores["DEMUCS_VOCAL_TO_MIX_DB"] <= args.demucs_threshold_db
        )
        vocal_score = ""
        vocal_pass = True
        if row.MUSIC_SOURCE_KIND == "sonics_joint":
            vocal_path = v7.locate_source(args.source_root, row.VOICE_ARCHIVE_MEMBER)
            vocal_score = panns.score(load_audio(vocal_path))["PANNS_ANY_VOICE_MAX"]
            vocal_pass = float(vocal_score) >= args.vocal_presence_threshold
        records.append({
            "MUSIC_SOURCE_ID": row.MUSIC_SOURCE_ID, "MUSIC_GROUP_ID": row.MUSIC_GROUP_ID,
            "MUSIC_GENERATOR": row.MUSIC_GENERATOR, "MUSIC_SOURCE_KIND": row.MUSIC_SOURCE_KIND,
            "MUSIC_AUDIO_SHA256": sha256_file(music_path),
            "PANNS_ANY_VOICE_MAX": music_scores["PANNS_ANY_VOICE_MAX"],
            "DEMUCS_VOCAL_TO_MIX_DB": music_scores["DEMUCS_VOCAL_TO_MIX_DB"],
            "PANNS_THRESHOLD": args.panns_threshold, "DEMUCS_THRESHOLD_DB": args.demucs_threshold_db,
            "MUSIC_ABSENCE_PASS": music_pass, "VOCAL_PANNS_ANY_VOICE_MAX": vocal_score,
            "VOCAL_PRESENCE_THRESHOLD": args.vocal_presence_threshold,
            "VOCAL_PRESENCE_PASS": vocal_pass, "SEMANTIC_SCREEN_PASS": music_pass and vocal_pass,
        })
        if index % 10 == 0:
            print(f"semantic {index}/{len(frame)}", flush=True)
    result = pd.DataFrame(records)
    staging, publish = v7.atomic_output_dir(args.output_dir)
    try:
        result.to_csv(staging / "source_scores.csv", index=False)
        summary = {
            "purpose": "component presence semantics only; no authenticity evaluation",
            "reservation_sha256": provenance["reservation_sha256"], "sources": len(result),
            "passes": int(v7.bool_series(result.SEMANTIC_SCREEN_PASS).sum()),
            "failures": int((~v7.bool_series(result.SEMANTIC_SCREEN_PASS)).sum()),
            "authenticity_detector_inference": False, "authenticity_score_computed": False,
        }
        (staging / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", "utf-8")
        publish()
    except Exception:
        shutil.rmtree(staging, ignore_errors=True); raise


def excluded_semantic_source_ids(provenance: dict, current_ids: set[str]) -> set[str]:
    """Return every source that has already undergone semantic screening."""
    historical = {
        str(item["old"])
        for item in provenance.get("replacement_history", [])
        if isinstance(item, dict) and item.get("old")
    }
    return set(current_ids) | historical


def revise(args: argparse.Namespace) -> None:
    frame, candidates, snapshot, provenance = load_reservation(args.reservation_dir)
    scores = pd.read_csv(args.semantic_scores, dtype=str).fillna("")
    if v7.FORBIDDEN_PREDICTION_COLUMNS & set(scores):
        raise ValueError("authenticity/scoring columns forbidden")
    decisions = dict(zip(scores.MUSIC_SOURCE_ID, v7.bool_series(scores.SEMANTIC_SCREEN_PASS)))
    if set(frame.MUSIC_SOURCE_ID) - set(decisions):
        raise ValueError("missing current semantic decisions")
    failed = [value for value in frame.MUSIC_SOURCE_ID if not decisions[value]]
    result = frame.copy()
    selected_groups = set(result.MUSIC_GROUP_ID)
    selected_ids = set(result.MUSIC_SOURCE_ID)
    excluded_screened = excluded_semantic_source_ids(provenance, set(decisions))
    failure_details = []
    for source_id in failed:
        score_row = scores.loc[scores.MUSIC_SOURCE_ID.eq(source_id)]
        if len(score_row) != 1:
            raise ValueError(f"semantic result must be unique for {source_id}")
        item = score_row.iloc[0]
        failure_details.append({
            "source_id": source_id,
            "source_kind": str(item.get("MUSIC_SOURCE_KIND", "")),
            "generator": str(item.get("MUSIC_GENERATOR", "")),
            "panns_any_voice_max": str(item.get("PANNS_ANY_VOICE_MAX", "")),
            "demucs_vocal_to_mix_db": str(item.get("DEMUCS_VOCAL_TO_MIX_DB", "")),
            "vocal_panns_any_voice_max": str(item.get("VOCAL_PANNS_ANY_VOICE_MAX", "")),
            "music_absence_pass": str(item.get("MUSIC_ABSENCE_PASS", "")),
            "vocal_presence_pass": str(item.get("VOCAL_PRESENCE_PASS", "")),
        })
    replacements = []
    for source_id in failed:
        index = int(result.index[result.MUSIC_SOURCE_ID.eq(source_id)][0])
        kind = result.at[index, "MUSIC_SOURCE_KIND"]
        generator = result.at[index, "MUSIC_GENERATOR"]
        old_group = result.at[index, "MUSIC_GROUP_ID"]
        pool_kind = "joint" if kind == "sonics_joint" else "music"
        available = candidates.loc[
            candidates.KIND.eq(pool_kind) & candidates.MUSIC_GENERATOR.eq(generator)
            & ~candidates.MUSIC_SOURCE_ID.isin(
                selected_ids | excluded_screened
            )
            & ~candidates.MUSIC_GROUP_ID.isin(selected_groups)
        ].copy()
        if available.empty:
            raise ValueError(f"no replacement for {source_id}")
        available["_rank"] = [stable_rank(int(provenance["seed"]), "replace", generator, value) for value in available.MUSIC_SOURCE_ID]
        choice = available.sort_values("_rank").iloc[0]
        selected_ids.add(choice.MUSIC_SOURCE_ID)
        selected_groups.remove(old_group); selected_groups.add(choice.MUSIC_GROUP_ID)
        music_columns = (
            "MUSIC_SOURCE_ID", "MUSIC_GROUP_ID", "MUSIC_GENERATOR", "MUSIC_PLATFORM",
            "MUSIC_ARCHIVE_ID", "MUSIC_ARCHIVE_MEMBER", "SONICS_ORIGINAL_ARCHIVE_MEMBER",
        )
        for column in music_columns:
            result.at[index, column] = choice[column]
        if kind == "sonics_joint":
            for column in (
                "VOICE_SOURCE_ID", "VOICE_SPEAKER", "VOICE_CONTENT_ID", "VOICE_REFERENCE_ID",
                "VOICE_REFERENCE_SPEAKER", "VOICE_GENERATOR", "VOICE_ARCHIVE_ID",
                "VOICE_ARCHIVE_MEMBER", "VOICE_PARQUET_SHARD", "REPLAY_PLAYER",
                "REPLAY_RECORDER", "REPLAY_DISTANCE",
            ):
                result.at[index, column] = choice[column]
        result.at[index, "MUSIC_VOCAL_SCREEN"] = SEMANTIC_PENDING
        replacements.append({"old": source_id, "new": choice.MUSIC_SOURCE_ID, "generator": generator})
    if not failed:
        result["MUSIC_VOCAL_SCREEN"] = SEMANTIC_PASS
    validate_reservation(result, require_semantic=not failed)
    next_provenance = dict(provenance)
    next_provenance.update({
        "stage": "source_semantics_validated" if not failed else "reserved_pending_semantic_screen",
        "parent_reservation_sha256": provenance["reservation_sha256"],
        "semantic_inputs_history": list(provenance.get("semantic_inputs_history", [])) + [{
            "path": str(args.semantic_scores.resolve()),
            "sha256": sha256_file(args.semantic_scores),
        }],
        "semantic_failure_history": list(provenance.get("semantic_failure_history", [])) + failure_details,
        "replacement_history": list(provenance.get("replacement_history", [])) + replacements,
        "semantic_selection_policy": {
            "music_voice_absence": "PANNs any-voice <= 0.20 AND Demucs vocal/mix <= -1.5 dB",
            "sonics_vocal_presence": "PANNs any-voice >= 0.20",
            "replacement": "same label and generator; precommitted stable hash order",
            "authenticity_outputs_used": False,
            "limitation": "final source distribution is conditioned on the frozen component-semantic gate",
        },
    })
    write_reservation(args.output_dir, result, candidates, snapshot, next_provenance)
    print(json.dumps({"status": next_provenance["stage"], "failures": len(failed), "replacements": replacements}, indent=2))


def validate_sources(args: argparse.Namespace) -> None:
    frame, _, snapshot, provenance = load_reservation(args.reservation_dir)
    validate_reservation(frame, require_semantic=True)
    protection = v7.protected_from_snapshot(snapshot)
    variants = set()
    for column in (
        "VOICE_SOURCE_ID", "VOICE_SPEAKER", "VOICE_CONTENT_ID", "VOICE_REFERENCE_ID",
        "VOICE_REFERENCE_SPEAKER", "VOICE_ARCHIVE_ID", "MUSIC_SOURCE_ID", "MUSIC_ARCHIVE_ID",
    ):
        for value in frame[column]:
            variants.update(v7.identity_variants(value))
    identity_overlap = sorted(variants & protection.exact)
    group_overlap = sorted(set(frame.MUSIC_GROUP_ID) & protection.music_groups)
    records = []
    hash_overlap = []
    for row in frame.itertuples(index=False):
        paths = [
            ("voice", row.VOICE_SOURCE_ID, row.VOICE_ARCHIVE_MEMBER),
            ("music", row.MUSIC_SOURCE_ID, row.MUSIC_ARCHIVE_MEMBER),
        ]
        if row.SONICS_ORIGINAL_ARCHIVE_MEMBER:
            paths.append(("joint_original", row.MUSIC_GROUP_ID, row.SONICS_ORIGINAL_ARCHIVE_MEMBER))
        for kind, source_id, member in paths:
            path = v7.locate_source(args.source_root, member)
            digest = sha256_file(path)
            if digest in protection.audio_hashes:
                hash_overlap.append(f"{kind}:{source_id}")
            records.append({"KIND": kind, "SOURCE_ID": source_id, "ARCHIVE_MEMBER": member, "LOCAL_PATH": str(path), "SHA256": digest})
    checks = {
        "snapshot_unchanged": True, "identity_disjoint": not identity_overlap,
        "canonical_song_disjoint": not group_overlap, "source_audio_hash_disjoint": not hash_overlap,
        "all_sources_resolve": bool(records), "semantic_pass": set(frame.MUSIC_VOCAL_SCREEN) == {SEMANTIC_PASS},
        "suno_udio_only_ff": not frame.loc[~frame.COMPONENT_CASE.eq("FF"), "MUSIC_PLATFORM"].isin({"suno", "udio"}).any(),
        "unscored": provenance.get("authenticity_detector_inference") is False and provenance.get("authenticity_score_computed") is False,
    }
    staging, publish = v7.atomic_output_dir(args.output_dir)
    try:
        pd.DataFrame(records).to_csv(staging / "source_hashes.csv", index=False)
        report = {
            "status": "PASS" if all(checks.values()) else "FAIL", "checks": checks,
            "identity_overlap": identity_overlap, "canonical_song_overlap": group_overlap,
            "source_hash_overlap": hash_overlap, "reservation_sha256": provenance["reservation_sha256"],
            "source_hashes_sha256": sha256_file(staging / "source_hashes.csv"),
            "authenticity_detector_scores_read": False,
        }
        (staging / "validation.json").write_text(json.dumps(report, indent=2) + "\n", "utf-8")
        if report["status"] != "PASS":
            raise RuntimeError(report)
        publish()
    except Exception:
        shutil.rmtree(staging, ignore_errors=True); raise
    print(json.dumps(report, indent=2))


def render(args: argparse.Namespace) -> None:
    frame, _, snapshot, provenance = load_reservation(args.reservation_dir)
    validate_reservation(frame, require_semantic=True)
    validation = json.loads((args.source_validation / "validation.json").read_text("utf-8"))
    if validation.get("status") != "PASS" or validation.get("reservation_sha256") != provenance["reservation_sha256"]:
        raise ValueError("source validation mismatch")
    staging, publish = v7.atomic_output_dir(args.output_dir)
    try:
        audio_dir = staging / "audio"; audio_dir.mkdir()
        truths, hashes = [], []
        for index, row in enumerate(frame.itertuples(index=False), 1):
            voice = load_audio(v7.locate_source(args.source_root, row.VOICE_ARCHIVE_MEMBER))
            music = load_audio(v7.locate_source(args.source_root, row.MUSIC_ARCHIVE_MEMBER))
            mixed = mix_layout(voice, music, pd.Series(row._asdict()))
            for channel in CHANNELS:
                key = int(stable_rank(0, row.BASE_ID, channel)[:16], 16) % (2**32)
                audio = apply_channel(mixed, channel, ffmpeg=None if channel == "clean" else args.ffmpeg, key=key)
                sample_id = f"{row.BASE_ID}__{channel}"
                path = audio_dir / f"{sample_id}.flac"
                sf.write(path, audio, SR, format="FLAC", subtype="PCM_16")
                hashes.append({"ID": sample_id, "SHA256": sha256_file(path), "BYTES": path.stat().st_size})
                record = row._asdict(); record.update({"ID": sample_id, "PARENT_ID": row.BASE_ID, "CHANNEL": channel, "DURATION": len(audio) / SR})
                truths.append(record)
            if index % 10 == 0: print(f"render {index}/{len(frame)}", flush=True)
        truth = pd.DataFrame(truths); truth.to_csv(staging / "truth.csv", index=False)
        pd.DataFrame(hashes).to_csv(staging / "audio_hashes.csv", index=False)
        shutil.copy2(args.source_validation / "source_hashes.csv", staging / "source_hashes.csv")
        sample = pd.DataFrame({"ID": truth.ID})
        for column in PREDICTION_COLUMNS: sample[column] = 0.5
        sample.to_csv(staging / "sample_submission.csv", index=False)
        built = dict(provenance); built.update({
            "stage": "built_unscored", "rendered_rows": len(truth),
            "truth_sha256": sha256_file(staging / "truth.csv"),
            "audio_hashes_sha256": sha256_file(staging / "audio_hashes.csv"),
            "source_hashes_sha256": sha256_file(staging / "source_hashes.csv"),
            "source_validation_sha256": sha256_file(args.source_validation / "validation.json"),
            "ffmpeg": subprocess.run([str(args.ffmpeg), "-version"], check=True, capture_output=True, text=True).stdout.splitlines()[0],
            "authenticity_detector_inference": False, "authenticity_score_computed": False,
        })
        (staging / "provenance.json").write_text(json.dumps(built, indent=2) + "\n", "utf-8")
        publish()
    except Exception:
        shutil.rmtree(staging, ignore_errors=True); raise
    print(json.dumps({"status": "built_unscored", "rows": len(truth), "truth_sha256": built["truth_sha256"]}, indent=2))


def validate_bank(args: argparse.Namespace) -> None:
    """Validate the frozen bank without opening truth for model selection.

    This is structural/integrity validation only.  In particular it neither
    imports an authenticity detector nor reads a candidate prediction file.
    """
    reservation, _, snapshot, reservation_provenance = load_reservation(args.reservation_dir)
    validate_reservation(reservation, require_semantic=True)
    v7.verify_snapshot(snapshot)
    required = {
        "truth": args.bank_dir / "truth.csv",
        "sample": args.bank_dir / "sample_submission.csv",
        "audio_hashes": args.bank_dir / "audio_hashes.csv",
        "source_hashes": args.bank_dir / "source_hashes.csv",
        "provenance": args.bank_dir / "provenance.json",
        "source_validation": args.source_validation / "validation.json",
    }
    for path in required.values():
        if not path.is_file():
            raise FileNotFoundError(path)

    truth = pd.read_csv(required["truth"], dtype=str).fillna("")
    sample = pd.read_csv(required["sample"], dtype=str).fillna("")
    audio_hashes = pd.read_csv(required["audio_hashes"], dtype=str).fillna("")
    recorded_hashes = dict(zip(audio_hashes.ID, audio_hashes.SHA256))
    provenance = json.loads(required["provenance"].read_text("utf-8"))
    source_validation = json.loads(required["source_validation"].read_text("utf-8"))
    audio_paths = sorted((args.bank_dir / "audio").glob("*.flac"))
    expected = len(reservation) * len(CHANNELS)

    recipe = truth.merge(
        reservation, on="BASE_ID", how="left", suffixes=("_truth", "_reserved"),
        validate="many_to_one",
    )
    recipe_exact = len(recipe) == expected
    for column in (item for item in reservation.columns if item != "BASE_ID"):
        recipe_exact = recipe_exact and recipe[f"{column}_truth"].astype(str).equals(
            recipe[f"{column}_reserved"].astype(str)
        )
    properties = [sf.info(path) for path in audio_paths]
    cell_layout = truth.groupby(["MIX_MODE", "COMPONENT_CASE"]).PARENT_ID.nunique()
    sonics = truth.MUSIC_GENERATOR.str.startswith(SONICS_PREFIX)
    forbidden = {"predictions.csv", "scores.csv", "metrics.json", "submission.csv"}
    checks = {
        "bank_provenance_unscored": (
            provenance.get("authenticity_detector_inference") is False
            and provenance.get("authenticity_score_computed") is False
        ),
        "source_validation_pass": source_validation.get("status") == "PASS",
        "reservation_hash_carried": (
            provenance.get("reservation_sha256")
            == reservation_provenance["reservation_sha256"]
            == source_validation.get("reservation_sha256")
        ),
        "truth_hash": sha256_file(required["truth"]) == provenance.get("truth_sha256"),
        "source_hashes_hash": sha256_file(required["source_hashes"]) == provenance.get("source_hashes_sha256"),
        "audio_hash_manifest_hash": sha256_file(required["audio_hashes"]) == provenance.get("audio_hashes_sha256"),
        "rendered_row_count": len(truth) == expected,
        "unique_rendered_ids": truth.ID.nunique() == expected,
        "balanced_cells_layouts": (
            set(cell_layout.index) == {
                (mode, f"{'F' if voice else 'R'}{'F' if music else 'R'}")
                for mode in MODES for voice, music in CELLS
            }
            and cell_layout.eq(PER_CELL).all()
        ),
        "five_paired_channels": (
            set(truth.CHANNEL) == set(CHANNELS)
            and truth.groupby("BASE_ID").CHANNEL.nunique().eq(len(CHANNELS)).all()
            and truth.groupby("CHANNEL").size().eq(len(reservation)).all()
        ),
        "recipe_metadata_exact": recipe_exact,
        "sonics_only_and_all_ff": (sonics == truth.COMPONENT_CASE.eq("FF")).all(),
        "audio_id_alignment": {path.stem for path in audio_paths} == set(truth.ID),
        "audio_format_and_duration": bool(properties) and all(
            item.samplerate == SR and item.channels == 1 and 4.0 <= item.duration <= 60.0
            for item in properties
        ),
        "audio_hash_id_alignment": set(recorded_hashes) == set(truth.ID),
        "audio_hash_values": len(recorded_hashes) == expected and all(
            recorded_hashes.get(path.stem) == sha256_file(path) for path in audio_paths
        ),
        "sample_is_neutral_template_only": (
            set(sample.ID) == set(truth.ID)
            and set(PREDICTION_COLUMNS).issubset(sample.columns)
            and all(set(pd.to_numeric(sample[column])) == {0.5} for column in PREDICTION_COLUMNS)
        ),
        "no_prediction_or_score_artifacts": not any(
            path.name in forbidden for path in args.bank_dir.rglob("*") if path.is_file()
        ),
    }
    report = {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": {key: bool(value) for key, value in checks.items()},
        "checks_passed": sum(bool(value) for value in checks.values()),
        "checks_total": len(checks), "base_rows": len(reservation),
        "rendered_rows": len(truth), "truth_sha256": sha256_file(required["truth"]),
        "reservation_sha256": reservation_provenance["reservation_sha256"],
        "authenticity_detector_scores_read": False,
    }
    staging, publish = v7.atomic_output_dir(args.output_dir)
    try:
        (staging / "validation.json").write_text(json.dumps(report, indent=2) + "\n", "utf-8")
        if report["status"] != "PASS":
            failed = [key for key, value in checks.items() if not value]
            raise RuntimeError(f"bank validation failed: {failed}")
        publish()
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    print(json.dumps(report, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    common_echo = [
        ROOT / "data/external/echofake/dev-00000-of-00001.parquet",
        ROOT / "data/external/echofake/open_set_eval-00000-of-00003.parquet",
        ROOT / "data/external/echofake/open_set_eval-00001-of-00003.parquet",
        ROOT / "data/external/echofake/open_set_eval-00002-of-00003.parquet",
    ]
    p = sub.add_parser("reserve")
    p.add_argument("--echo-parquet", type=Path, action="append", default=[])
    p.add_argument("--unmixed-details", type=Path, default=ROOT / "data/external/mixfake/MixFake/protocols/unmixed_details.csv")
    p.add_argument("--background-protocol", type=Path, default=ROOT / "data/external/mixfake/MixFake/protocols/Mixed_and_Back_BackLabel.txt")
    p.add_argument("--musiccaps-metadata", type=Path, default=ROOT / "data/sources/musiccaps_metadata/musiccaps-public.csv")
    p.add_argument("--sonics-metadata", type=Path, default=ROOT / "data/sources/sonics_metadata/fake_songs.csv")
    p.add_argument("--seed", type=int, default=SEED); p.add_argument("--output-dir", type=Path, required=True); p.set_defaults(func=reserve)

    p = sub.add_parser("extract-music"); p.add_argument("--reservation-dir", type=Path, required=True)
    p.add_argument("--archive-base", type=Path, default=ROOT / "data/external/mixfake_archive/MixFake.7z")
    p.add_argument("--source-root", type=Path, required=True); p.set_defaults(func=extract_music)
    p = sub.add_parser("materialize-echo"); p.add_argument("--reservation-dir", type=Path, required=True)
    p.add_argument("--echo-parquet", type=Path, action="append", default=[]); p.add_argument("--source-root", type=Path, required=True); p.set_defaults(func=materialize_echo)
    p = sub.add_parser("materialize-sonics"); p.add_argument("--reservation-dir", type=Path, required=True)
    p.add_argument("--source-root", type=Path, required=True); p.add_argument("--device", default="cuda:3")
    p.add_argument("--demucs-repo", type=Path, default=ROOT / "models/htdemucs"); p.set_defaults(func=materialize_sonics)
    p = sub.add_parser("semantic-screen"); p.add_argument("--reservation-dir", type=Path, required=True)
    p.add_argument("--source-root", type=Path, action="append", required=True); p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--device", default="cuda:3"); p.add_argument("--panns-dir", type=Path, default=ROOT / "models/panns")
    p.add_argument("--demucs-repo", type=Path, default=ROOT / "models/htdemucs")
    p.add_argument("--panns-threshold", type=float, default=0.20); p.add_argument("--demucs-threshold-db", type=float, default=-1.5)
    p.add_argument("--vocal-presence-threshold", type=float, default=0.20); p.set_defaults(func=semantic_screen)
    p = sub.add_parser("revise"); p.add_argument("--reservation-dir", type=Path, required=True)
    p.add_argument("--semantic-scores", type=Path, required=True); p.add_argument("--output-dir", type=Path, required=True); p.set_defaults(func=revise)
    p = sub.add_parser("validate-sources"); p.add_argument("--reservation-dir", type=Path, required=True)
    p.add_argument("--source-root", type=Path, action="append", required=True); p.add_argument("--output-dir", type=Path, required=True); p.set_defaults(func=validate_sources)
    p = sub.add_parser("render"); p.add_argument("--reservation-dir", type=Path, required=True)
    p.add_argument("--source-validation", type=Path, required=True); p.add_argument("--source-root", type=Path, action="append", required=True)
    p.add_argument("--ffmpeg", type=Path, default=ROOT.parent / "conda_envs/envs/davianspeech/bin/ffmpeg")
    p.add_argument("--output-dir", type=Path, required=True); p.set_defaults(func=render)
    p = sub.add_parser("validate-bank"); p.add_argument("--reservation-dir", type=Path, required=True)
    p.add_argument("--source-validation", type=Path, required=True); p.add_argument("--bank-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True); p.set_defaults(func=validate_bank)
    args = parser.parse_args()
    if hasattr(args, "echo_parquet") and not args.echo_parquet: args.echo_parquet = common_echo
    args.func(args)


if __name__ == "__main__":
    main()
