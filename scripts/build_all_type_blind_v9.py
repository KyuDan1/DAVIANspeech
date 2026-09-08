#!/usr/bin/env python3
"""Build the prospective all-type blind v9 without authenticity inference."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import shutil
import subprocess
import sys
import re

import numpy as np
import pandas as pd
import soundfile as sf


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))
import build_codec_mixed_blind_v7 as v7  # noqa: E402
import build_codec_mixed_blind_v8 as v8  # noqa: E402
from build_prospective_mixed_phone_v3 import (  # noqa: E402
    CHANNELS, CELLS, MODES, PREDICTION_COLUMNS, crop_or_tile, load_audio,
    mix_layout, peak_limit, sha256_file, stable_rank,
)
from telephone_channel import apply_channel  # noqa: E402


SR = 16_000
SEED = 20260916
PER_MIXED_CELL_LAYOUT = 6
PER_SINGLE_LABEL = 12
ID_PREFIX = "atbv9"
SEMANTIC_PENDING = "presence_only_semantic_pending_v9"
SEMANTIC_PASS = "presence_only_semantic_pass_v9"
SEMANTIC_NA = "not_applicable_voice_only_v9"
JOINT_KIND = "sonics_joint_original"

VOICE_COLUMNS = (
    "VOICE_SOURCE_ID", "VOICE_SPEAKER", "VOICE_CONTENT_ID", "VOICE_REFERENCE_ID",
    "VOICE_REFERENCE_SPEAKER", "VOICE_GENERATOR", "VOICE_ARCHIVE_ID",
    "VOICE_ARCHIVE_MEMBER", "VOICE_PARQUET_SHARD", "REPLAY_PLAYER",
    "REPLAY_RECORDER", "REPLAY_DISTANCE",
)
MUSIC_COLUMNS = (
    "MUSIC_SOURCE_ID", "MUSIC_GROUP_ID", "MUSIC_GENERATOR", "MUSIC_PLATFORM",
    "MUSIC_ARCHIVE_ID", "MUSIC_ARCHIVE_MEMBER", "SONICS_ORIGINAL_ARCHIVE_MEMBER",
)
# These are sample/person/song identities. Generator, platform, replay device and
# archive-family labels are deliberately excluded: v9 must be identity-disjoint,
# not generator-family-disjoint (the latter would make prospective coverage
# impossible and was the source of false overlaps such as ``FMA`` itself).
IDENTITY_COLUMNS = (
    "VOICE_SOURCE_ID", "VOICE_SPEAKER", "VOICE_CONTENT_ID",
    "VOICE_REFERENCE_ID", "VOICE_REFERENCE_SPEAKER",
    "MUSIC_SOURCE_ID", "MUSIC_GROUP_ID",
)


def all_protected_files() -> tuple[list[Path], list[Path]]:
    truths = sorted(path for path in (ROOT / "data").glob("**/truth*.csv") if path.is_file())
    hashes = sorted({
        *[path for path in (ROOT / "data").glob("**/source_hashes.csv") if path.is_file()],
        *[path for path in (ROOT / "data").glob("**/audio_hashes.csv") if path.is_file()],
    })
    if not truths:
        raise ValueError("no truth manifests available for protection")
    return truths, hashes


def direct_joint(row: dict | pd.Series) -> dict:
    result = dict(row)
    original = str(result["SONICS_ORIGINAL_ARCHIVE_MEMBER"])
    result["VOICE_ARCHIVE_MEMBER"] = original
    result["MUSIC_ARCHIVE_MEMBER"] = original
    return result


def blank(columns: tuple[str, ...]) -> dict[str, str]:
    return {column: "" for column in columns}


def base_record(
    index: int, audio_type: str, voice: dict | None, music: dict | None,
    voice_fake: int | str, music_fake: int | str, mode: str,
    cell: str, seed: int,
) -> dict:
    voice_present, music_present = voice is not None, music is not None
    file_fake = int((voice_present and int(voice_fake)) or (music_present and int(music_fake)))
    key = f"{seed}|{audio_type}|{mode}|{cell}|{index}"
    snr = (-10, -5, 0, 5, 10)[int(stable_rank(seed, key, "snr")[:8], 16) % 5]
    overlap = (0.25, 0.50, 0.75)[int(stable_rank(seed, key, "overlap")[:8], 16) % 3]
    order = "voice_first" if int(stable_rank(seed, key, "order")[:8], 16) % 2 == 0 else "music_first"
    gap = (0.0, 0.2, 0.5)[int(stable_rank(seed, key, "gap")[:8], 16) % 3]
    result = {
        "BASE_ID": f"{ID_PREFIX}_{index:04d}", "FILE_FAKE": file_fake,
        "VOICE_FAKE": voice_fake, "MUSIC_FAKE": music_fake,
        "VOICE_PRESENT": int(voice_present), "MUSIC_PRESENT": int(music_present),
        "AUDIO_TYPE": audio_type, "MIX_MODE": mode, "COMPONENT_CASE": cell,
        "EVAL_CELL": f"{mode}__{cell}",
        "VOICE_SOURCE_KIND": "" if voice is None else voice["VOICE_SOURCE_KIND"],
        **(blank(VOICE_COLUMNS) if voice is None else {column: voice[column] for column in VOICE_COLUMNS}),
        "MUSIC_SOURCE_KIND": "" if music is None else music["MUSIC_SOURCE_KIND"],
        **(blank(MUSIC_COLUMNS) if music is None else {column: music[column] for column in MUSIC_COLUMNS}),
        "MUSIC_VOCAL_SCREEN": SEMANTIC_NA if music is None else SEMANTIC_PENDING,
        "SNR_DB": snr if audio_type == "mixed" and mode != "sequential" else "",
        "OVERLAP_FRACTION": overlap if mode == "partial_overlap" else (1.0 if mode == "concurrent" else 0.0),
        "ORDER": order, "GAP_SECONDS": gap if mode == "sequential" else 0.0,
        "SOURCE_DATASET": "EchoFake_replay+MixFake_FMC_FMA_SONICS",
        "PROVENANCE_MODE": "all_prior_truth_speaker_song_source_and_render_hash_disjoint",
    }
    return result


def validate_reservation(frame: pd.DataFrame, require_semantic: bool = False) -> None:
    if len(frame) != 120 or frame.BASE_ID.nunique() != 120:
        raise ValueError("v9 requires 120 unique base recipes")
    types = frame.AUDIO_TYPE.value_counts()
    if types.to_dict() != {"mixed": 72, "voice": 24, "music": 24}:
        raise ValueError(f"audio-type imbalance: {types.to_dict()}")
    voice = frame.AUDIO_TYPE.eq("voice")
    music = frame.AUDIO_TYPE.eq("music")
    mixed = frame.AUDIO_TYPE.eq("mixed")
    if frame.loc[voice, "VOICE_FAKE"].astype(int).value_counts().to_dict() != {0: 12, 1: 12}:
        raise ValueError("voice-only authenticity imbalance")
    if frame.loc[music, "MUSIC_FAKE"].astype(int).value_counts().to_dict() != {0: 12, 1: 12}:
        raise ValueError("music-only authenticity imbalance")
    expected_cells = {f"{'F' if vf else 'R'}{'F' if mf else 'R'}" for vf, mf in CELLS}
    counts = frame.loc[mixed].groupby(["MIX_MODE", "COMPONENT_CASE"]).size()
    expected_index = {(mode, cell) for mode in MODES for cell in expected_cells}
    if set(counts.index) != expected_index or not counts.eq(PER_MIXED_CELL_LAYOUT).all():
        raise ValueError(f"mixed layout/cell imbalance: {counts.to_dict()}")
    if not frame.loc[voice, ["VOICE_PRESENT", "MUSIC_PRESENT"]].astype(int).eq([1, 0]).all().all():
        raise ValueError("voice-only presence labels invalid")
    if not frame.loc[music, ["VOICE_PRESENT", "MUSIC_PRESENT"]].astype(int).eq([0, 1]).all().all():
        raise ValueError("music-only presence labels invalid")
    if not frame.loc[mixed, ["VOICE_PRESENT", "MUSIC_PRESENT"]].astype(int).eq([1, 1]).all().all():
        raise ValueError("mixed presence labels invalid")
    component_or = frame[["VOICE_FAKE", "MUSIC_FAKE"]].replace("", np.nan).astype(float).fillna(0).max(axis=1)
    if not np.array_equal(component_or.astype(int), frame.FILE_FAKE.astype(int)):
        raise ValueError("FILE_FAKE is not component OR")
    for present, column in ((frame.VOICE_PRESENT.astype(int).eq(1), "VOICE_SOURCE_ID"),
                            (frame.MUSIC_PRESENT.astype(int).eq(1), "MUSIC_SOURCE_ID"),
                            (frame.MUSIC_PRESENT.astype(int).eq(1), "MUSIC_GROUP_ID")):
        values = frame.loc[present, column]
        if values.eq("").any() or values.duplicated().any():
            raise ValueError(f"missing/reused source identity: {column}")
    speakers = frame.loc[frame.VOICE_PRESENT.astype(int).eq(1), "VOICE_SPEAKER"]
    if speakers.eq("").any() or speakers.duplicated().any():
        raise ValueError("target speaker/source reused")
    references = frame.VOICE_REFERENCE_SPEAKER.loc[frame.VOICE_REFERENCE_SPEAKER.ne("")]
    if references.duplicated().any() or set(references) & set(speakers):
        raise ValueError("reference speaker reuse/crossover")
    joint = frame.MUSIC_SOURCE_KIND.eq(JOINT_KIND)
    if len(frame.loc[joint]) != 6 or not frame.loc[joint, "COMPONENT_CASE"].eq("FF").all():
        raise ValueError("six SONICS originals must be FF")
    if not frame.loc[joint, "MIX_MODE"].eq("concurrent").all():
        raise ValueError("direct joint songs are concurrent only")
    if not frame.loc[joint, ["VOICE_FAKE", "MUSIC_FAKE"]].astype(int).eq(1).all().all():
        raise ValueError("joint song components must both be fake")
    if frame.loc[~joint, "MUSIC_PLATFORM"].isin({"suno", "udio"}).any():
        raise ValueError("Suno/Udio vocal song leaked outside direct joint FF")
    replay = frame.loc[frame.VOICE_SOURCE_KIND.eq("echofake_replay"), "VOICE_GENERATOR"].value_counts()
    fake_music = frame.loc[frame.MUSIC_SOURCE_KIND.eq("fakemusiccaps_instrumental"), "MUSIC_GENERATOR"].value_counts()
    if len(replay) < 8 or replay.max() - replay.min() > 1:
        raise ValueError(f"replay generator imbalance: {replay.to_dict()}")
    if len(fake_music) != 5 or fake_music.max() - fake_music.min() > 1:
        raise ValueError(f"fake instrumental generator imbalance: {fake_music.to_dict()}")
    if require_semantic:
        if set(frame.loc[frame.MUSIC_PRESENT.astype(int).eq(1), "MUSIC_VOCAL_SCREEN"]) != {SEMANTIC_PASS}:
            raise ValueError("music semantics incomplete")
        if set(frame.loc[frame.MUSIC_PRESENT.astype(int).eq(0), "MUSIC_VOCAL_SCREEN"]) != {SEMANTIC_NA}:
            raise ValueError("voice-only semantic marker invalid")


def write_requirements(directory: Path, frame: pd.DataFrame) -> None:
    selected = frame.loc[frame.MUSIC_PRESENT.astype(int).eq(1)]
    rows = []
    for row in selected.itertuples(index=False):
        member = row.SONICS_ORIGINAL_ARCHIVE_MEMBER or row.MUSIC_ARCHIVE_MEMBER
        rows.append({"SOURCE_ID": row.MUSIC_SOURCE_ID, "CANONICAL_GROUP": row.MUSIC_GROUP_ID,
                     "GENERATOR": row.MUSIC_GENERATOR, "ARCHIVE_MEMBER": member,
                     "ARCHIVE_TARGET": "MixFake/" + str(member).lstrip("/")})
    pd.DataFrame(rows).to_csv(directory / "source_requirements.csv", index=False)


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
            "truth_open_count": 0, "score_open_count": 0,
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
    for key, name in (("reservation_sha256", "reservation.csv"),
                      ("candidate_pool_sha256", "candidate_pool.csv"),
                      ("protected_inputs_sha256", "protected_inputs.csv")):
        if sha256_file(directory / name) != provenance[key]:
            raise ValueError(f"{name} hash mismatch")
    v7.verify_snapshot(snapshot)
    return frame, candidates, snapshot, provenance


def reserve(args: argparse.Namespace) -> None:
    truths, hashes = all_protected_files()
    protection = v7.build_protection(truths, hashes)
    voice_pools, voice_counts = v8.echo_candidates(args.echo_parquet, protection)
    music_pools, music_counts = v8.standard_music_candidates(
        args.unmixed_details, args.background_protocol, args.musiccaps_metadata, protection,
    )
    joint_pool, joint_counts = v8.sonics_candidates(
        args.unmixed_details, args.background_protocol, args.sonics_metadata, protection,
    )
    real_voice = v8.balanced_voice(voice_pools[0], 48, args.seed + 1)
    used_voice = set().union(*({row["VOICE_SPEAKER"], row["VOICE_CONTENT_ID"],
                               row["VOICE_REFERENCE_ID"], row["VOICE_REFERENCE_SPEAKER"]} - {""}
                              for row in real_voice))
    fake_voice = v8.balanced_voice(voice_pools[1], 42, args.seed + 2, used_voice)
    real_music = v8.balanced_music(music_pools[0], 48, args.seed + 3)
    fake_music = v8.balanced_music(music_pools[1], 42, args.seed + 4)
    joint = [direct_joint(row) for row in v8.balanced_music(joint_pool, 6, args.seed + 5)]
    for row in real_voice: row["VOICE_SOURCE_KIND"] = "echofake_real"
    for row in fake_voice: row["VOICE_SOURCE_KIND"] = "echofake_replay"
    for label in (0, 1):
        for row in music_pools[label]:
            row["MUSIC_SOURCE_KIND"] = "fma_instrumental" if label == 0 else "fakemusiccaps_instrumental"
    for row in real_music: row["MUSIC_SOURCE_KIND"] = "fma_instrumental"
    for row in fake_music: row["MUSIC_SOURCE_KIND"] = "fakemusiccaps_instrumental"
    for row in joint:
        row["VOICE_SOURCE_KIND"] = row["MUSIC_SOURCE_KIND"] = JOINT_KIND

    offsets = Counter()
    records: list[dict] = []
    for label in (0, 1):
        pool = real_voice if label == 0 else fake_voice
        for _ in range(PER_SINGLE_LABEL):
            voice = pool[offsets[f"voice:{label}"]]; offsets[f"voice:{label}"] += 1
            records.append(base_record(len(records), "voice", voice, None, label, "", "voice_only",
                                       "voice_fake" if label else "voice_real", args.seed))
    for label in (0, 1):
        pool = real_music if label == 0 else fake_music
        for _ in range(PER_SINGLE_LABEL):
            music = pool[offsets[f"music:{label}"]]; offsets[f"music:{label}"] += 1
            records.append(base_record(len(records), "music", None, music, "", label, "music_only",
                                       "music_fake" if label else "music_real", args.seed))
    for mode in MODES:
        for voice_fake, music_fake in CELLS:
            cell = f"{'F' if voice_fake else 'R'}{'F' if music_fake else 'R'}"
            for _ in range(PER_MIXED_CELL_LAYOUT):
                if cell == "FF" and mode == "concurrent":
                    item = joint[offsets["joint"]]; offsets["joint"] += 1
                    voice = music = item
                else:
                    vp = fake_voice if voice_fake else real_voice
                    mp = fake_music if music_fake else real_music
                    voice = vp[offsets[f"voice:{voice_fake}"]]; offsets[f"voice:{voice_fake}"] += 1
                    music = mp[offsets[f"music:{music_fake}"]]; offsets[f"music:{music_fake}"] += 1
                records.append(base_record(len(records), "mixed", voice, music, voice_fake,
                                           music_fake, mode, cell, args.seed))
    frame = pd.DataFrame(records)
    validate_reservation(frame)
    candidate_rows = [*voice_pools[0], *voice_pools[1], *music_pools[0], *music_pools[1], *joint_pool]
    candidates = pd.DataFrame(candidate_rows).fillna("")
    provenance = {
        "schema_version": "all_type_blind_v9", "stage": "reserved_pending_semantic_screen",
        "seed": args.seed, "base_rows": len(frame), "rendered_rows_planned": len(frame) * len(CHANNELS),
        "channels": list(CHANNELS), "protected_truth_manifests": len(truths),
        "protected_hash_manifests": len(hashes), "voice_candidate_counts": voice_counts,
        "music_candidate_counts": music_counts, "joint_candidate_counts": joint_counts,
        "semantic_selection_policy": {
            "instrumental": "PANNs any-voice <= 0.20 AND Demucs vocal/mix <= -1.5 dB",
            "joint_song": "PANNs any-voice >= 0.20 AND -20 <= Demucs vocal/mix <= -1.5 dB",
            "replacement": "same source kind, label, and generator; stable hash order",
            "authenticity_outputs_used": False,
        },
    }
    write_reservation(args.output_dir, frame, candidates, protection.snapshot, provenance)
    print(json.dumps({"stage": "reserved", "bases": len(frame), "rendered": len(frame) * 5,
                      "authenticity_inference": False, "score_open_count": 0}, indent=2))


def extract_music(args: argparse.Namespace) -> None:
    import multivolumefile
    import py7zr

    frame, _, _, provenance = load_reservation(args.reservation_dir)
    rows = frame.loc[frame.MUSIC_PRESENT.astype(int).eq(1)]
    members = {str(row.SONICS_ORIGINAL_ARCHIVE_MEMBER or row.MUSIC_ARCHIVE_MEMBER).replace("\\", "/")
               for row in rows.itertuples(index=False)}
    targets = sorted("MixFake/" + member.lstrip("/") for member in members)
    todo = [target for target in targets if not (args.source_root / target).is_file()]
    if todo:
        staging = args.source_root / ".mixfake_extract_v9.partial"
        if staging.exists(): raise FileExistsError(staging)
        staging.mkdir(parents=True)
        try:
            with multivolumefile.open(args.archive_base, "rb") as volume:
                with py7zr.SevenZipFile(volume, "r") as archive:
                    if missing := sorted(set(todo) - set(archive.getnames())):
                        raise ValueError(f"archive members missing: {missing[:5]}")
                    archive.extract(path=staging, targets=todo)
            actual = sorted(path.relative_to(staging).as_posix() for path in staging.rglob("*") if path.is_file())
            if actual != sorted(todo): raise ValueError("unsafe/incomplete archive extraction")
            for target in todo:
                destination = args.source_root / target; destination.parent.mkdir(parents=True, exist_ok=True)
                (staging / target).rename(destination)
            shutil.rmtree(staging)
        except Exception:
            shutil.rmtree(staging, ignore_errors=True); raise
    print(json.dumps({"stage": "music_extracted", "targets": len(targets), "new": len(todo),
                      "reservation_sha256": provenance["reservation_sha256"],
                      "authenticity_inference": False}, indent=2))


def materialize_echo(args: argparse.Namespace) -> None:
    proxy = argparse.Namespace(reservation_dir=args.reservation_dir, echo_parquet=args.echo_parquet,
                               source_root=args.source_root)
    original = v8.load_reservation
    v8.load_reservation = load_reservation
    try: v8.materialize_echo(proxy)
    finally: v8.load_reservation = original


def semantic_screen(args: argparse.Namespace) -> None:
    from audit_prospective_music_vocals import IndependentPannsVocalScreen, demucs_energy
    from separation import HTDemucsSeparator

    frame, _, _, provenance = load_reservation(args.reservation_dir)
    selected = frame.loc[frame.MUSIC_PRESENT.astype(int).eq(1)].drop_duplicates("MUSIC_SOURCE_ID")
    if args.only_source_id:
        selected = selected.loc[selected.MUSIC_SOURCE_ID.isin(set(args.only_source_id))]
        missing = set(args.only_source_id) - set(selected.MUSIC_SOURCE_ID)
        if missing: raise ValueError(f"requested sources absent: {sorted(missing)}")
    panns = IndependentPannsVocalScreen(args.panns_dir, args.device)
    separator = HTDemucsSeparator(device=args.device, repo=args.demucs_repo, shifts=0, overlap=0.25)
    records = []
    for index, row in enumerate(selected.itertuples(index=False), 1):
        member = row.SONICS_ORIGINAL_ARCHIVE_MEMBER or row.MUSIC_ARCHIVE_MEMBER
        path = v7.locate_source(args.source_root, member)
        scores = panns.score(load_audio(path)); scores.update(demucs_energy(separator, path))
        panns_value = float(scores["PANNS_ANY_VOICE_MAX"])
        demucs_value = float(scores["DEMUCS_VOCAL_TO_MIX_DB"])
        if row.MUSIC_SOURCE_KIND == JOINT_KIND:
            passed = panns_value >= args.joint_panns_min and args.joint_demucs_min_db <= demucs_value <= args.demucs_max_db
            rule = "joint_voice_and_accompaniment_presence"
        else:
            passed = panns_value <= args.instrumental_panns_max and demucs_value <= args.demucs_max_db
            rule = "instrumental_voice_absence"
        records.append({"MUSIC_SOURCE_ID": row.MUSIC_SOURCE_ID, "MUSIC_GROUP_ID": row.MUSIC_GROUP_ID,
                        "MUSIC_GENERATOR": row.MUSIC_GENERATOR, "MUSIC_SOURCE_KIND": row.MUSIC_SOURCE_KIND,
                        "MUSIC_AUDIO_SHA256": sha256_file(path), "SEMANTIC_RULE": rule,
                        "PANNS_ANY_VOICE_MAX": panns_value, "DEMUCS_VOCAL_TO_MIX_DB": demucs_value,
                        "INSTRUMENTAL_PANNS_MAX": args.instrumental_panns_max,
                        "DEMUCS_MAX_DB": args.demucs_max_db, "JOINT_PANNS_MIN": args.joint_panns_min,
                        "JOINT_DEMUCS_MIN_DB": args.joint_demucs_min_db,
                        "SEMANTIC_SCREEN_PASS": passed})
        if index % 10 == 0: print(f"semantic {index}/{len(selected)}", flush=True)
    result = pd.DataFrame(records)
    staging, publish = v7.atomic_output_dir(args.output_dir)
    try:
        result.to_csv(staging / "source_scores.csv", index=False)
        summary = {"purpose": "presence semantics only", "sources": len(result),
                   "passes": int(v7.bool_series(result.SEMANTIC_SCREEN_PASS).sum()),
                   "failures": int((~v7.bool_series(result.SEMANTIC_SCREEN_PASS)).sum()),
                   "reservation_sha256": provenance["reservation_sha256"],
                   "authenticity_detector_inference": False, "authenticity_score_computed": False}
        (staging / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", "utf-8")
        publish()
    except Exception:
        shutil.rmtree(staging, ignore_errors=True); raise
    print(json.dumps(summary, indent=2))


def load_semantic(paths: list[Path]) -> pd.DataFrame:
    frames = []
    for order, path in enumerate(paths):
        frame = pd.read_csv(path, dtype=str).fillna("")
        if v7.FORBIDDEN_PREDICTION_COLUMNS & set(frame):
            raise ValueError("authenticity/scoring columns forbidden")
        frame["_order"] = order
        frames.append(frame)
    result = pd.concat(frames, ignore_index=True).sort_values("_order").drop_duplicates("MUSIC_SOURCE_ID", keep="last")
    return result.drop(columns="_order")


def revise(args: argparse.Namespace) -> None:
    frame, candidates, snapshot, provenance = load_reservation(args.reservation_dir)
    scores = load_semantic(args.semantic_scores)
    current = set(frame.loc[frame.MUSIC_PRESENT.astype(int).eq(1), "MUSIC_SOURCE_ID"])
    decisions = dict(zip(scores.MUSIC_SOURCE_ID, v7.bool_series(scores.SEMANTIC_SCREEN_PASS)))
    if current - set(decisions): raise ValueError("missing current semantic decisions")
    failed = sorted(source for source in current if not decisions[source])
    result = frame.copy(); selected_ids = set(result.MUSIC_SOURCE_ID) - {""}; selected_groups = set(result.MUSIC_GROUP_ID) - {""}
    historical = {str(item["old"]) for item in provenance.get("replacement_history", [])}
    excluded = set(decisions) | historical
    replacements, failure_details = [], []
    for source_id in failed:
        index = int(result.index[result.MUSIC_SOURCE_ID.eq(source_id)][0])
        row = result.loc[index]; kind = row.MUSIC_SOURCE_KIND; generator = row.MUSIC_GENERATOR
        score = scores.loc[scores.MUSIC_SOURCE_ID.eq(source_id)].iloc[0]
        failure_details.append({"source_id": source_id, "kind": kind, "generator": generator,
                                "panns": score.PANNS_ANY_VOICE_MAX, "demucs_db": score.DEMUCS_VOCAL_TO_MIX_DB,
                                "rule": score.SEMANTIC_RULE})
        pool_kind = "joint" if kind == JOINT_KIND else "music"
        available = candidates.loc[
            candidates.KIND.eq(pool_kind) & candidates.MUSIC_GENERATOR.eq(generator)
            & ~candidates.MUSIC_SOURCE_ID.isin(selected_ids | excluded)
            & ~candidates.MUSIC_GROUP_ID.isin(selected_groups)
        ].copy()
        if available.empty: raise ValueError(f"no replacement for {source_id}")
        available["_rank"] = [stable_rank(int(provenance["seed"]), "v9-replace", generator, value)
                              for value in available.MUSIC_SOURCE_ID]
        choice = available.sort_values("_rank").iloc[0].to_dict()
        if kind == JOINT_KIND: choice = direct_joint(choice)
        old_group = result.at[index, "MUSIC_GROUP_ID"]
        for column in MUSIC_COLUMNS: result.at[index, column] = choice[column]
        if kind == JOINT_KIND:
            for column in VOICE_COLUMNS: result.at[index, column] = choice[column]
            result.at[index, "VOICE_SOURCE_KIND"] = JOINT_KIND
        result.at[index, "MUSIC_VOCAL_SCREEN"] = SEMANTIC_PENDING
        selected_ids.add(str(choice["MUSIC_SOURCE_ID"])); selected_groups.discard(old_group); selected_groups.add(str(choice["MUSIC_GROUP_ID"]))
        replacements.append({"old": source_id, "new": str(choice["MUSIC_SOURCE_ID"]), "generator": generator})
    if not failed:
        result.loc[result.MUSIC_PRESENT.astype(int).eq(1), "MUSIC_VOCAL_SCREEN"] = SEMANTIC_PASS
    validate_reservation(result, require_semantic=not failed)
    next_provenance = dict(provenance)
    next_provenance.update({
        "stage": "source_semantics_validated" if not failed else "reserved_pending_semantic_screen",
        "parent_reservation_sha256": provenance["reservation_sha256"],
        "semantic_inputs_history": list(provenance.get("semantic_inputs_history", [])) +
            [{"path": str(path.resolve()), "sha256": sha256_file(path)} for path in args.semantic_scores],
        "semantic_failure_history": list(provenance.get("semantic_failure_history", [])) + failure_details,
        "replacement_history": list(provenance.get("replacement_history", [])) + replacements,
    })
    write_reservation(args.output_dir, result, candidates, snapshot, next_provenance)
    print(json.dumps({"stage": next_provenance["stage"], "failures": len(failed),
                      "replacements": replacements}, indent=2))


def protection_from_snapshot(snapshot: pd.DataFrame) -> v7.Protection:
    v7.verify_snapshot(snapshot)
    truths = [ROOT / path for path in snapshot.loc[snapshot.KIND.eq("truth"), "PATH"]]
    hashes = [ROOT / path for path in snapshot.loc[snapshot.KIND.eq("source_hashes"), "PATH"]]
    return v7.build_protection(truths, hashes)


def validate_sources(args: argparse.Namespace) -> None:
    frame, _, snapshot, provenance = load_reservation(args.reservation_dir)
    validate_reservation(frame, require_semantic=True)
    protection = protection_from_snapshot(snapshot)
    variants = set()
    for column in IDENTITY_COLUMNS:
        for value in frame[column]: variants.update(v7.identity_variants(value))
    identity_overlap = sorted(variants & protection.exact)
    group_overlap = sorted((set(frame.MUSIC_GROUP_ID) - {""}) & protection.music_groups)
    records, hash_overlap = [], []
    for row in frame.itertuples(index=False):
        sources = []
        if int(row.VOICE_PRESENT): sources.append(("voice", row.VOICE_SOURCE_ID, row.VOICE_ARCHIVE_MEMBER))
        if int(row.MUSIC_PRESENT): sources.append(("music", row.MUSIC_SOURCE_ID, row.MUSIC_ARCHIVE_MEMBER))
        for kind, source_id, member in sources:
            path = v7.locate_source(args.source_root, member); digest = sha256_file(path)
            if digest.lower() in protection.audio_hashes: hash_overlap.append(f"{kind}:{source_id}")
            records.append({"KIND": kind, "SOURCE_ID": source_id, "ARCHIVE_MEMBER": member,
                            "LOCAL_PATH": str(path), "SHA256": digest, "BYTES": path.stat().st_size})
    checks = {"protected_snapshot_unchanged": True, "identity_disjoint": not identity_overlap,
              "canonical_song_disjoint": not group_overlap, "source_audio_hash_disjoint": not hash_overlap,
              "all_sources_resolve": len(records) == int(frame.VOICE_PRESENT.astype(int).sum() + frame.MUSIC_PRESENT.astype(int).sum()),
              "presence_semantics_pass": set(frame.loc[frame.MUSIC_PRESENT.astype(int).eq(1), "MUSIC_VOCAL_SCREEN"]) == {SEMANTIC_PASS},
              "unscored": provenance.get("authenticity_detector_inference") is False and provenance.get("authenticity_score_computed") is False and provenance.get("score_open_count") == 0}
    staging, publish = v7.atomic_output_dir(args.output_dir)
    try:
        pd.DataFrame(records).to_csv(staging / "source_hashes.csv", index=False)
        report = {"status": "PASS" if all(checks.values()) else "FAIL", "checks": checks,
                  "identity_overlap": identity_overlap, "canonical_song_overlap": group_overlap,
                  "source_hash_overlap": hash_overlap, "reservation_sha256": provenance["reservation_sha256"],
                  "source_hashes_sha256": sha256_file(staging / "source_hashes.csv"),
                  "authenticity_detector_scores_read": False}
        (staging / "validation.json").write_text(json.dumps(report, indent=2) + "\n", "utf-8")
        if report["status"] != "PASS": raise RuntimeError(report)
        publish()
    except Exception:
        shutil.rmtree(staging, ignore_errors=True); raise
    print(json.dumps(report, indent=2))


def render_base(row, roots: list[Path]) -> np.ndarray:
    if row.AUDIO_TYPE == "voice":
        voice = load_audio(v7.locate_source(roots, row.VOICE_ARCHIVE_MEMBER))
        return peak_limit(crop_or_tile(voice, 12 * SR, row.BASE_ID + "|voice"))
    if row.AUDIO_TYPE == "music":
        music = load_audio(v7.locate_source(roots, row.MUSIC_ARCHIVE_MEMBER))
        return peak_limit(crop_or_tile(music, 12 * SR, row.BASE_ID + "|music"))
    if row.MUSIC_SOURCE_KIND == JOINT_KIND:
        joint = load_audio(v7.locate_source(roots, row.MUSIC_ARCHIVE_MEMBER))
        return peak_limit(crop_or_tile(joint, 12 * SR, row.BASE_ID + "|joint"))
    voice = load_audio(v7.locate_source(roots, row.VOICE_ARCHIVE_MEMBER))
    music = load_audio(v7.locate_source(roots, row.MUSIC_ARCHIVE_MEMBER))
    return mix_layout(voice, music, pd.Series(row._asdict()))


def render(args: argparse.Namespace) -> None:
    frame, _, snapshot, provenance = load_reservation(args.reservation_dir)
    validate_reservation(frame, require_semantic=True); v7.verify_snapshot(snapshot)
    validation = json.loads((args.source_validation / "validation.json").read_text("utf-8"))
    if validation.get("status") != "PASS" or validation.get("reservation_sha256") != provenance["reservation_sha256"]:
        raise ValueError("source validation mismatch")
    protection = protection_from_snapshot(snapshot)
    staging, publish = v7.atomic_output_dir(args.output_dir)
    try:
        audio_dir = staging / "audio"; audio_dir.mkdir()
        truth_rows, hashes, protected_overlap = [], [], []
        for index, row in enumerate(frame.itertuples(index=False), 1):
            base = render_base(row, args.source_root)
            for channel in CHANNELS:
                key = int(stable_rank(0, row.BASE_ID, channel)[:16], 16) % (2**32)
                audio = apply_channel(base, channel, ffmpeg=None if channel == "clean" else args.ffmpeg, key=key)
                sample_id = f"{row.BASE_ID}__{channel}"; path = audio_dir / f"{sample_id}.flac"
                sf.write(path, audio, SR, format="FLAC", subtype="PCM_16"); digest = sha256_file(path)
                if digest.lower() in protection.audio_hashes: protected_overlap.append(sample_id)
                hashes.append({"ID": sample_id, "SHA256": digest, "BYTES": path.stat().st_size})
                record = row._asdict(); record.update({"ID": sample_id, "PARENT_ID": row.BASE_ID,
                                                       "CHANNEL": channel, "DURATION": len(audio) / SR})
                truth_rows.append(record)
            if index % 10 == 0: print(f"render {index}/{len(frame)}", flush=True)
        if protected_overlap: raise ValueError(f"rendered audio hash overlap: {protected_overlap[:5]}")
        truth = pd.DataFrame(truth_rows); truth.to_csv(staging / "truth.csv", index=False)
        pd.DataFrame(hashes).to_csv(staging / "audio_hashes.csv", index=False)
        shutil.copy2(args.source_validation / "source_hashes.csv", staging / "source_hashes.csv")
        sample = pd.DataFrame({"ID": truth.ID})
        for column in PREDICTION_COLUMNS: sample[column] = 0.5
        sample.to_csv(staging / "sample_submission.csv", index=False)
        built = dict(provenance); built.update({"stage": "built_unscored", "rendered_rows": len(truth),
            "truth_sha256": sha256_file(staging / "truth.csv"), "audio_hashes_sha256": sha256_file(staging / "audio_hashes.csv"),
            "source_hashes_sha256": sha256_file(staging / "source_hashes.csv"),
            "protected_rendered_hash_overlap": 0, "authenticity_detector_inference": False,
            "authenticity_score_computed": False, "truth_open_count": 0, "score_open_count": 0,
            "ffmpeg": subprocess.run([str(args.ffmpeg), "-version"], check=True, capture_output=True, text=True).stdout.splitlines()[0]})
        (staging / "provenance.json").write_text(json.dumps(built, indent=2) + "\n", "utf-8")
        publish()
    except Exception:
        shutil.rmtree(staging, ignore_errors=True); raise
    print(json.dumps({"stage": "built_unscored", "rows": len(truth), "truth_sha256": built["truth_sha256"]}, indent=2))


def validate_bank(args: argparse.Namespace) -> None:
    frame, _, snapshot, reservation_provenance = load_reservation(args.reservation_dir)
    validate_reservation(frame, require_semantic=True); protection = protection_from_snapshot(snapshot)
    paths = {name: args.bank_dir / filename for name, filename in {
        "truth": "truth.csv", "sample": "sample_submission.csv", "audio": "audio_hashes.csv",
        "source": "source_hashes.csv", "provenance": "provenance.json"}.items()}
    truth = pd.read_csv(paths["truth"], dtype=str).fillna(""); sample = pd.read_csv(paths["sample"])
    hashes = pd.read_csv(paths["audio"], dtype=str); recorded = dict(zip(hashes.ID, hashes.SHA256))
    provenance = json.loads(paths["provenance"].read_text("utf-8")); audio_paths = sorted((args.bank_dir / "audio").glob("*.flac"))
    validation = json.loads((args.source_validation / "validation.json").read_text("utf-8"))
    recipe = truth.merge(frame, on="BASE_ID", suffixes=("_truth", "_reserved"), validate="many_to_one")
    recipe_exact = len(recipe) == 600 and all(recipe[f"{column}_truth"].astype(str).equals(recipe[f"{column}_reserved"].astype(str))
                                            for column in frame.columns if column != "BASE_ID")
    digest_values = [sha256_file(path) for path in audio_paths]
    checks = {
        "unscored_open_zero": provenance.get("authenticity_detector_inference") is False and provenance.get("authenticity_score_computed") is False and provenance.get("score_open_count") == 0,
        "source_validation": validation.get("status") == "PASS",
        "reservation_hash": provenance.get("reservation_sha256") == reservation_provenance["reservation_sha256"] == validation.get("reservation_sha256"),
        "manifest_hashes": sha256_file(paths["truth"]) == provenance.get("truth_sha256") and sha256_file(paths["audio"]) == provenance.get("audio_hashes_sha256") and sha256_file(paths["source"]) == provenance.get("source_hashes_sha256"),
        "rows_ids": len(truth) == truth.ID.nunique() == len(audio_paths) == 600,
        "five_channels": set(truth.CHANNEL) == set(CHANNELS) and truth.groupby("BASE_ID").CHANNEL.nunique().eq(5).all() and truth.groupby("CHANNEL").size().eq(120).all(),
        "recipe_exact": recipe_exact,
        "audio_properties": all(sf.info(path).samplerate == SR and sf.info(path).channels == 1 and 4 <= sf.info(path).duration <= 60 for path in audio_paths),
        "audio_hashes": set(recorded) == set(truth.ID) and all(recorded.get(path.stem) == digest for path, digest in zip(audio_paths, digest_values)),
        "audio_hash_unique": len(set(digest_values)) == 600,
        "prior_hash_disjoint": not (set(digest_values) & protection.audio_hashes),
        "sample_neutral": set(sample.ID.astype(str)) == set(truth.ID) and all(set(pd.to_numeric(sample[c])) == {0.5} for c in PREDICTION_COLUMNS),
        "ads_cps_defined": all(truth[c].astype(int).nunique() == 2 for c in ("FILE_FAKE", "VOICE_PRESENT", "MUSIC_PRESENT")) and truth.loc[truth.VOICE_PRESENT.eq("1"), "VOICE_FAKE"].astype(int).nunique() == 2 and truth.loc[truth.MUSIC_PRESENT.eq("1"), "MUSIC_FAKE"].astype(int).nunique() == 2,
        "no_prediction_artifacts": not any(path.name in {"predictions.csv", "scores.csv", "metrics.json", "submission.csv"} for path in args.bank_dir.rglob("*") if path.is_file()),
    }
    # Pandas reductions return numpy.bool_, which json cannot encode reliably.
    checks = {name: bool(value) for name, value in checks.items()}
    report = {"status": "PASS" if all(checks.values()) else "FAIL", "checks": checks,
              "checks_passed": sum(map(bool, checks.values())), "checks_total": len(checks),
              "base_rows": len(frame), "rendered_rows": len(truth), "truth_sha256": sha256_file(paths["truth"]),
              "reservation_sha256": reservation_provenance["reservation_sha256"],
              "authenticity_detector_scores_read": False, "score_open_count": 0}
    staging, publish = v7.atomic_output_dir(args.output_dir)
    try:
        (staging / "validation.json").write_text(json.dumps(report, indent=2) + "\n", "utf-8")
        if report["status"] != "PASS": raise RuntimeError([key for key, value in checks.items() if not value])
        publish()
    except Exception:
        shutil.rmtree(staging, ignore_errors=True); raise
    print(json.dumps(report, indent=2))


def main() -> None:
    global ID_PREFIX
    parser = argparse.ArgumentParser(description=__doc__); sub = parser.add_subparsers(dest="command", required=True)
    echo_default = [ROOT / "data/external/echofake/dev-00000-of-00001.parquet",
                    *sorted((ROOT / "data/external/echofake").glob("open_set_eval-*.parquet"))]
    p = sub.add_parser("reserve"); p.add_argument("--echo-parquet", type=Path, action="append", default=[])
    p.add_argument("--unmixed-details", type=Path, default=ROOT / "data/external/mixfake/MixFake/protocols/unmixed_details.csv")
    p.add_argument("--background-protocol", type=Path, default=ROOT / "data/external/mixfake/MixFake/protocols/Mixed_and_Back_BackLabel.txt")
    p.add_argument("--musiccaps-metadata", type=Path, default=ROOT / "data/sources/musiccaps_metadata/musiccaps-public.csv")
    p.add_argument("--sonics-metadata", type=Path, default=ROOT / "data/sources/sonics_metadata/fake_songs.csv")
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--id-prefix", default=ID_PREFIX)
    p.add_argument("--output-dir", type=Path, required=True); p.set_defaults(func=reserve)
    p = sub.add_parser("extract-music"); p.add_argument("--reservation-dir", type=Path, required=True)
    p.add_argument("--archive-base", type=Path, default=ROOT / "data/external/mixfake_archive/MixFake.7z")
    p.add_argument("--source-root", type=Path, required=True); p.set_defaults(func=extract_music)
    p = sub.add_parser("materialize-echo"); p.add_argument("--reservation-dir", type=Path, required=True)
    p.add_argument("--echo-parquet", type=Path, action="append", default=[]); p.add_argument("--source-root", type=Path, required=True); p.set_defaults(func=materialize_echo)
    p = sub.add_parser("semantic-screen"); p.add_argument("--reservation-dir", type=Path, required=True)
    p.add_argument("--source-root", type=Path, action="append", required=True); p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--only-source-id", action="append"); p.add_argument("--device", default="cuda:3")
    p.add_argument("--panns-dir", type=Path, default=ROOT / "models/panns"); p.add_argument("--demucs-repo", type=Path, default=ROOT / "models/htdemucs")
    p.add_argument("--instrumental-panns-max", type=float, default=.20); p.add_argument("--demucs-max-db", type=float, default=-1.5)
    p.add_argument("--joint-panns-min", type=float, default=.20); p.add_argument("--joint-demucs-min-db", type=float, default=-20.0); p.set_defaults(func=semantic_screen)
    p = sub.add_parser("revise"); p.add_argument("--reservation-dir", type=Path, required=True)
    p.add_argument("--semantic-scores", type=Path, action="append", required=True); p.add_argument("--output-dir", type=Path, required=True); p.set_defaults(func=revise)
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
    if hasattr(args, "id_prefix"):
        if not re.fullmatch(r"[a-z][a-z0-9_]{2,31}", args.id_prefix):
            raise ValueError("--id-prefix must be a short lowercase identifier")
        ID_PREFIX = args.id_prefix
    if hasattr(args, "echo_parquet") and not args.echo_parquet: args.echo_parquet = echo_default
    args.func(args)


if __name__ == "__main__": main()
