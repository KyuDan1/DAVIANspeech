#!/usr/bin/env python3
"""Reserve and extract protected MixFake voice+music mixtures for v93.

This operates only on protocol metadata and raw audio.  No detector score is
computed.  Existing truth/hash manifests are snapshotted before reservation;
their voice identities and canonical music-song groups are excluded.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from build_codec_mixed_blind_v7 import (  # noqa: E402
    build_protection, discover_protected_files, identity_variants,
    verify_snapshot,
)
from build_prospective_mixed_phone_v3 import (  # noqa: E402
    canonical_music_group, sha256_file, stable_rank,
)

DETAILS = ROOT / "data/external/mixfake/MixFake/protocols/mixed_details.csv"
ARCHIVE_BASE = ROOT / "data/external/mixfake_archive/MixFake.7z"


def generator(path: str, authentic: str) -> str:
    if authentic == "bonafide":
        return "FMA"
    normalized = str(path).replace("\\", "/")
    if "/FakeMusicCaps/" in normalized:
        return "FakeMusicCaps:" + normalized.split("/FakeMusicCaps/", 1)[1].split("/", 1)[0]
    lower = Path(normalized).name.lower()
    if "_suno_" in lower:
        return "SONICS:Suno"
    if "_udio_" in lower:
        return "SONICS:Udio"
    return "unknown_fake_music"


def archive_member(row) -> str:
    return f"MixFake/mixed_dataset/{row.split}/audio/{Path(row.syn_file).name}"


def frame_for_manifest(frame: pd.DataFrame, role: str) -> pd.DataFrame:
    result = pd.DataFrame({
        "ID": frame.syn_file.map(lambda value: Path(str(value)).stem),
        "ARCHIVE_MEMBER": [archive_member(row) for row in frame.itertuples(index=False)],
        "ROLE": role,
        "FILE_FAKE": frame.syn_authenticity.eq("spoof").astype(int),
        "VOICE_PRESENT": 1,
        "MUSIC_PRESENT": 1,
        "VOICE_FAKE": frame.fore_authenticity.eq("spoof").astype(int),
        "MUSIC_FAKE": frame.back_authenticity.eq("spoof").astype(int),
        "COMPONENT_CASE": (
            frame.fore_authenticity.map({"bonafide": "R", "spoof": "F"})
            + frame.back_authenticity.map({"bonafide": "R", "spoof": "F"})
        ),
        "VOICE_SOURCE_ID": frame.fore_file.map(lambda value: Path(str(value)).stem),
        "MUSIC_GROUP_ID": frame.MUSIC_GROUP_ID,
        "MUSIC_GENERATOR": [generator(path, auth) for path, auth in
                            zip(frame.back_file, frame.back_authenticity)],
        "SNR_DB": frame.snr_db,
        "SOURCE_SPLIT": frame.split,
    })
    if result.ID.duplicated().any():
        raise ValueError(f"duplicate IDs in {role}")
    if not result.FILE_FAKE.equals(result.VOICE_FAKE | result.MUSIC_FAKE):
        raise ValueError("File labels do not equal component OR")
    return result.sort_values("ID").reset_index(drop=True)


def reserve(output: Path, locked_per_cell: int, seed: int) -> None:
    if output.exists():
        raise FileExistsError(output)
    truths, hashes = discover_protected_files()
    protection = build_protection(truths, hashes)
    details = pd.read_csv(DETAILS, low_memory=False)
    details = details.loc[details.back_type.eq("music")].copy()
    details["MUSIC_GROUP_ID"] = details.back_file.map(canonical_music_group)
    details["PROTECTED_VOICE"] = details.fore_file.map(
        lambda value: not identity_variants(value).isdisjoint(protection.exact)
    )
    details["PROTECTED_MUSIC"] = details.MUSIC_GROUP_ID.isin(protection.music_groups)
    eligible = details.loc[~details.PROTECTED_VOICE & ~details.PROTECTED_MUSIC].copy()

    train = eligible.loc[eligible.split.eq("train")].copy()
    train_music = set(train.MUSIC_GROUP_ID)
    development = eligible.loc[
        eligible.split.eq("dev") & ~eligible.MUSIC_GROUP_ID.isin(train_music)
    ].copy()
    development_music = set(development.MUSIC_GROUP_ID)
    pool = eligible.loc[
        eligible.split.eq("eval")
        & ~eligible.MUSIC_GROUP_ID.isin(train_music | development_music)
    ].copy()

    used_voice: set[str] = set()
    used_music: set[str] = set()
    locked_parts = []
    for voice in ("bonafide", "spoof"):
        for music in ("bonafide", "spoof"):
            block = pool.loc[
                pool.fore_authenticity.eq(voice) & pool.back_authenticity.eq(music)
            ].copy()
            block["RANK"] = [stable_rank(seed, voice, music, item)
                             for item in block.syn_file]
            selected = []
            for row in block.sort_values("RANK").itertuples(index=False):
                voice_id = Path(str(row.fore_file)).stem
                music_id = str(row.MUSIC_GROUP_ID)
                if voice_id in used_voice or music_id in used_music:
                    continue
                selected.append(row._asdict())
                used_voice.add(voice_id)
                used_music.add(music_id)
                if len(selected) == locked_per_cell:
                    break
            if len(selected) != locked_per_cell:
                raise ValueError(f"insufficient unique locked rows for {voice}/{music}")
            locked_parts.append(pd.DataFrame(selected))
    locked = pd.concat(locked_parts, ignore_index=True)

    # Cross-role component identities are strictly disjoint.
    role_frames = {"train": train, "development": development, "locked": locked}
    for left, right in (("train", "development"), ("train", "locked"),
                        ("development", "locked")):
        if set(role_frames[left].fore_file.map(lambda x: Path(str(x)).stem)) & set(
                role_frames[right].fore_file.map(lambda x: Path(str(x)).stem)):
            raise ValueError(f"voice overlap: {left}/{right}")
        if set(role_frames[left].MUSIC_GROUP_ID) & set(role_frames[right].MUSIC_GROUP_ID):
            raise ValueError(f"music overlap: {left}/{right}")

    output.mkdir(parents=True)
    protection.snapshot.to_csv(output / "protected_inputs.csv", index=False)
    manifests = {}
    for role, frame in role_frames.items():
        manifest = frame_for_manifest(frame, role)
        path = output / f"{role}.csv"
        manifest.to_csv(path, index=False)
        manifests[role] = {"rows": len(manifest), "sha256": sha256_file(path),
                           "component_cases": manifest.COMPONENT_CASE.value_counts().sort_index().to_dict(),
                           "music_generators": manifest.MUSIC_GENERATOR.value_counts().sort_index().to_dict()}
    report = {
        "schema": "mixfake_alltype_v93_reservation", "status": "reserved_unscored",
        "seed": seed, "locked_per_cell": locked_per_cell,
        "details_sha256": sha256_file(DETAILS), "protected_truths": len(truths),
        "protected_hash_manifests": len(hashes), "manifests": manifests,
        "source_overlap": {"train_development": 0, "train_locked": 0,
                           "development_locked": 0},
        "authenticity_detector_scores_read": False,
        "locked_selection_allowed": False,
        "automatic_submission_allowed": False,
    }
    (output / "reservation.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


def extract(reservation: Path, destination: Path) -> None:
    report = json.loads((reservation / "reservation.json").read_text())
    if report["status"] != "reserved_unscored" or report["authenticity_detector_scores_read"]:
        raise ValueError("invalid reservation")
    verify_snapshot(pd.read_csv(reservation / "protected_inputs.csv"))
    if destination.exists():
        raise FileExistsError(destination)
    staging = destination.with_name(destination.name + ".partial")
    if staging.exists():
        raise FileExistsError(staging)
    staging.mkdir(parents=True)
    targets = []
    for role in ("train", "development", "locked"):
        path = reservation / f"{role}.csv"
        if sha256_file(path) != report["manifests"][role]["sha256"]:
            raise ValueError(f"reservation changed: {role}")
        targets.extend(pd.read_csv(path, dtype=str).ARCHIVE_MEMBER.tolist())
    if len(targets) != len(set(targets)):
        raise ValueError("duplicate archive members")

    import multivolumefile
    import py7zr
    started = time.monotonic()
    with multivolumefile.MultiVolume(ARCHIVE_BASE, mode="rb", ext_digits=3) as stream:
        with py7zr.SevenZipFile(stream, mode="r") as archive:
            available = set(archive.getnames())
            missing = sorted(set(targets) - available)
            if missing:
                raise FileNotFoundError(f"archive misses {missing[:3]}")
            archive.extract(path=staging, targets=targets)
    records = []
    for index, member in enumerate(targets):
        path = staging / member
        if not path.is_file() or path.stat().st_size <= 44:
            raise ValueError(f"invalid extracted audio: {member}")
        records.append({"ARCHIVE_MEMBER": member, "BYTES": path.stat().st_size,
                        "SHA256": sha256_file(path)})
        if index % 1000 == 0:
            print(json.dumps({"validated": index + 1, "total": len(targets),
                              "seconds": time.monotonic() - started}), flush=True)
    pd.DataFrame(records).to_csv(staging / "audio_hashes.csv", index=False)
    completed = {
        "status": "complete", "files": len(records),
        "seconds": time.monotonic() - started,
        "audio_hashes_sha256": sha256_file(staging / "audio_hashes.csv"),
        "reservation_sha256": sha256_file(reservation / "reservation.json"),
        "authenticity_detector_scores_read": False,
    }
    (staging / "extraction.json").write_text(json.dumps(completed, indent=2) + "\n")
    staging.rename(destination)
    print(json.dumps(completed, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="mode", required=True)
    create = sub.add_parser("reserve")
    create.add_argument("--output", type=Path, required=True)
    create.add_argument("--locked-per-cell", type=int, default=120)
    create.add_argument("--seed", type=int, default=20260923)
    unpack = sub.add_parser("extract")
    unpack.add_argument("--reservation", type=Path, required=True)
    unpack.add_argument("--destination", type=Path, required=True)
    args = parser.parse_args()
    if args.mode == "reserve":
        reserve(args.output, args.locked_per_cell, args.seed)
    else:
        extract(args.reservation, args.destination)


if __name__ == "__main__":
    main()
