#!/usr/bin/env python3
"""Build separator-free, source-matched instrumental pairs from Echoes/FMA.

Echoes generated several variants conditioned on original FMA tracks.  This
builder resolves those titles back to the FMA-small archive, assigns each
original track to exactly one split, and keeps only original/generated clips
that a frozen presence model judges to have no voice.  Previously evaluated
Echoes source songs and FMA track IDs are excluded.
"""

from __future__ import annotations

import argparse
import hashlib
import re
import unicodedata
from io import BytesIO
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd
import soundfile as sf
from scipy.signal import resample_poly


SR = 16_000
PREDICTION_COLUMNS = [
    "FILE_FAKE_PROB", "VOICE_FAKE_PROB", "MUSIC_FAKE_PROB",
    "VOICE_PRESENT_PROB", "MUSIC_PRESENT_PROB",
]


def normalize_key(value: str) -> str:
    ascii_text = unicodedata.normalize("NFKD", str(value)).encode(
        "ascii", "ignore").decode().lower()
    return re.sub(r"[^a-z0-9]+", "", ascii_text)


def decode(payload: bytes) -> np.ndarray:
    audio, sample_rate = sf.read(BytesIO(payload), dtype="float32", always_2d=True)
    audio = audio.mean(axis=1)
    if sample_rate != SR:
        divisor = np.gcd(sample_rate, SR)
        audio = resample_poly(audio, SR // divisor, sample_rate // divisor).astype(np.float32)
    return audio


def crop(audio: np.ndarray, seconds: float, key: str) -> np.ndarray:
    samples = int(seconds * SR)
    if audio.size < samples:
        audio = np.pad(audio, (0, samples - audio.size))
    span = audio.size - samples
    start = int(hashlib.sha256(key.encode()).hexdigest()[:8], 16) % (span + 1)
    return audio[start:start + samples]


def split_for(group: str, splits: list[str], seed: int) -> str:
    value = int(hashlib.sha256(f"{seed}|{group}".encode()).hexdigest()[:8], 16)
    return splits[value % len(splits)]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--echoes-zip", type=Path, required=True)
    parser.add_argument("--fma-zip", type=Path, required=True)
    parser.add_argument("--fma-metadata-zip", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--exclude-truth", type=Path, nargs="*", default=[])
    parser.add_argument("--panns-dir", type=Path, required=True)
    parser.add_argument("--max-voice-prob", type=float, default=0.20)
    parser.add_argument("--min-music-prob", type=float, default=0.10)
    parser.add_argument("--extra-real-per-split", type=int, default=50)
    parser.add_argument("--seconds", type=float, default=12.0)
    parser.add_argument("--splits", nargs="+", default=["dev", "holdout", "locked"])
    parser.add_argument("--seed", type=int, default=20260831)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output_dir}")

    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from presence import PannsPresence
    presence = PannsPresence(args.panns_dir, device="cuda")

    excluded_echoes: set[str] = set()
    excluded_fma: set[int] = set()
    for path in args.exclude_truth:
        truth = pd.read_csv(path, dtype=str)
        if "GROUP_ID" not in truth:
            continue
        for value in truth.GROUP_ID.dropna():
            if value.startswith("sonics_"):
                continue
            if value.isdigit() or value.startswith("fma_"):
                excluded_fma.add(int(value.removeprefix("fma_")))
            else:
                excluded_echoes.add(value)

    with ZipFile(args.fma_zip) as fma_archive:
        fma_members = {
            int(Path(name).stem): name for name in fma_archive.namelist()
            if name.lower().endswith(".mp3")
        }
        with ZipFile(args.fma_metadata_zip) as metadata_archive:
            tracks = pd.read_csv(
                metadata_archive.open("fma_metadata/tracks.csv"), index_col=0,
                header=[0, 1], low_memory=False,
            )
        tracks = tracks.loc[tracks.index.intersection(fma_members)]
        title_to_id = {
            normalize_key(f"{row[('track', 'title')]} - {row[('artist', 'name')]} "): int(track_id)
            for track_id, row in tracks.iterrows()
        }

        with ZipFile(args.echoes_zip) as echoes_archive:
            manifest = pd.read_csv(echoes_archive.open("Echoes/dataset_manifest.csv"))
            manifest["track_id"] = manifest.original_audio.map(
                lambda value: title_to_id.get(normalize_key(value)))
            manifest = manifest[manifest.track_id.notna()].copy()
            manifest.track_id = manifest.track_id.astype(int)
            manifest = manifest[
                ~manifest.original_audio.astype(str).isin(excluded_echoes)
                & ~manifest.track_id.isin(excluded_fma)
            ]
            print(f"Matched unseen candidates: {len(manifest)} fakes / "
                  f"{manifest.track_id.nunique()} source tracks", flush=True)

            audio_dir = args.output_dir / "audio"
            audio_dir.mkdir(parents=True)
            rows: list[dict] = []
            accepted_groups = 0
            for track_id, block in manifest.groupby("track_id", sort=True):
                original_name = str(block.original_audio.iloc[0])
                group_id = f"fma_{track_id:06d}"
                split = split_for(group_id, args.splits, args.seed)
                real_audio = crop(
                    decode(fma_archive.read(fma_members[track_id])),
                    args.seconds, f"{group_id}|real",
                )
                real_voice, real_music = presence.predict(real_audio)
                if real_voice > args.max_voice_prob or real_music < args.min_music_prob:
                    continue

                fake_items = []
                for row in block.sort_values(["generator", "path_in_dataset"]).itertuples(index=False):
                    try:
                        fake_audio = crop(
                            decode(echoes_archive.read("Echoes/" + row.path_in_dataset)),
                            args.seconds, f"{group_id}|{row.generator}|{row.path_in_dataset}",
                        )
                        fake_voice, fake_music = presence.predict(fake_audio)
                    except Exception as error:
                        print(f"skip decode {row.path_in_dataset}: {error}", flush=True)
                        continue
                    if fake_voice <= args.max_voice_prob and fake_music >= args.min_music_prob:
                        fake_items.append((row, fake_audio, fake_voice, fake_music))
                if not fake_items:
                    continue

                real_id = f"efp_real_{track_id:06d}"
                sf.write(audio_dir / f"{real_id}.flac", real_audio, SR,
                         format="FLAC", subtype="PCM_16")
                common = {
                    "VOICE_FAKE": "", "VOICE_PRESENT": 0, "MUSIC_PRESENT": 1,
                    "AUDIO_TYPE": "music", "CONDITION": "source_matched_instrumental",
                    "GROUP_ID": group_id, "FMA_TRACK_ID": track_id,
                    "ORIGINAL_AUDIO": original_name, "SPLIT": split,
                    "CODEC": "common_flac16k", "DURATION": args.seconds,
                }
                rows.append({
                    "ID": real_id, "FILE_FAKE": 0, "MUSIC_FAKE": 0,
                    "SOURCE": "FMA", "GENERATOR": "real",
                    "VOICE_SCREEN": round(real_voice, 6),
                    "MUSIC_SCREEN": round(real_music, 6), **common,
                })
                for index, (row, fake_audio, fake_voice, fake_music) in enumerate(fake_items):
                    fake_id = f"efp_fake_{track_id:06d}_{row.generator}_{index:02d}"
                    sf.write(audio_dir / f"{fake_id}.flac", fake_audio, SR,
                             format="FLAC", subtype="PCM_16")
                    rows.append({
                        "ID": fake_id, "FILE_FAKE": 1, "MUSIC_FAKE": 1,
                        "SOURCE": "Echoes", "GENERATOR": row.generator,
                        "GENERATION_TYPE": row.type, "GENRE": row.genre,
                        "VOICE_SCREEN": round(fake_voice, 6),
                        "MUSIC_SCREEN": round(fake_music, 6), **common,
                    })
                accepted_groups += 1
                print(f"accepted {group_id} ({split}): {len(fake_items)} fakes", flush=True)

        # The source-matched subset has many generated variants per original
        # but relatively few bona-fide groups.  Add unseen, voice-screened FMA
        # instrumentals so EER is not dominated by repeated real tracks.
        occupied_ids = set(manifest.track_id.astype(int))
        eligible_ids = [
            track_id for track_id in fma_members
            if track_id not in occupied_ids and track_id not in excluded_fma
        ]
        eligible_ids.sort(key=lambda track_id: hashlib.sha256(
            f"{args.seed}|extra-real|{track_id}".encode()).hexdigest())
        extra_counts = {split: 0 for split in args.splits}
        for track_id in eligible_ids:
            if all(count >= args.extra_real_per_split for count in extra_counts.values()):
                break
            group_id = f"fma_{track_id:06d}"
            split = split_for(group_id, args.splits, args.seed)
            if extra_counts[split] >= args.extra_real_per_split:
                continue
            try:
                real_audio = crop(
                    decode(fma_archive.read(fma_members[track_id])),
                    args.seconds, f"{group_id}|extra-real",
                )
                real_voice, real_music = presence.predict(real_audio)
            except Exception as error:
                print(f"skip extra real {track_id}: {error}", flush=True)
                continue
            if real_voice > args.max_voice_prob or real_music < args.min_music_prob:
                continue
            track = tracks.loc[track_id]
            real_id = f"efp_extra_real_{track_id:06d}"
            sf.write(audio_dir / f"{real_id}.flac", real_audio, SR,
                     format="FLAC", subtype="PCM_16")
            rows.append({
                "ID": real_id, "FILE_FAKE": 0, "MUSIC_FAKE": 0,
                "VOICE_FAKE": "", "VOICE_PRESENT": 0, "MUSIC_PRESENT": 1,
                "AUDIO_TYPE": "music", "SOURCE": "FMA", "GENERATOR": "real",
                "CONDITION": "voice_screened_instrumental",
                "GROUP_ID": group_id, "FMA_TRACK_ID": track_id,
                "ORIGINAL_AUDIO": f"{track[('track', 'title')]} - {track[('artist', 'name')]}",
                "SPLIT": split, "CODEC": "common_flac16k",
                "DURATION": args.seconds, "VOICE_SCREEN": round(real_voice, 6),
                "MUSIC_SCREEN": round(real_music, 6),
            })
            extra_counts[split] += 1
            print(f"extra real {split} {extra_counts[split]}/{args.extra_real_per_split}",
                  flush=True)
        if not all(count == args.extra_real_per_split for count in extra_counts.values()):
            raise RuntimeError(f"Insufficient extra real instrumentals: {extra_counts}")

    truth = pd.DataFrame(rows).sort_values("ID")
    if truth.empty or truth[truth.FILE_FAKE == 0].GROUP_ID.nunique() < 15:
        raise RuntimeError("Too few source-disjoint instrumental groups survived")
    truth.to_csv(args.output_dir / "truth.csv", index=False)
    sample = pd.DataFrame({"ID": truth.ID})
    for column in PREDICTION_COLUMNS:
        sample[column] = 0.0
    sample.to_csv(args.output_dir / "sample_submission.csv", index=False)
    print(truth.groupby(["SPLIT", "FILE_FAKE", "GENERATOR"]).size().to_string())
    print(f"Built {len(truth)} files from {truth.GROUP_ID.nunique()} groups "
          f"({accepted_groups} source-matched) at {args.output_dir}")


if __name__ == "__main__":
    main()
