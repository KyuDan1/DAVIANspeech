#!/usr/bin/env python3
"""Build a source-disjoint AI-music evaluation set from SONICS and FMA.

The fitted music heads use FakeMusicCaps and Echoes.  This builder deliberately
uses a third fake collection (SONICS Suno/Udio) and a third bona-fide collection
(FMA), then applies the exact same decoding, mono conversion, resampling, crop,
and FLAC encoding to both classes.  This reduces source codec shortcuts while
preserving generator artifacts.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
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


def _decode(payload: bytes) -> tuple[np.ndarray, int]:
    audio, sr = sf.read(BytesIO(payload), dtype="float32", always_2d=True)
    audio = audio.mean(axis=1)
    if sr != SR:
        divisor = np.gcd(sr, SR)
        audio = resample_poly(audio, SR // divisor, sr // divisor).astype(np.float32)
    return audio, SR


def _fixed_crop(audio: np.ndarray, seconds: float, key: str) -> np.ndarray:
    samples = int(seconds * SR)
    if len(audio) < samples:
        repeats = int(np.ceil(samples / max(len(audio), 1)))
        audio = np.tile(audio, repeats)
    span = len(audio) - samples
    digest = int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], "big")
    start = digest % (span + 1)
    return audio[start:start + samples]


def _write_audio(payload: bytes, destination: Path, seconds: float, key: str) -> None:
    audio, _ = _decode(payload)
    audio = _fixed_crop(audio, seconds, key)
    sf.write(destination, audio, SR, format="FLAC", subtype="PCM_16")


def _select_sonics(zip_file: ZipFile, metadata: pd.DataFrame, pairs: int, seed: int):
    archive = {
        Path(name).stem: name for name in zip_file.namelist()
        if name.lower().endswith((".mp3", ".wav", ".flac"))
    }
    rows = metadata[metadata["filename"].isin(archive)].copy()
    rows["split_rank"] = rows["split"].map({"test": 0, "valid": 1, "train": 2})
    # One fake rendition per source song prevents near-duplicate leakage.
    rows = rows.sort_values(["split_rank", "id", "filename"]).drop_duplicates("id")
    rng = np.random.default_rng(seed)
    rows["random_rank"] = rng.random(len(rows))
    per_source = pairs // 2
    selected = []
    for source in ("suno", "udio"):
        candidates = rows[rows["source"] == source].sort_values(
            ["split_rank", "random_rank"]
        )
        selected.append(candidates.head(per_source))
    result = pd.concat(selected).sample(frac=1, random_state=seed).reset_index(drop=True)
    result["archive_name"] = result["filename"].map(archive)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sonics-zip", type=Path, required=True)
    parser.add_argument("--sonics-metadata", type=Path, required=True)
    parser.add_argument("--fma-zip", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--pairs", type=int, default=200)
    parser.add_argument("--seconds", type=float, default=12.0)
    parser.add_argument("--seed", type=int, default=20260829)
    args = parser.parse_args()
    if args.pairs % 2:
        raise ValueError("--pairs must be even for balanced Suno/Udio sampling")

    audio_dir = args.output_dir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    truth_rows = []
    sonics_metadata = pd.read_csv(args.sonics_metadata, low_memory=False)

    with ZipFile(args.sonics_zip) as sonics_zip:
        selected = _select_sonics(
            sonics_zip, sonics_metadata, args.pairs, args.seed
        )
        for index, row in selected.iterrows():
            identifier = f"sdm_fake_{index:04d}"
            _write_audio(
                sonics_zip.read(row["archive_name"]),
                audio_dir / f"{identifier}.flac", args.seconds, identifier,
            )
            truth_rows.append({
                "ID": identifier, "FILE_FAKE": 1, "VOICE_FAKE": "",
                "MUSIC_FAKE": 1, "VOICE_PRESENT": 0, "MUSIC_PRESENT": 1,
                "AUDIO_TYPE": "music", "SOURCE": f"SONICS-{row['source']}",
                "CONDITION": "source_disjoint", "GENERATOR": row["algorithm"],
                "CODEC": "common_flac16k", "GROUP_ID": f"sonics_{int(row['id'])}",
                "SPLIT": "alignment" if index % 2 == 0 else "prospective",
            })

    with ZipFile(args.fma_zip) as fma_zip:
        candidates = sorted(
            name for name in fma_zip.namelist() if name.lower().endswith(".mp3")
        )
        rng = np.random.default_rng(args.seed)
        order = rng.permutation(len(candidates))
        written = 0
        for position in order:
            if written >= args.pairs:
                break
            archive_name = candidates[int(position)]
            identifier = f"sdm_real_{written:04d}"
            try:
                _write_audio(
                    fma_zip.read(archive_name), audio_dir / f"{identifier}.flac",
                    args.seconds, identifier,
                )
            except Exception:
                continue
            truth_rows.append({
                "ID": identifier, "FILE_FAKE": 0, "VOICE_FAKE": "",
                "MUSIC_FAKE": 0, "VOICE_PRESENT": 0, "MUSIC_PRESENT": 1,
                "AUDIO_TYPE": "music", "SOURCE": "FMA",
                "CONDITION": "source_disjoint", "GENERATOR": "real",
                "CODEC": "common_flac16k", "GROUP_ID": Path(archive_name).stem,
                "SPLIT": "alignment" if written % 2 == 0 else "prospective",
            })
            written += 1
    if written != args.pairs:
        raise RuntimeError(f"Only decoded {written}/{args.pairs} FMA tracks")

    truth_rows.sort(key=lambda row: row["ID"])
    truth_path = args.output_dir / "truth.csv"
    with truth_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(truth_rows[0]))
        writer.writeheader()
        writer.writerows(truth_rows)
    with (args.output_dir / "sample_submission.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=["ID", *PREDICTION_COLUMNS])
        writer.writeheader()
        for row in truth_rows:
            writer.writerow({"ID": row["ID"], **{name: 0.0 for name in PREDICTION_COLUMNS}})
    print(f"Wrote {len(truth_rows)} files to {args.output_dir}")


if __name__ == "__main__":
    main()
