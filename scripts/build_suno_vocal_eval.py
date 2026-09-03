#!/usr/bin/env python3
"""Build an evaluation-only set from user-provided Suno songs with vocals.

Every source song remains a single evaluation unit.  A deterministic 60-second
crop matches the competition duration limit, while common mono 16 kHz PCM FLAC
prevents MP3 bitrate from becoming the only usable cue.  This is a positive-only
stress set: report score distributions and recall at fixed thresholds, not EER.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf


SR = 16_000
PREDICTION_COLUMNS = [
    "FILE_FAKE_PROB", "VOICE_FAKE_PROB", "MUSIC_FAKE_PROB",
    "VOICE_PRESENT_PROB", "MUSIC_PRESENT_PROB",
]


def crop(audio: np.ndarray, samples: int, key: str) -> tuple[np.ndarray, int]:
    if len(audio) <= samples:
        return np.pad(audio, (0, samples - len(audio))), 0
    digest = int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], "big")
    # Avoid intros/outros, which are less likely to contain vocals.
    low = min(int(0.15 * len(audio)), len(audio) - samples)
    high = max(low, min(int(0.70 * len(audio)), len(audio) - samples))
    start = low + digest % (high - low + 1)
    return audio[start:start + samples], start


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seconds", type=float, default=60.0)
    args = parser.parse_args()

    sources = sorted(args.input_dir.glob("*.mp3"))
    if not sources:
        raise FileNotFoundError(f"No MP3 files under {args.input_dir}")
    audio_dir = args.output_dir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)

    records = []
    for index, source in enumerate(sources):
        identifier = f"suno_vocal_{index:03d}"
        audio, _ = librosa.load(source, sr=SR, mono=True, dtype=np.float32)
        segment, start = crop(audio, int(args.seconds * SR), source.name)
        sf.write(
            audio_dir / f"{identifier}.flac", segment, SR,
            format="FLAC", subtype="PCM_16",
        )
        records.append({
            "ID": identifier,
            "FILE_FAKE": 1,
            "VOICE_FAKE": 1,
            "MUSIC_FAKE": 1,
            "VOICE_PRESENT": 1,
            "MUSIC_PRESENT": 1,
            "AUDIO_TYPE": "mixed",
            "SOURCE": "user_suno",
            "GENERATOR": "Suno",
            "CONDITION": "vocal_song",
            "CODEC": "common_flac16k",
            "GROUP_ID": source.stem,
            "SOURCE_FILE": source.name,
            "CROP_START_SEC": round(start / SR, 3),
            "DURATION_SEC": args.seconds,
        })

    with (args.output_dir / "truth.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    with (args.output_dir / "sample_submission.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=["ID", *PREDICTION_COLUMNS])
        writer.writeheader()
        for record in records:
            writer.writerow({
                "ID": record["ID"],
                **{column: 0.0 for column in PREDICTION_COLUMNS},
            })
    print(f"Wrote {len(records)} Suno vocal-song examples to {args.output_dir}")


if __name__ == "__main__":
    main()
