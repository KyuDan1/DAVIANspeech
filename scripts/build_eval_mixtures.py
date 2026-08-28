"""Build deterministic simultaneous/sequential voice+music diagnostics."""

from __future__ import annotations

import argparse
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
import soundfile as sf

SR = 16_000
PREDICTION_COLUMNS = [
    "FILE_FAKE_PROB", "VOICE_FAKE_PROB", "MUSIC_FAKE_PROB",
    "VOICE_PRESENT_PROB", "MUSIC_PRESENT_PROB",
]


def load_four_seconds(path):
    audio, _ = librosa.load(path, sr=SR, mono=True, dtype=np.float32)
    return audio[:4 * SR]


def rms(audio):
    return max(float(np.sqrt(np.mean(audio.astype(np.float64) ** 2))), 1e-6)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--voice-dir", type=Path, required=True)
    parser.add_argument("--music-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--per-combination", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260827)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    voice_truth = pd.read_csv(args.voice_dir / "truth.csv")
    music_truth = pd.read_csv(args.music_dir / "truth.csv")
    voice_by_label = {label: group.ID.tolist() for label, group in voice_truth.groupby("VOICE_FAKE")}
    music_by_label = {label: group.ID.tolist() for label, group in music_truth.groupby("MUSIC_FAKE")}
    audio_dir = args.output_dir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)

    records = []
    index = 0
    for mode in ["simultaneous", "sequential"]:
        for voice_fake in [0, 1]:
            for music_fake in [0, 1]:
                for repetition in range(args.per_combination):
                    voice_id = rng.choice(voice_by_label[voice_fake])
                    music_id = rng.choice(music_by_label[music_fake])
                    voice = load_four_seconds(args.voice_dir / "audio" / f"{voice_id}.flac")
                    music = load_four_seconds(args.music_dir / "audio" / f"{music_id}.flac")
                    length = min(len(voice), len(music), 4 * SR)
                    voice, music = voice[:length], music[:length]
                    snr_db = [-6, 0, 6][repetition % 3]
                    voice = voice * (rms(music) / rms(voice)) * (10 ** (snr_db / 20))
                    if mode == "simultaneous":
                        mixed = voice + music
                    else:
                        mixed = np.concatenate([voice, music])
                    peak = max(float(np.max(np.abs(mixed))), 1.0)
                    mixed = (mixed / peak).astype(np.float32)

                    sample_id = f"mixed_{index:04d}"
                    sf.write(audio_dir / f"{sample_id}.flac", mixed, SR)
                    records.append({
                        "ID": sample_id, "FILE_FAKE": max(voice_fake, music_fake),
                        "VOICE_FAKE": voice_fake, "MUSIC_FAKE": music_fake,
                        "VOICE_PRESENT": 1, "MUSIC_PRESENT": 1,
                        "AUDIO_TYPE": "mixed", "CONDITION": mode,
                        "SOURCE": "synthetic_mixture", "CODEC": "flac16k",
                        "VOICE_SOURCE_ID": voice_id, "MUSIC_SOURCE_ID": music_id,
                        "SNR_DB": snr_db,
                    })
                    index += 1

    truth = pd.DataFrame(records)
    truth.to_csv(args.output_dir / "truth.csv", index=False)
    sample = pd.DataFrame({"ID": truth.ID})
    for column in PREDICTION_COLUMNS:
        sample[column] = 0.0
    sample.to_csv(args.output_dir / "sample_submission.csv", index=False)
    print(f"Built {len(truth)} mixtures at {args.output_dir}")
    print(truth.groupby(["CONDITION", "VOICE_FAKE", "MUSIC_FAKE"]).size())


if __name__ == "__main__":
    main()
