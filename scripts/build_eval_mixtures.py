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


def load_seconds(path, seconds):
    audio, _ = librosa.load(path, sr=SR, mono=True, dtype=np.float32)
    samples = int(seconds * SR)
    if len(audio) < samples:
        audio = np.tile(audio, int(np.ceil(samples / max(len(audio), 1))))
    return audio[:samples]


def rms(audio):
    return max(float(np.sqrt(np.mean(audio.astype(np.float64) ** 2))), 1e-6)


def audio_path(directory: Path, sample_id: str) -> Path:
    matches = list((directory / "audio").glob(f"{sample_id}.*"))
    if len(matches) != 1:
        raise ValueError(f"Expected one audio file for {sample_id}: {matches}")
    return matches[0]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--voice-dir", type=Path, required=True)
    parser.add_argument("--music-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--per-combination", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260827)
    parser.add_argument("--exclude-truth", type=Path,
                        help="Mixture truth whose component source IDs must not be reused.")
    parser.add_argument("--id-prefix", default="mixed")
    parser.add_argument(
        "--equal-duration", type=float, default=0.0,
        help="If positive, make both simultaneous and sequential mixtures this "
             "many seconds long (avoids duration leakage).",
    )
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    voice_truth = pd.read_csv(args.voice_dir / "truth.csv")
    music_truth = pd.read_csv(args.music_dir / "truth.csv")
    # Source datasets may themselves contain mixed examples. Components used
    # to synthesize this diagnostic must be isolated single-component files.
    if {"VOICE_PRESENT", "MUSIC_PRESENT"}.issubset(voice_truth.columns):
        voice_truth = voice_truth[
            (voice_truth.VOICE_PRESENT == 1) & (voice_truth.MUSIC_PRESENT == 0)
        ]
    if {"VOICE_PRESENT", "MUSIC_PRESENT"}.issubset(music_truth.columns):
        music_truth = music_truth[
            (music_truth.MUSIC_PRESENT == 1) & (music_truth.VOICE_PRESENT == 0)
        ]
    if args.exclude_truth:
        excluded = pd.read_csv(args.exclude_truth)
        voice_truth = voice_truth[
            ~voice_truth.ID.isin(set(excluded.VOICE_SOURCE_ID.astype(str)))
        ]
        music_truth = music_truth[
            ~music_truth.ID.isin(set(excluded.MUSIC_SOURCE_ID.astype(str)))
        ]
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
                    component_seconds = (
                        args.equal_duration
                        if mode == "simultaneous" and args.equal_duration
                        else args.equal_duration / 2
                        if args.equal_duration
                        else 4.0
                    )
                    voice = load_seconds(
                        audio_path(args.voice_dir, voice_id), component_seconds
                    )
                    music = load_seconds(
                        audio_path(args.music_dir, music_id), component_seconds
                    )
                    length = min(len(voice), len(music))
                    voice, music = voice[:length], music[:length]
                    snr_db = [-6, 0, 6][repetition % 3]
                    voice = voice * (rms(music) / rms(voice)) * (10 ** (snr_db / 20))
                    if mode == "simultaneous":
                        mixed = voice + music
                    else:
                        mixed = np.concatenate([voice, music])
                    peak = max(float(np.max(np.abs(mixed))), 1.0)
                    mixed = (mixed / peak).astype(np.float32)

                    sample_id = f"{args.id_prefix}_{index:04d}"
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
