"""Build a fixed 1,200-file, competition-shaped diagnostic suite.

The source sets are grouped before splitting, so codec variants and mixtures
derived from the same source can never leak across calibration and holdout
splits.  The output intentionally mixes FLAC, MP3, WAV and OGG containers and
includes a narrow-band telephone condition.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
import soundfile as sf

PREDICTION_COLUMNS = [
    "FILE_FAKE_PROB", "VOICE_FAKE_PROB", "MUSIC_FAKE_PROB",
    "VOICE_PRESENT_PROB", "MUSIC_PRESENT_PROB",
]
SEED = 20260827


def run_ffmpeg(source: Path, destination: Path, *options: str) -> None:
    subprocess.run([
        "ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-y",
        "-i", str(source), *options, str(destination),
    ], check=True)


def source_audio(directory: Path, sample_id: str) -> Path:
    matches = list((directory / "audio").glob(f"{sample_id}.*"))
    if len(matches) != 1:
        raise ValueError(f"Expected one audio file for {sample_id}, found {matches}")
    return matches[0]


def assign_splits(frame: pd.DataFrame, seed: int) -> pd.Series:
    """Deterministic stratified split at GROUP_ID granularity."""
    # Semantic music pairs deliberately contain one real and one fake row.
    # Stratify those groups by max(FILE_FAKE), but always keep both sides in
    # the same split.
    groups = frame.groupby("GROUP_ID", as_index=False).agg(
        FILE_FAKE=("FILE_FAKE", "max"), AUDIO_TYPE=("AUDIO_TYPE", "first")
    )
    rng = np.random.default_rng(seed)
    mapping = {}
    for _, block in groups.groupby(["FILE_FAKE", "AUDIO_TYPE"], dropna=False):
        ids = block.GROUP_ID.to_numpy().copy()
        rng.shuffle(ids)
        n = len(ids)
        train_end, validation_end = round(0.6 * n), round(0.8 * n)
        for group_id in ids[:train_end]:
            mapping[group_id] = "calibration"
        for group_id in ids[train_end:validation_end]:
            mapping[group_id] = "validation"
        for group_id in ids[validation_end:]:
            mapping[group_id] = "holdout"
    return frame.GROUP_ID.map(mapping)


def load_sources(directories: list[Path]) -> pd.DataFrame:
    frames = []
    for directory in directories:
        truth = pd.read_csv(directory / "truth.csv", dtype={"ID": str})
        truth["SOURCE_DIR"] = str(directory.resolve())
        if "PAIR_ID" in truth:
            truth["GROUP_ID"] = truth.PAIR_ID.fillna(truth.ID)
        elif {"VOICE_SOURCE_ID", "MUSIC_SOURCE_ID"}.issubset(truth.columns):
            # A mixture itself is one group. Its clean/codec variants must stay
            # together, while separate mixtures remain independent examples.
            truth["GROUP_ID"] = truth.ID
        else:
            truth["GROUP_ID"] = truth.ID
        frames.append(truth)
    frame = pd.concat(frames, ignore_index=True, sort=False)
    required = {
        "ID", "FILE_FAKE", "VOICE_FAKE", "MUSIC_FAKE",
        "VOICE_PRESENT", "MUSIC_PRESENT", "AUDIO_TYPE", "GROUP_ID",
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Missing truth columns: {sorted(missing)}")
    # Presence is known by construction. Older music diagnostics used NA for
    # absent voice, which makes Voice Presence AUC impossible to compute.
    frame["VOICE_PRESENT"] = frame.AUDIO_TYPE.isin(["voice", "mixed"]).astype(int)
    frame["MUSIC_PRESENT"] = frame.AUDIO_TYPE.isin(["music", "mixed"]).astype(int)
    return frame


def select_sources(
    frame: pd.DataFrame, voice_files: int, music_pairs: int, mixed_files: int, seed: int
) -> pd.DataFrame:
    """Select a type-balanced source pool without breaking semantic pairs."""
    rng = np.random.default_rng(seed)
    selected = []

    voice = frame[frame.AUDIO_TYPE == "voice"]
    if voice_files:
        per_class = voice_files // 2
        if voice_files % 2:
            raise ValueError("--voice-files must be even")
        voice = pd.concat([
            block.sample(n=per_class, random_state=seed + int(label))
            for label, block in voice.groupby("FILE_FAKE")
        ])
    selected.append(voice)

    music = frame[frame.AUDIO_TYPE == "music"]
    if music_pairs:
        pair_ids = music.GROUP_ID.drop_duplicates().to_numpy()
        rng.shuffle(pair_ids)
        chosen = set(pair_ids[:music_pairs])
        if len(chosen) < music_pairs:
            raise ValueError(f"Need {music_pairs} music pairs, found {len(pair_ids)}")
        music = music[music.GROUP_ID.isin(chosen)]
    selected.append(music)

    mixed = frame[frame.AUDIO_TYPE == "mixed"]
    if mixed_files:
        if len(mixed) < mixed_files:
            raise ValueError(f"Need {mixed_files} mixtures, found {len(mixed)}")
        # The source builder is balanced across mode and component labels;
        # preserve that ordering when the full requested count is available.
        mixed = mixed.sample(n=mixed_files, random_state=seed)
    selected.append(mixed)
    return pd.concat(selected, ignore_index=True, sort=False)


def write_noise_variant(source: Path, destination: Path, seed: int) -> None:
    audio, _ = librosa.load(source, sr=16_000, mono=True, dtype=np.float32)
    rng = np.random.default_rng(seed)
    noise = rng.standard_normal(len(audio)).astype(np.float32)
    signal_rms = max(float(np.sqrt(np.mean(audio.astype(np.float64) ** 2))), 1e-5)
    noise_rms = max(float(np.sqrt(np.mean(noise.astype(np.float64) ** 2))), 1e-5)
    # 15 dB SNR: audible channel noise without destroying the source.
    mixed = audio + noise * signal_rms / noise_rms * (10 ** (-15 / 20))
    mixed /= max(float(np.max(np.abs(mixed))), 1.0)
    sf.write(destination, mixed, 16_000, format="OGG", subtype="VORBIS")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--voice-dir", type=Path, required=True)
    parser.add_argument("--music-dir", type=Path, required=True)
    parser.add_argument("--mixed-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--voice-files", type=int, default=0,
                        help="Balanced voice files to retain; 0 keeps all.")
    parser.add_argument("--music-pairs", type=int, default=0,
                        help="Semantic real/fake pairs to retain; 0 keeps all.")
    parser.add_argument("--mixed-files", type=int, default=0,
                        help="Mixtures to retain; 0 keeps all.")
    args = parser.parse_args()

    frame = load_sources([args.voice_dir, args.music_dir, args.mixed_dir])
    frame = select_sources(
        frame, args.voice_files, args.music_pairs, args.mixed_files, args.seed
    )
    frame["SPLIT"] = assign_splits(frame, args.seed)
    if args.output_dir.exists():
        raise FileExistsError(
            f"Refusing to overwrite fixed suite: {args.output_dir}. "
            "Choose a new versioned directory."
        )
    audio_dir = args.output_dir / "audio"
    audio_dir.mkdir(parents=True)

    records = []
    for row_index, row in frame.iterrows():
        source_dir = Path(row.SOURCE_DIR)
        source = source_audio(source_dir, row.ID)
        variants = [
            ("clean", "flac"),
            ("mp3_64k", "mp3"),
            ("telephone", "wav"),
        ]
        for condition, extension in variants:
            sample_id = f"suite_{row_index:04d}_{condition}"
            destination = audio_dir / f"{sample_id}.{extension}"
            if condition == "clean":
                run_ffmpeg(source, destination, "-ac", "1", "-ar", "16000")
            elif condition == "mp3_64k":
                run_ffmpeg(source, destination, "-ac", "1", "-ar", "16000", "-b:a", "64k")
            else:
                run_ffmpeg(
                    source, destination, "-ac", "1", "-ar", "16000", "-af",
                    "highpass=f=300,lowpass=f=3400,aresample=8000,aresample=16000",
                )
            record = row.drop(labels=["SOURCE_DIR"]).to_dict()
            record.update({
                "ID": sample_id, "BASE_ID": row.ID, "CHANNEL": condition,
                "FORMAT": extension,
            })
            records.append(record)

    missing_to_1200 = 1200 - len(records)
    if missing_to_1200 < 0:
        raise ValueError(
            f"Three variants create {len(records)} files; select at most 400 sources"
        )
    if missing_to_1200:
        # Add balanced noise variants only when fewer than 400 base sources
        # were selected. Each source can receive at most one extra variant.
        if missing_to_1200 > len(frame):
            raise ValueError("Not enough sources to reach 1,200 with one noise variant")
        pieces = []
        per_label = missing_to_1200 // 2
        for label, block in frame.groupby("FILE_FAKE"):
            pieces.append(block.sample(n=per_label, random_state=args.seed + int(label)))
        noisy = pd.concat(pieces)
        remainder = missing_to_1200 - len(noisy)
        if remainder:
            available = frame[~frame.ID.isin(noisy.ID)]
            noisy = pd.concat([noisy, available.sample(n=remainder, random_state=args.seed)])
        for noise_index, (row_index, row) in enumerate(noisy.sort_values("ID").iterrows()):
            source = source_audio(Path(row.SOURCE_DIR), row.ID)
            sample_id = f"suite_{row_index:04d}_noise15"
            destination = audio_dir / f"{sample_id}.ogg"
            write_noise_variant(source, destination, args.seed + noise_index)
            record = row.drop(labels=["SOURCE_DIR"]).to_dict()
            record.update({
                "ID": sample_id, "BASE_ID": row.ID, "CHANNEL": "noise15",
                "FORMAT": "ogg",
            })
            records.append(record)

    truth = pd.DataFrame(records)
    if len(truth) != 1200 or truth.ID.duplicated().any():
        raise AssertionError(f"Invalid suite size/IDs: {len(truth)}")
    truth.to_csv(args.output_dir / "truth.csv", index=False)
    sample = pd.DataFrame({"ID": truth.ID})
    for column in PREDICTION_COLUMNS:
        sample[column] = 0.0
    sample.to_csv(args.output_dir / "sample_submission.csv", index=False)
    shutil.copy2(__file__, args.output_dir / "BUILD_SCRIPT.py")

    print(f"Built {len(truth)} files at {args.output_dir}")
    print(truth.groupby(["SPLIT", "AUDIO_TYPE", "FILE_FAKE"]).size())
    print(truth.groupby(["CHANNEL", "FORMAT"]).size())


if __name__ == "__main__":
    main()
