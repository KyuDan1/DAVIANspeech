#!/usr/bin/env python3
"""Build a 1,200-file separator-free factorial audio deepfake evaluation set."""

from __future__ import annotations

import argparse
import hashlib
import subprocess
import tempfile
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
from scipy.signal import butter, resample_poly, sosfilt

try:
    import imageio_ffmpeg
    FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()
except ImportError:
    FFMPEG = "ffmpeg"


SR = 16_000
SPLITS = ["dev", "holdout", "locked"]
MODES = ["voice_only", "music_only", "concurrent", "partial_overlap", "sequential"]
CHANNELS = ["clean_flac", "stereo_wav", "mp3_64k", "ogg_48k", "telephone_flac", "noisy_flac"]
PREDICTION_COLUMNS = [
    "FILE_FAKE_PROB", "VOICE_FAKE_PROB", "MUSIC_FAKE_PROB",
    "VOICE_PRESENT_PROB", "MUSIC_PRESENT_PROB",
]


def component_case(voice_fake: int | None, music_fake: int | None) -> str:
    if voice_fake is None:
        return f"music_{'fake' if music_fake else 'real'}"
    if music_fake is None:
        return f"voice_{'fake' if voice_fake else 'real'}"
    voice = "fake" if voice_fake else "real"
    music = "fake" if music_fake else "real"
    return f"voice_{voice}__music_{music}"


def write_slice_indexes(truth: pd.DataFrame, output_dir: Path) -> None:
    """Write small CSV indexes; audio is never duplicated between slices."""
    slice_dir = output_dir / "slices"
    slice_dir.mkdir(exist_ok=True)
    manifest = []
    for cell, frame in truth.groupby("EVAL_CELL", sort=True):
        relative = Path("slices") / f"all__{cell}.csv"
        frame.to_csv(output_dir / relative, index=False)
        manifest.append({"SPLIT": "all", "EVAL_CELL": cell,
                         "ROWS": len(frame), "TRUTH_CSV": str(relative)})
        for split, split_frame in frame.groupby("SPLIT", sort=True):
            relative = Path("slices") / f"{split}__{cell}.csv"
            split_frame.to_csv(output_dir / relative, index=False)
            manifest.append({"SPLIT": split, "EVAL_CELL": cell,
                             "ROWS": len(split_frame), "TRUTH_CSV": str(relative)})
    pd.DataFrame(manifest).to_csv(output_dir / "slice_manifest.csv", index=False)


def stable_int(*parts: object) -> int:
    return int(hashlib.sha256("|".join(map(str, parts)).encode()).hexdigest()[:16], 16)


def find_audio(directory: Path, sample_id: str) -> Path:
    matches = [path for path in directory.glob(f"{sample_id}.*") if path.is_file()]
    if len(matches) != 1:
        raise ValueError(f"Expected one audio file for {sample_id}, found {matches}")
    return matches[0]


def load_audio(path: Path) -> np.ndarray:
    audio, sample_rate = sf.read(path, dtype="float32", always_2d=True)
    audio = audio.mean(axis=1)
    if sample_rate != SR:
        divisor = np.gcd(sample_rate, SR)
        audio = resample_poly(audio, SR // divisor, sample_rate // divisor).astype(np.float32)
    return audio


def rms(audio: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(audio), dtype=np.float64) + 1e-10))


def music_segment(audio: np.ndarray, samples: int, key: str) -> np.ndarray:
    if audio.size < samples:
        audio = np.tile(audio, int(np.ceil(samples / max(audio.size, 1))))
    span = audio.size - samples
    start = stable_int(key, "music_crop") % (span + 1)
    return audio[start:start + samples]


def peak_limit(audio: np.ndarray) -> np.ndarray:
    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    if peak > 0.98:
        audio = audio * (0.98 / peak)
    return audio.astype(np.float32)


def mix_concurrent(voice: np.ndarray, music: np.ndarray, snr_db: float, key: str) -> np.ndarray:
    length = min(max(voice.size, 8 * SR), 18 * SR)
    voice_canvas = np.zeros(length, dtype=np.float32)
    voice = voice[:length]
    voice_canvas[:voice.size] = voice
    music = music_segment(music, length, key)
    music *= rms(voice) / (rms(music) * 10 ** (snr_db / 20))
    return peak_limit(voice_canvas + music)


def mix_partial(voice: np.ndarray, music: np.ndarray, snr_db: float,
                fraction: float, voice_first: bool, key: str) -> np.ndarray:
    voice = voice[:14 * SR]
    music = music_segment(music, 10 * SR, key)
    overlap = max(int(min(voice.size, music.size) * fraction), 1)
    if voice_first:
        voice_start, music_start = 0, max(voice.size - overlap, 0)
    else:
        music_start, voice_start = 0, max(music.size - overlap, 0)
    length = max(voice_start + voice.size, music_start + music.size)
    voice_canvas = np.zeros(length, dtype=np.float32)
    music_canvas = np.zeros(length, dtype=np.float32)
    voice_canvas[voice_start:voice_start + voice.size] = voice
    scale = rms(voice) / (rms(music) * 10 ** (snr_db / 20))
    music_canvas[music_start:music_start + music.size] = music * scale
    return peak_limit(voice_canvas + music_canvas)


def mix_sequential(voice: np.ndarray, music: np.ndarray, voice_first: bool,
                   gap_seconds: float, key: str) -> np.ndarray:
    voice = voice[:12 * SR]
    music = music_segment(music, 8 * SR, key)
    gap = np.zeros(int(gap_seconds * SR), dtype=np.float32)
    parts = [voice, gap, music] if voice_first else [music, gap, voice]
    return peak_limit(np.concatenate(parts))


def telephone(audio: np.ndarray) -> np.ndarray:
    sos = butter(6, [300, 3400], btype="bandpass", fs=SR, output="sos")
    filtered = sosfilt(sos, audio).astype(np.float32)
    narrow = resample_poly(filtered, 1, 2).astype(np.float32)
    return resample_poly(narrow, 2, 1).astype(np.float32)[:audio.size]


def noisy(audio: np.ndarray, key: str) -> np.ndarray:
    rng = np.random.default_rng(stable_int(key, "noise"))
    white = rng.standard_normal(audio.size).astype(np.float32)
    noise = np.cumsum(white, dtype=np.float64).astype(np.float32)
    noise -= noise.mean()
    noise /= rms(noise)
    target_snr = [18, 24, 30][stable_int(key, "noise_snr") % 3]
    result = audio + noise * rms(audio) / (10 ** (target_snr / 20))
    if stable_int(key, "clip") % 2:
        result = np.tanh(result * 1.35) / np.tanh(1.35)
    return peak_limit(result)


def write_condition(audio: np.ndarray, condition: str, destination_stem: Path) -> Path:
    if condition == "telephone_flac":
        audio = telephone(audio)
    elif condition == "noisy_flac":
        audio = noisy(audio, destination_stem.name)
    if condition == "stereo_wav":
        delay = int(0.007 * SR)
        right = np.pad(audio[:-delay], (delay, 0)) * 0.97
        destination = destination_stem.with_suffix(".wav")
        sf.write(destination, np.stack([audio, right], axis=1), SR, subtype="PCM_16")
    elif condition in {"mp3_64k", "ogg_48k"}:
        suffix = ".mp3" if condition == "mp3_64k" else ".ogg"
        bitrate = "64k" if condition == "mp3_64k" else "48k"
        destination = destination_stem.with_suffix(suffix)
        with tempfile.NamedTemporaryFile(suffix=".wav") as temporary:
            sf.write(temporary.name, audio, SR, subtype="PCM_16")
            subprocess.run([
                FFMPEG, "-nostdin", "-hide_banner", "-loglevel", "error", "-y",
                "-i", temporary.name, "-ac", "1", "-ar", str(SR), "-b:a", bitrate,
                str(destination),
            ], check=True)
    else:
        destination = destination_stem.with_suffix(".flac")
        sf.write(destination, audio, SR, format="FLAC", subtype="PCM_16")
    return destination


class Selector:
    def __init__(self, truth: pd.DataFrame, audio_dir: Path, component: str):
        self.truth = truth.copy()
        self.audio_dir = audio_dir
        self.component = component
        self.counts: dict[tuple, int] = defaultdict(int)

    def take(self, split: str, fake: int, key: str) -> tuple[pd.Series, np.ndarray]:
        label = f"{self.component.upper()}_FAKE"
        candidates = self.truth[
            (self.truth.SPLIT == split) & (pd.to_numeric(self.truth[label]) == fake)
        ].copy()
        if candidates.empty:
            raise RuntimeError(f"No {self.component} split={split} fake={fake}")
        if fake:
            generators = sorted(candidates.GENERATOR.dropna().unique())
            generator = generators[stable_int(key, self.component, "generator") % len(generators)]
            candidates = candidates[candidates.GENERATOR == generator]
        candidates = candidates.sort_values("ID")
        bucket = (split, fake, tuple(candidates.ID))
        index = (stable_int(key, self.component, "row") + self.counts[bucket]) % len(candidates)
        self.counts[bucket] += 1
        row = candidates.iloc[index]
        return row, load_audio(find_audio(self.audio_dir, row.ID))


def validate_bank(truth: pd.DataFrame, group_column: str) -> None:
    if set(truth.SPLIT.dropna().unique()) != set(SPLITS):
        raise ValueError("Bank must contain dev, holdout, and locked splits")
    crossing = truth.groupby(group_column).SPLIT.nunique()
    if (crossing > 1).any():
        raise ValueError(f"{group_column} crosses splits")


def plan_rows() -> list[tuple[str, str, int | None, int | None]]:
    plan = []
    for split in SPLITS:
        for fake in (0, 1):
            plan.extend((split, "voice_only", fake, None) for _ in range(25))
            plan.extend((split, "music_only", None, fake) for _ in range(25))
        for mode in ("concurrent", "partial_overlap", "sequential"):
            for voice_fake in (0, 1):
                for music_fake in (0, 1):
                    plan.extend((split, mode, voice_fake, music_fake) for _ in range(25))
    return plan


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--voice-bank", type=Path, required=True)
    parser.add_argument("--music-bank", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260831)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output_dir}")

    voice_truth = pd.read_csv(args.voice_bank / "truth.csv", dtype=str)
    music_truth = pd.read_csv(args.music_bank / "truth.csv", dtype=str)
    validate_bank(voice_truth, "SPEAKER")
    validate_bank(music_truth, "GROUP_ID")
    voice_selector = Selector(voice_truth, args.voice_bank / "audio", "voice")
    music_selector = Selector(music_truth, args.music_bank / "audio", "music")

    audio_dir = args.output_dir / "audio"
    audio_dir.mkdir(parents=True)
    records = []
    plan = plan_rows()
    if len(plan) != 1200:
        raise AssertionError(len(plan))
    snrs = [-10, -5, 0, 5, 10]
    overlaps = [0.25, 0.50, 0.75]
    for index, (split, mode, voice_fake, music_fake) in enumerate(plan):
        key = f"{args.seed}|{split}|{mode}|{voice_fake}|{music_fake}|{index}"
        voice_row = music_row = None
        if voice_fake is not None:
            voice_row, voice = voice_selector.take(split, voice_fake, key)
        if music_fake is not None:
            music_row, music = music_selector.take(split, music_fake, key)
        snr_db = snrs[stable_int(key, "snr") % len(snrs)]
        voice_first = bool(stable_int(key, "order") % 2)
        overlap = overlaps[stable_int(key, "overlap") % len(overlaps)]
        if mode == "voice_only":
            audio = voice
        elif mode == "music_only":
            length = (8 + stable_int(key, "duration") % 7) * SR
            audio = music_segment(music, length, key)
        elif mode == "concurrent":
            audio = mix_concurrent(voice, music, snr_db, key)
        elif mode == "partial_overlap":
            audio = mix_partial(voice, music, snr_db, overlap, voice_first, key)
        else:
            gap = [0.0, 0.2, 0.5][stable_int(key, "gap") % 3]
            audio = mix_sequential(voice, music, voice_first, gap, key)
        if audio.size < 4 * SR:
            audio = np.pad(audio, (0, 4 * SR - audio.size))
        audio = peak_limit(audio[:60 * SR])

        sample_id = f"factorial_{index:04d}"
        channel = CHANNELS[(index + stable_int(split, args.seed)) % len(CHANNELS)]
        destination = write_condition(audio, channel, audio_dir / sample_id)
        vf = int(voice_fake) if voice_fake is not None else None
        mf = int(music_fake) if music_fake is not None else None
        case = component_case(vf, mf)
        records.append({
            "ID": sample_id, "FILE_FAKE": max(value for value in (vf, mf) if value is not None),
            "VOICE_FAKE": "" if vf is None else vf,
            "MUSIC_FAKE": "" if mf is None else mf,
            "VOICE_PRESENT": int(vf is not None), "MUSIC_PRESENT": int(mf is not None),
            "AUDIO_TYPE": "mixed" if vf is not None and mf is not None else mode.removesuffix("_only"),
            "MIX_MODE": mode, "COMPONENT_CASE": case,
            "EVAL_CELL": f"{mode}__{case}", "CHANNEL": channel, "SPLIT": split,
            "VOICE_SOURCE_ID": "" if voice_row is None else voice_row.ID,
            "VOICE_GENERATOR": "" if voice_row is None else voice_row.GENERATOR,
            "VOICE_SPEAKER": "" if voice_row is None else voice_row.SPEAKER,
            "MUSIC_SOURCE_ID": "" if music_row is None else music_row.ID,
            "MUSIC_GENERATOR": "" if music_row is None else music_row.GENERATOR,
            "MUSIC_GROUP_ID": "" if music_row is None else music_row.GROUP_ID,
            "SNR_DB": "" if mode in {"voice_only", "music_only", "sequential"} else snr_db,
            "OVERLAP_FRACTION": overlap if mode == "partial_overlap" else "",
            "ORDER": ("voice_first" if voice_first else "music_first") if mode in {"partial_overlap", "sequential"} else "",
            "DURATION": round(audio.size / SR, 3), "FILE_EXTENSION": destination.suffix.lower(),
        })

    truth = pd.DataFrame(records)
    truth.to_csv(args.output_dir / "truth.csv", index=False)
    for split in SPLITS:
        truth[truth.SPLIT == split].to_csv(args.output_dir / f"truth_{split}.csv", index=False)
    write_slice_indexes(truth, args.output_dir)
    sample = pd.DataFrame({"ID": truth.ID})
    for column in PREDICTION_COLUMNS:
        sample[column] = 0.0
    sample.to_csv(args.output_dir / "sample_submission.csv", index=False)
    print(truth.groupby(["SPLIT", "MIX_MODE"]).size().unstack().to_string())
    print(truth.groupby(["VOICE_FAKE", "MUSIC_FAKE"], dropna=False).size().to_string())
    print(truth.groupby("CHANNEL").size().to_string())
    print(f"Built {len(truth)} files at {args.output_dir}")


if __name__ == "__main__":
    main()
