#!/usr/bin/env python3
"""Extract train-free HTDemucs stem statistics for component presence audits."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pipeline import AUDIO_EXTENSIONS, load_audio  # noqa: E402
from separation import build_separator  # noqa: E402


EPS = 1e-10


def _rms(values: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(values, dtype=np.float64))))


def _frame_rms(values: np.ndarray, frame: int = 8_000, hop: int = 4_000) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    if len(values) < frame:
        values = np.pad(values, (0, frame - len(values)))
    starts = list(range(0, max(len(values) - frame + 1, 1), hop))
    last = max(len(values) - frame, 0)
    if starts[-1] != last:
        starts.append(last)
    return np.asarray([_rms(values[start:start + frame]) for start in starts])


def statistics(original: np.ndarray, voice: np.ndarray, music: np.ndarray) -> dict[str, float]:
    length = min(len(original), len(voice), len(music))
    original, voice, music = original[:length], voice[:length], music[:length]
    original_rms, voice_rms, music_rms = map(_rms, (original, voice, music))
    original_frames = _frame_rms(original)
    voice_frames = _frame_rms(voice)
    music_frames = _frame_rms(music)
    count = min(len(original_frames), len(voice_frames), len(music_frames))
    original_frames = original_frames[:count]
    voice_frames = voice_frames[:count]
    music_frames = music_frames[:count]

    voice_ratio = voice_frames / (original_frames + EPS)
    music_ratio = music_frames / (original_frames + EPS)
    dominance = np.log((voice_frames + EPS) / (music_frames + EPS))
    reconstruction = _rms(original - voice - music) / (original_rms + EPS)
    energy_sum = voice_rms**2 + music_rms**2 + EPS

    result = {
        "ORIGINAL_RMS": original_rms,
        "VOICE_RMS": voice_rms,
        "MUSIC_RMS": music_rms,
        "VOICE_ENERGY_RATIO": voice_rms**2 / (original_rms**2 + EPS),
        "MUSIC_ENERGY_RATIO": music_rms**2 / (original_rms**2 + EPS),
        "VOICE_STEM_SHARE": voice_rms**2 / energy_sum,
        "MUSIC_STEM_SHARE": music_rms**2 / energy_sum,
        "RECONSTRUCTION_ERROR": reconstruction,
        "VOICE_DOMINANT_FRACTION": float(np.mean(dominance > 0.0)),
        "MUSIC_DOMINANT_FRACTION": float(np.mean(dominance < 0.0)),
    }
    for name, values in (("VOICE_FRAME_RATIO", voice_ratio),
                         ("MUSIC_FRAME_RATIO", music_ratio)):
        for quantile in (0.5, 0.75, 0.9, 1.0):
            suffix = str(int(quantile * 100))
            result[f"{name}_Q{suffix}"] = float(np.quantile(values, quantile))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audio-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--repo", type=Path, default=ROOT / "models/htdemucs")
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    if args.num_shards < 1 or not 0 <= args.shard_index < args.num_shards:
        parser.error("invalid shard configuration")

    files = sorted(
        path for path in args.audio_dir.iterdir()
        if path.is_file() and path.suffix.lower() in AUDIO_EXTENSIONS
    )[args.shard_index::args.num_shards]
    if args.limit is not None:
        files = files[:args.limit]
    if not files:
        raise FileNotFoundError(f"No shard audio under {args.audio_dir}")

    separator = build_separator("htdemucs", device=args.device, repo=args.repo)
    records = []
    for path in tqdm(files, desc=f"stem-stats-{args.shard_index}"):
        original = load_audio(path)
        voice, music = separator.separate(path)
        records.append({"ID": path.stem, **statistics(original, voice, music)})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records).to_csv(args.output, index=False)
    print(f"Wrote {len(records)} rows to {args.output}")


if __name__ == "__main__":
    main()
