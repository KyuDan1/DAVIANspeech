#!/usr/bin/env python3
"""Build source-safe telephone mixtures for Voice-presence training."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import soundfile as sf
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from build_factorial_eval_1200 import (  # noqa: E402
    SR, mix_concurrent, mix_partial, mix_sequential, music_segment,
    peak_limit, stable_int,
)
from data_guard import assert_no_locked_eval_leakage  # noqa: E402
from pipeline import find_audio_files, load_audio  # noqa: E402
from telephone_channel import apply_channel  # noqa: E402


VARIANTS = (
    "resample8k", "pstn_bandpass", "g711_ulaw", "g726_24k", "opus_nb_8k",
)
MODES = (
    ("music_only", 400),
    ("voice_only", 400),
    ("concurrent", 600),
    ("partial_overlap", 500),
    ("sequential", 500),
)


def paths(directory: Path) -> dict[str, Path]:
    return {path.stem: path for path in find_audio_files(directory)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--voice-bank", type=Path,
        default=ROOT / "data/eval/phone_router_voice_train_v1",
    )
    parser.add_argument(
        "--music-bank", type=Path,
        default=ROOT / "data/eval/multigen_music_presence_train_v1",
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=ROOT / "data/eval/phone_presence_factorial_train_v1",
    )
    parser.add_argument(
        "--ffmpeg", type=Path,
        default=ROOT.parent / "conda_envs/envs/davianspeech/bin/ffmpeg",
    )
    parser.add_argument("--seed", type=int, default=20260903)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output_dir}")
    for bank in (args.voice_bank, args.music_bank):
        assert_no_locked_eval_leakage(
            bank / "truth.csv", ROOT / "configs/data_partitions.yaml"
        )

    voice_truth = pd.read_csv(args.voice_bank / "truth.csv", dtype={"ID": str})
    music_truth = pd.read_csv(args.music_bank / "truth.csv", dtype={"ID": str})
    voice_files, music_files = paths(args.voice_bank / "audio"), paths(args.music_bank / "audio")
    voice_truth = voice_truth[voice_truth.ID.isin(voice_files)].sort_values("ID").reset_index(drop=True)
    music_truth = music_truth[music_truth.ID.isin(music_files)].sort_values("ID").reset_index(drop=True)
    if len(voice_truth) < 100 or len(music_truth) < 100:
        raise ValueError("Insufficient independent component sources")

    audio_dir = args.output_dir / "audio"
    audio_dir.mkdir(parents=True)
    records, index = [], 0
    snrs = (-15, -10, -5, 0, 5, 10)
    overlaps = (0.15, 0.30, 0.50, 0.75)
    for mode, count in MODES:
        for local_index in tqdm(range(count), desc=mode):
            key = f"{args.seed}|{mode}|{local_index}"
            voice_row = voice_truth.iloc[
                (stable_int(key, "voice") + local_index) % len(voice_truth)
            ]
            music_row = music_truth.iloc[
                (stable_int(key, "music") + 7 * local_index) % len(music_truth)
            ]
            voice = load_audio(voice_files[voice_row.ID])
            music = load_audio(music_files[music_row.ID])
            snr = snrs[stable_int(key, "snr") % len(snrs)]
            voice_first = bool(stable_int(key, "order") % 2)
            overlap = overlaps[stable_int(key, "overlap") % len(overlaps)]
            if mode == "music_only":
                duration = (8 + stable_int(key, "duration") % 11) * SR
                audio = music_segment(music, duration, key)
                voice_present = 0
            elif mode == "voice_only":
                audio = voice[:20 * SR]
                voice_present = 1
            elif mode == "concurrent":
                audio = mix_concurrent(voice, music, snr, key)
                voice_present = 1
            elif mode == "partial_overlap":
                audio = mix_partial(voice, music, snr, overlap, voice_first, key)
                voice_present = 1
            else:
                gap = (0.0, 0.2, 0.5)[stable_int(key, "gap") % 3]
                audio = mix_sequential(voice, music, voice_first, gap, key)
                voice_present = 1
            if len(audio) < 4 * SR:
                audio = np.pad(audio, (0, 4 * SR - len(audio)))
            audio = peak_limit(audio[:60 * SR])
            variant = VARIANTS[(index + stable_int(mode, args.seed)) % len(VARIANTS)]
            sample_id = f"phone_presence_train_{index:04d}"
            audio = apply_channel(
                audio, variant, ffmpeg=args.ffmpeg,
                key=stable_int(sample_id, "channel"),
            )
            sf.write(audio_dir / f"{sample_id}.flac", audio, SR, subtype="PCM_16")
            component_fakes = []
            if voice_present:
                component_fakes.append(int(voice_row.VOICE_FAKE))
            if mode != "voice_only":
                component_fakes.append(int(music_row.MUSIC_FAKE))
            records.append({
                "ID": sample_id,
                "FILE_FAKE": max(component_fakes),
                "VOICE_FAKE": int(voice_row.VOICE_FAKE) if voice_present else np.nan,
                "MUSIC_FAKE": (
                    int(music_row.MUSIC_FAKE) if mode != "voice_only" else np.nan
                ),
                "VOICE_PRESENT": voice_present,
                "MUSIC_PRESENT": int(mode != "voice_only"),
                "AUDIO_TYPE": (
                    "mixed" if mode not in {"voice_only", "music_only"}
                    else mode.removesuffix("_only")
                ),
                "MIX_MODE": mode,
                "CHANNEL": variant,
                "VOICE_SOURCE_ID": voice_row.ID if voice_present else "",
                "MUSIC_SOURCE_ID": music_row.ID if mode != "voice_only" else "",
                "SNR_DB": snr if mode in {"concurrent", "partial_overlap"} else np.nan,
                "OVERLAP_FRACTION": overlap if mode == "partial_overlap" else np.nan,
                "ORDER": (
                    "voice_first" if voice_first else "music_first"
                ) if mode in {"partial_overlap", "sequential"} else "",
                "DURATION": round(len(audio) / SR, 3),
            })
            index += 1

    truth = pd.DataFrame(records)
    truth.to_csv(args.output_dir / "truth.csv", index=False)
    sample = pd.DataFrame({"ID": truth.ID})
    for column in (
        "FILE_FAKE_PROB", "VOICE_FAKE_PROB", "MUSIC_FAKE_PROB",
        "VOICE_PRESENT_PROB", "MUSIC_PRESENT_PROB",
    ):
        sample[column] = 0.5
    sample.to_csv(args.output_dir / "sample_submission.csv", index=False)
    print(truth.groupby(["MIX_MODE", "CHANNEL"]).size().unstack(fill_value=0))
    print(f"Built {len(truth)} source-safe telephone training files")


if __name__ == "__main__":
    main()
