#!/usr/bin/env python3
"""Build clean/telephone paired mixtures for channel-invariant training.

Every source-safe mixture is saved once without channel degradation and again
through deterministic telephone variants.  ``PARENT_ID`` links each degraded
file to its clean counterpart, enabling an explicit channel-consistency loss.
"""

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
from telephone_channel import POSITIVE_VARIANTS, apply_channel  # noqa: E402


DEFAULT_VARIANTS = (
    "resample8k", "pstn_bandpass", "g711_ulaw", "g711_alaw",
    "g726_24k", "opus_nb_8k", "opus_nb_12k", "random_bandpass",
    "mulaw_numpy", "fft_narrowband", "packetloss_opus_nb",
    "packetloss_g711",
)
MODES = (
    ("music_only", 200),
    ("voice_only", 200),
    ("concurrent", 250),
    ("partial_overlap", 175),
    ("sequential", 175),
)


def audio_paths(directory: Path) -> dict[str, Path]:
    return {path.stem: path for path in find_audio_files(directory)}


def choose(
    groups: dict[int, pd.DataFrame], label: int, key: str, offset: int,
) -> pd.Series:
    group = groups[label]
    return group.iloc[(stable_int(key, label) + offset) % len(group)]


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
        default=ROOT / "data/eval/channel_invariant_factorial_train_v1",
    )
    parser.add_argument(
        "--ffmpeg", type=Path,
        default=ROOT.parent / "conda_envs/envs/davianspeech/bin/ffmpeg",
    )
    parser.add_argument(
        "--variants", nargs="+", choices=POSITIVE_VARIANTS,
        default=list(DEFAULT_VARIANTS),
    )
    parser.add_argument("--variants-per-source", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260905)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output_dir}")
    if not 1 <= args.variants_per_source <= len(args.variants):
        parser.error("variants-per-source must be between 1 and len(variants)")
    for bank in (args.voice_bank, args.music_bank):
        assert_no_locked_eval_leakage(
            bank / "truth.csv", ROOT / "configs/data_partitions.yaml"
        )

    voice_truth = pd.read_csv(args.voice_bank / "truth.csv", dtype={"ID": str})
    music_truth = pd.read_csv(args.music_bank / "truth.csv", dtype={"ID": str})
    voice_files = audio_paths(args.voice_bank / "audio")
    music_files = audio_paths(args.music_bank / "audio")
    voice_truth = voice_truth.loc[voice_truth.ID.isin(voice_files)].copy()
    music_truth = music_truth.loc[music_truth.ID.isin(music_files)].copy()
    voice_groups = {
        label: group.sort_values("ID").reset_index(drop=True)
        for label, group in voice_truth.groupby(voice_truth.VOICE_FAKE.astype(int))
    }
    music_groups = {
        label: group.sort_values("ID").reset_index(drop=True)
        for label, group in music_truth.groupby(music_truth.MUSIC_FAKE.astype(int))
    }
    if set(voice_groups) != {0, 1} or set(music_groups) != {0, 1}:
        raise ValueError("Both component banks must contain real and fake rows")

    audio_dir = args.output_dir / "audio"
    audio_dir.mkdir(parents=True)
    records: list[dict] = []
    base_index = 0
    snrs = (-15, -10, -5, 0, 5, 10)
    overlaps = (0.15, 0.30, 0.50, 0.75)
    for mode, count in MODES:
        for local_index in tqdm(range(count), desc=mode):
            key = f"{args.seed}|{mode}|{local_index}"
            if mode == "music_only":
                voice_label, music_label = 0, local_index % 2
            elif mode == "voice_only":
                voice_label, music_label = local_index % 2, 0
            else:
                voice_label, music_label = divmod(local_index % 4, 2)
            voice_row = choose(
                voice_groups, voice_label, key + "|voice", local_index
            )
            music_row = choose(
                music_groups, music_label, key + "|music", 7 * local_index
            )
            voice = load_audio(voice_files[voice_row.ID])
            music = load_audio(music_files[music_row.ID])
            snr = snrs[stable_int(key, "snr") % len(snrs)]
            overlap = overlaps[stable_int(key, "overlap") % len(overlaps)]
            voice_first = bool(stable_int(key, "order") % 2)
            if mode == "music_only":
                duration = (8 + stable_int(key, "duration") % 11) * SR
                clean = music_segment(music, duration, key)
                voice_present = 0
            elif mode == "voice_only":
                clean = voice[:20 * SR]
                voice_present = 1
            elif mode == "concurrent":
                clean = mix_concurrent(voice, music, snr, key)
                voice_present = 1
            elif mode == "partial_overlap":
                clean = mix_partial(
                    voice, music, snr, overlap, voice_first, key
                )
                voice_present = 1
            else:
                gap = (0.0, 0.2, 0.5)[stable_int(key, "gap") % 3]
                clean = mix_sequential(voice, music, voice_first, gap, key)
                voice_present = 1
            if len(clean) < 4 * SR:
                clean = np.pad(clean, (0, 4 * SR - len(clean)))
            clean = peak_limit(clean[:60 * SR])

            base_id = f"channel_inv_{base_index:04d}"
            start = stable_int(key, "variant") % len(args.variants)
            variants = ["clean"] + [
                args.variants[(start + index) % len(args.variants)]
                for index in range(args.variants_per_source)
            ]
            for variant in variants:
                sample_id = f"{base_id}__{variant}"
                audio = clean if variant == "clean" else apply_channel(
                    clean, variant, ffmpeg=args.ffmpeg,
                    key=stable_int(sample_id, "channel"),
                )
                sf.write(
                    audio_dir / f"{sample_id}.flac", audio, SR,
                    subtype="PCM_16",
                )
                component_fakes = []
                if voice_present:
                    component_fakes.append(voice_label)
                if mode != "voice_only":
                    component_fakes.append(music_label)
                records.append({
                    "ID": sample_id,
                    "PARENT_ID": (
                        "" if variant == "clean" else f"{base_id}__clean"
                    ),
                    "MIXTURE_ID": base_id,
                    "FILE_FAKE": max(component_fakes),
                    "VOICE_FAKE": voice_label if voice_present else np.nan,
                    "MUSIC_FAKE": (
                        music_label if mode != "voice_only" else np.nan
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
                    "MUSIC_SOURCE_ID": (
                        music_row.ID if mode != "voice_only" else ""
                    ),
                    "SNR_DB": (
                        snr if mode in {"concurrent", "partial_overlap"}
                        else np.nan
                    ),
                    "OVERLAP_FRACTION": (
                        overlap if mode == "partial_overlap" else np.nan
                    ),
                    "ORDER": (
                        "voice_first" if voice_first else "music_first"
                    ) if mode in {"partial_overlap", "sequential"} else "",
                    "DURATION": round(len(audio) / SR, 3),
                })
            base_index += 1

    truth = pd.DataFrame(records)
    truth.to_csv(args.output_dir / "truth.csv", index=False)
    sample = pd.DataFrame({"ID": truth.ID})
    for column in (
        "FILE_FAKE_PROB", "VOICE_FAKE_PROB", "MUSIC_FAKE_PROB",
        "VOICE_PRESENT_PROB", "MUSIC_PRESENT_PROB",
    ):
        sample[column] = 0.5
    sample.to_csv(args.output_dir / "sample_submission.csv", index=False)
    print(truth.groupby(["CHANNEL", "MIX_MODE"]).size().unstack(fill_value=0))
    print(truth[["FILE_FAKE", "VOICE_FAKE", "MUSIC_FAKE"]].apply(
        lambda column: column.value_counts(dropna=False).to_dict()
    ))
    print(f"Built {len(truth)} files from {base_index} clean source mixtures")


if __name__ == "__main__":
    main()
