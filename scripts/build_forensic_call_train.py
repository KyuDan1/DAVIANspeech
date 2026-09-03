#!/usr/bin/env python3
"""Build leakage-safe two-party call bags for sparse-spoof MIL training.

The waveform layout mirrors :mod:`build_forensic_call_audit`, but every
component is drawn only from the training mixtures' authorized source pools.
Sources are split by speaker/music group before calls are assembled.  Exact
fake-component ranges are stored so a crop is supervised by what it actually
contains rather than by the file-level OR label.

Channel codecs are deliberately not rendered here.  They are applied to the
final clean mixture by ``extract_dual_domain_stats.py --channel-variant`` so
all variants of one parent keep identical time labels without duplicating
audio on disk.
"""

from __future__ import annotations

import argparse
from functools import lru_cache
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import soundfile as sf
from tqdm import tqdm


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from build_forensic_call_audit import (  # noqa: E402
    CONVERSATION_MODES,
    MUSIC_CASES,
    SR,
    VOICE_CASES,
    add_music,
    conversation,
    stable_int,
)
from build_temporal_mixed_train import (  # noqa: E402
    SourceItem,
    collect_sources,
    load_audio,
    source_split,
)


def choose(
    pool: list[SourceItem], key: str, role: str,
    excluded_group: str | None = None,
) -> SourceItem:
    candidates = (
        [item for item in pool if item.group != excluded_group]
        if excluded_group is not None else pool
    )
    candidates = candidates or pool
    return candidates[stable_int(key, role) % len(candidates)]


def contiguous_ranges(mask: np.ndarray) -> list[list[float]]:
    """Encode true runs as half-open second ranges."""
    padded = np.pad(np.asarray(mask, dtype=np.int8), (1, 1))
    edges = np.flatnonzero(np.diff(padded))
    return [
        [round(float(start / SR), 6), round(float(end / SR), 6)]
        for start, end in edges.reshape(-1, 2)
    ]


def planned_cells(train_repeats: int, dev_repeats: int):
    for split, repeats in (("train", train_repeats), ("dev", dev_repeats)):
        for voice_case, first_fake, second_fake in VOICE_CASES:
            for mode in CONVERSATION_MODES:
                for music_case in MUSIC_CASES:
                    for repeat in range(repeats):
                        yield (
                            split, voice_case, first_fake, second_fake,
                            mode, music_case, repeat,
                        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir", type=Path,
        default=ROOT / "data" / "eval" / "forensic_call_train_v1",
    )
    parser.add_argument("--train-repeats", type=int, default=16)
    parser.add_argument("--dev-repeats", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260903)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output_dir}")
    if min(args.train_repeats, args.dev_repeats) < 1:
        parser.error("repeat counts must be positive")

    voices, music = collect_sources(ROOT / "data" / "eval")
    voice_pools = {
        (split, label): sorted(
            (item for item in voices
             if item.label == label and source_split(item, args.seed) == split),
            key=lambda item: item.key,
        )
        for split in ("train", "dev") for label in (0, 1)
    }
    music_pools = {
        (split, label): sorted(
            (item for item in music
             if item.label == label and source_split(item, args.seed) == split),
            key=lambda item: item.key,
        )
        for split in ("train", "dev") for label in (0, 1)
    }
    if any(not pool for pool in (*voice_pools.values(), *music_pools.values())):
        raise ValueError("A source/group split produced an empty label pool")

    @lru_cache(maxsize=512)
    def cached_audio(path: str) -> np.ndarray:
        return load_audio(Path(path))

    audio_dir = args.output_dir / "audio"
    audio_dir.mkdir(parents=True)
    records = []
    cells = list(planned_cells(args.train_repeats, args.dev_repeats))
    for index, (
        split, voice_case, first_fake, second_fake, mode, music_case, repeat,
    ) in enumerate(tqdm(cells, desc="forensic call train/dev")):
        key = (
            f"{args.seed}|{split}|{voice_case}|{mode}|{music_case}|{repeat}"
        )
        first = choose(voice_pools[(split, first_fake)], key, "first")
        second = choose(
            voice_pools[(split, second_fake)], key, "second", first.group
        )
        voice, first_mask, second_mask = conversation(
            cached_audio(str(first.path)), cached_audio(str(second.path)), mode, key
        )
        music_fake = None if music_case == "absent" else int(music_case == "fake")
        music_item = (
            None if music_fake is None
            else choose(music_pools[(split, music_fake)], key, "music")
        )
        mixed, first_mask, second_mask, music_fraction, music_layout = add_music(
            voice, first_mask, second_mask,
            None if music_item is None else cached_audio(str(music_item.path)),
            mode, key,
        )
        voice_mask = first_mask | second_mask
        fake_voice_mask = (
            (first_mask if first_fake else np.zeros_like(first_mask))
            | (second_mask if second_fake else np.zeros_like(second_mask))
        )
        music_mask = np.zeros(len(mixed), dtype=bool)
        if music_item is not None:
            if music_layout == "background":
                music_mask[:] = True
            else:
                hold_samples = int(round(music_fraction * len(mixed)))
                hold_start = (len(mixed) - hold_samples) // 2
                music_mask[hold_start:hold_start + hold_samples] = True
        fake_music_mask = music_mask if music_fake else np.zeros_like(music_mask)

        sample_id = f"forensic_train_{index:05d}"
        sf.write(audio_dir / f"{sample_id}.flac", mixed, SR, subtype="PCM_16")
        voice_fake = int(first_fake or second_fake)
        records.append({
            "ID": sample_id,
            "FILE_FAKE": int(voice_fake or (music_fake or 0)),
            "VOICE_FAKE": voice_fake,
            "MUSIC_FAKE": music_fake,
            "VOICE_PRESENT": 1,
            "MUSIC_PRESENT": int(music_item is not None),
            "AUDIO_TYPE": "voice" if music_item is None else "mixed",
            "VOICE_CASE": voice_case,
            "CONVERSATION_MODE": mode,
            "MUSIC_CASE": music_case,
            "MUSIC_LAYOUT": music_layout,
            "SPLIT": split,
            "FAKE_VOICE_FRACTION": float(
                fake_voice_mask.sum() / max(voice_mask.sum(), 1)
            ),
            "MUSIC_FRACTION": music_fraction,
            "VOICE_RANGES": json.dumps(contiguous_ranges(voice_mask)),
            "MUSIC_RANGES": json.dumps(contiguous_ranges(music_mask)),
            "VOICE_FAKE_RANGES": json.dumps(contiguous_ranges(fake_voice_mask)),
            "MUSIC_FAKE_RANGES": json.dumps(contiguous_ranges(fake_music_mask)),
            "FIRST_SOURCE_ID": first.source_id,
            "FIRST_SOURCE_BANK": first.bank_name,
            "FIRST_GENERATOR": first.generator,
            "FIRST_GROUP": first.group,
            "SECOND_SOURCE_ID": second.source_id,
            "SECOND_SOURCE_BANK": second.bank_name,
            "SECOND_GENERATOR": second.generator,
            "SECOND_GROUP": second.group,
            "MUSIC_SOURCE_ID": None if music_item is None else music_item.source_id,
            "MUSIC_SOURCE_BANK": None if music_item is None else music_item.bank_name,
            "MUSIC_GENERATOR": None if music_item is None else music_item.generator,
            "MUSIC_GROUP": None if music_item is None else music_item.group,
            "PARENT_ID": sample_id,
            "DURATION": len(mixed) / SR,
            "ROLE": "train_or_development_sparse_call_mil",
        })

    result = pd.DataFrame(records)
    result.to_csv(args.output_dir / "truth.csv", index=False)
    result.loc[result.SPLIT.eq("train")].to_csv(
        args.output_dir / "truth_train.csv", index=False
    )
    result.loc[result.SPLIT.eq("dev")].to_csv(
        args.output_dir / "truth_dev.csv", index=False
    )
    sample = pd.DataFrame({"ID": result.ID})
    for column in (
        "FILE_FAKE_PROB", "VOICE_FAKE_PROB", "MUSIC_FAKE_PROB",
        "VOICE_PRESENT_PROB", "MUSIC_PRESENT_PROB",
    ):
        sample[column] = 0.5
    sample.to_csv(args.output_dir / "sample_submission.csv", index=False)

    train_groups = set(result.loc[result.SPLIT.eq("train"), "FIRST_GROUP"])
    train_groups |= set(result.loc[result.SPLIT.eq("train"), "SECOND_GROUP"])
    dev_groups = set(result.loc[result.SPLIT.eq("dev"), "FIRST_GROUP"])
    dev_groups |= set(result.loc[result.SPLIT.eq("dev"), "SECOND_GROUP"])
    if train_groups & dev_groups:
        raise AssertionError("Voice groups cross the train/dev boundary")
    print(result.groupby(["SPLIT", "VOICE_CASE"]).size().unstack(fill_value=0))
    print(result.groupby(["SPLIT", "CONVERSATION_MODE"]).size().unstack(fill_value=0))
    print(result.groupby("SPLIT").FAKE_VOICE_FRACTION.describe())
    print(f"Built {len(result)} clean parent calls in {args.output_dir}")


if __name__ == "__main__":
    main()
