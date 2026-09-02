#!/usr/bin/env python3
"""Build a leakage-safe temporal mixture bank with exact component intervals.

Only component IDs already used by the configured training mixtures are
eligible.  The output is clean FLAC; codec/channel variants are applied during
feature extraction so the same interval labels remain valid.
"""

from __future__ import annotations

import argparse
import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
from scipy.signal import resample_poly


SR = 16_000
MODES = (
    "concurrent", "partial_overlap", "sequential",
    "sparse_voice", "sparse_music",
)
SOURCE_RECIPES = (
    ("external_mixed_train_v1", "echofake_v2", "echoes_v1"),
    ("mixed_devvoice_train_v1", "echofake_dev_v1", "echoes_v1"),
    ("mixed_fmc_music_train_v1", "echofake_v2", "competition_v2_bad_presence"),
)


def stable_int(*parts: object) -> int:
    value = "|".join(map(str, parts)).encode("utf-8")
    return int(hashlib.sha256(value).hexdigest()[:16], 16)


def rms(audio: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(audio), dtype=np.float64) + 1e-10))


def peak_limit(audio: np.ndarray) -> np.ndarray:
    audio = np.nan_to_num(np.asarray(audio, dtype=np.float32))
    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    if peak > 0.98:
        audio = audio * (0.98 / peak)
    return audio.astype(np.float32, copy=False)


def load_audio(path: Path) -> np.ndarray:
    audio, rate = sf.read(path, dtype="float32", always_2d=True)
    audio = audio.mean(axis=1)
    if rate != SR:
        divisor = np.gcd(rate, SR)
        audio = resample_poly(audio, SR // divisor, rate // divisor).astype(np.float32)
    return audio


def find_audio(bank: Path, sample_id: str) -> Path:
    matches = [path for path in (bank / "audio").glob(f"{sample_id}.*") if path.is_file()]
    if len(matches) != 1:
        raise ValueError(f"Expected one audio for {bank.name}/{sample_id}, got {matches}")
    return matches[0]


def crop_or_tile(audio: np.ndarray, samples: int, key: str) -> np.ndarray:
    if not audio.size:
        return np.zeros(samples, dtype=np.float32)
    if audio.size < samples:
        audio = np.tile(audio, int(np.ceil(samples / audio.size)))
    span = audio.size - samples
    start = stable_int(key, "crop") % (span + 1)
    return np.asarray(audio[start:start + samples], dtype=np.float32)


@dataclass(frozen=True)
class SourceItem:
    source_id: str
    label: int
    bank_name: str
    path: Path
    generator: str
    group: str

    @property
    def key(self) -> str:
        return f"{self.bank_name}:{self.source_id}"


def _label_map(frame: pd.DataFrame, id_column: str, label_column: str) -> dict[str, int]:
    pairs = frame[[id_column, label_column]].dropna().drop_duplicates()
    conflicts = pairs.groupby(id_column)[label_column].nunique()
    if (conflicts > 1).any():
        raise ValueError(f"Conflicting {label_column} values for source IDs")
    return {
        str(row[id_column]): int(float(row[label_column]))
        for _, row in pairs.iterrows()
    }


def first_value(row: pd.Series, columns: tuple[str, ...], fallback: str) -> str:
    for column in columns:
        value = row.get(column)
        if pd.notna(value) and str(value).strip() and str(value).lower() != "nan":
            return str(value)
    return fallback


def collect_sources(eval_root: Path) -> tuple[list[SourceItem], list[SourceItem]]:
    voice_items: dict[str, SourceItem] = {}
    music_items: dict[str, SourceItem] = {}
    for mixture_name, voice_bank_name, music_bank_name in SOURCE_RECIPES:
        mixture = pd.read_csv(eval_root / mixture_name / "truth.csv", dtype=str)
        voice_labels = _label_map(mixture, "VOICE_SOURCE_ID", "VOICE_FAKE")
        music_labels = _label_map(mixture, "MUSIC_SOURCE_ID", "MUSIC_FAKE")
        for component, bank_name, labels, destination in (
            ("voice", voice_bank_name, voice_labels, voice_items),
            ("music", music_bank_name, music_labels, music_items),
        ):
            bank = eval_root / bank_name
            truth = pd.read_csv(bank / "truth.csv", dtype=str).set_index("ID")
            for source_id, label in labels.items():
                if source_id not in truth.index:
                    raise KeyError(f"{source_id} is missing from {bank_name}")
                row = truth.loc[source_id]
                if isinstance(row, pd.DataFrame):
                    row = row.iloc[0]
                truth_value = row[f"{component.upper()}_FAKE"]
                present_value = row.get(f"{component.upper()}_PRESENT", "1")
                other = "MUSIC" if component == "voice" else "VOICE"
                other_present = row.get(f"{other}_PRESENT", "0")
                # The older FMC mining manifest deliberately included hard
                # presence mistakes (speech scored as music).  Those examples
                # are useful for a presence classifier but must not become
                # music-authenticity supervision here.
                if (
                    pd.isna(truth_value)
                    or int(float(present_value)) != 1
                    or (pd.notna(other_present) and int(float(other_present)) != 0)
                ):
                    continue
                truth_label = int(float(truth_value))
                if truth_label != label:
                    raise ValueError(f"Label mismatch for {bank_name}/{source_id}")
                generator = first_value(row, ("GENERATOR",), "bonafide")
                group = first_value(
                    row, ("SPEAKER_ID", "SPEAKER", "GROUP_ID", "SOURCE_ID"),
                    source_id,
                )
                item = SourceItem(
                    source_id=source_id, label=label, bank_name=bank_name,
                    path=find_audio(bank, source_id), generator=generator,
                    group=group,
                )
                destination[item.key] = item
    return list(voice_items.values()), list(music_items.values())


def source_split(item: SourceItem, seed: int) -> str:
    # Split by original source/group, not by generated mixture row.
    # Do not include bank_name: EchoFake releases share speaker identities.
    bucket = stable_int(seed, item.group, "source_split") % 5
    return "dev" if bucket == 0 else "train"


class Selector:
    def __init__(self, items: list[SourceItem], seed: int, component: str) -> None:
        self.pools: dict[tuple[str, int], list[SourceItem]] = {}
        self.counts: dict[tuple[str, int], int] = {}
        self.seed = seed
        self.component = component
        for split in ("train", "dev"):
            for label in (0, 1):
                pool = sorted(
                    (item for item in items if item.label == label and source_split(item, seed) == split),
                    key=lambda item: item.key,
                )
                if not pool:
                    raise ValueError(f"Empty {component} pool for {split}/{label}")
                self.pools[(split, label)] = pool
                self.counts[(split, label)] = 0

    def take(self, split: str, label: int, key: str) -> SourceItem:
        pool = self.pools[(split, label)]
        offset = stable_int(self.seed, self.component, split, label, key) % len(pool)
        index = (offset + self.counts[(split, label)]) % len(pool)
        self.counts[(split, label)] += 1
        return pool[index]


def place(canvas: np.ndarray, audio: np.ndarray, start: int, gain: float = 1.0) -> tuple[int, int]:
    end = min(start + audio.size, canvas.size)
    if end > start:
        canvas[start:end] += gain * audio[:end - start]
    return start, end


def mix_with_intervals(
    voice: np.ndarray, music: np.ndarray, mode: str, snr_db: float, key: str,
) -> tuple[np.ndarray, tuple[int, int], tuple[int, int]]:
    """Return mixture plus half-open voice/music sample intervals."""
    voice = np.asarray(voice[:14 * SR], dtype=np.float32)
    voice = voice if voice.size else np.zeros(SR, dtype=np.float32)
    if mode == "concurrent":
        duration = min(max(voice.size, 8 * SR), 18 * SR)
        voice = voice[:duration]
        music = crop_or_tile(music, duration, key + "|music")
        starts = (0, 0)
    elif mode == "partial_overlap":
        music = crop_or_tile(music, 10 * SR, key + "|music")
        fraction = (0.25, 0.50, 0.75)[stable_int(key, "overlap") % 3]
        overlap = max(1, int(min(voice.size, music.size) * fraction))
        if stable_int(key, "order") % 2:
            starts = (0, max(voice.size - overlap, 0))
        else:
            starts = (max(music.size - overlap, 0), 0)
        duration = max(starts[0] + voice.size, starts[1] + music.size)
    elif mode == "sequential":
        music = crop_or_tile(music, 8 * SR, key + "|music")
        gap = (0, SR // 5, SR // 2)[stable_int(key, "gap") % 3]
        if stable_int(key, "order") % 2:
            starts = (0, voice.size + gap)
        else:
            starts = (music.size + gap, 0)
        duration = starts[0] + voice.size if starts[0] else starts[1] + music.size
    elif mode == "sparse_voice":
        duration = (18 + stable_int(key, "duration") % 13) * SR
        voice = voice[: min(voice.size, (3 + stable_int(key, "voice_seconds") % 4) * SR)]
        music = crop_or_tile(music, duration, key + "|music")
        starts = (stable_int(key, "voice_start") % max(duration - voice.size + 1, 1), 0)
    elif mode == "sparse_music":
        duration = max(voice.size, 8 * SR)
        music_seconds = 2 + stable_int(key, "music_seconds") % 4
        music = crop_or_tile(music, min(music_seconds * SR, duration), key + "|music")
        voice = voice[:duration]
        starts = (0, stable_int(key, "music_start") % max(duration - music.size + 1, 1))
    else:
        raise ValueError(f"Unknown mode: {mode}")

    duration = min(int(duration), 60 * SR)
    canvas = np.zeros(duration, dtype=np.float32)
    # SNR is voice/music.  Place voice first, then scale music to the target.
    voice_interval = place(canvas, voice, int(starts[0]))
    music_gain = rms(voice) / (rms(music) * 10 ** (snr_db / 20))
    music_interval = place(canvas, music, int(starts[1]), music_gain)
    if canvas.size < 4 * SR:
        canvas = np.pad(canvas, (0, 4 * SR - canvas.size))
    return peak_limit(canvas), voice_interval, music_interval


def plan_rows(train_per_cell: int, dev_per_cell: int):
    for split, count in (("train", train_per_cell), ("dev", dev_per_cell)):
        for mode in MODES:
            for voice_fake in (0, 1):
                for music_fake in (0, 1):
                    for repeat in range(count):
                        yield split, mode, voice_fake, music_fake, repeat


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-root", type=Path, default=Path("data/eval"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260902)
    parser.add_argument("--train-per-cell", type=int, default=128)
    parser.add_argument("--dev-per-cell", type=int, default=32)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output_dir}")

    voices, music = collect_sources(args.eval_root)
    voice_selector = Selector(voices, args.seed, "voice")
    music_selector = Selector(music, args.seed, "music")
    audio_dir = args.output_dir / "audio"
    audio_dir.mkdir(parents=True)
    records = []
    snrs = (-15, -10, -5, 0, 5, 10, 15)
    voice_cache: dict[tuple[str, str, int, int], SourceItem] = {}
    music_cache: dict[tuple[str, str, int, int], SourceItem] = {}
    for index, (split, mode, voice_fake, music_fake, repeat) in enumerate(
        plan_rows(args.train_per_cell, args.dev_per_cell)
    ):
        pair_group = f"{split}|{mode}|{repeat}"
        key = f"{args.seed}|{pair_group}|{voice_fake}|{music_fake}"
        voice_key = (split, mode, repeat, voice_fake)
        music_key = (split, mode, repeat, music_fake)
        if voice_key not in voice_cache:
            voice_cache[voice_key] = voice_selector.take(
                split, voice_fake, f"{args.seed}|{pair_group}|voice|{voice_fake}"
            )
        if music_key not in music_cache:
            music_cache[music_key] = music_selector.take(
                split, music_fake, f"{args.seed}|{pair_group}|music|{music_fake}"
            )
        voice_item = voice_cache[voice_key]
        music_item = music_cache[music_key]
        voice_audio = load_audio(voice_item.path)
        music_audio = load_audio(music_item.path)
        layout_key = f"{args.seed}|{pair_group}|layout"
        snr_db = snrs[stable_int(layout_key, "snr") % len(snrs)]
        mixed, voice_interval, music_interval = mix_with_intervals(
            voice_audio, music_audio, mode, snr_db, layout_key
        )
        sample_id = f"temporal_mix_{index:05d}"
        sf.write(audio_dir / f"{sample_id}.flac", mixed, SR, subtype="PCM_16")
        records.append({
            "ID": sample_id,
            "FILE_FAKE": max(voice_fake, music_fake),
            "VOICE_FAKE": voice_fake,
            "MUSIC_FAKE": music_fake,
            "VOICE_PRESENT": 1,
            "MUSIC_PRESENT": 1,
            "AUDIO_TYPE": "mixed",
            "MIX_MODE": mode,
            "COMPONENT_CASE": f"voice_{'fake' if voice_fake else 'real'}__music_{'fake' if music_fake else 'real'}",
            "SPLIT": split,
            "PAIR_GROUP": pair_group,
            "VOICE_SOURCE_ID": voice_item.source_id,
            "VOICE_SOURCE_BANK": voice_item.bank_name,
            "VOICE_GENERATOR": voice_item.generator,
            "VOICE_GROUP": voice_item.group,
            "MUSIC_SOURCE_ID": music_item.source_id,
            "MUSIC_SOURCE_BANK": music_item.bank_name,
            "MUSIC_GENERATOR": music_item.generator,
            "MUSIC_GROUP": music_item.group,
            "SNR_DB": snr_db,
            "VOICE_START": voice_interval[0] / SR,
            "VOICE_END": voice_interval[1] / SR,
            "MUSIC_START": music_interval[0] / SR,
            "MUSIC_END": music_interval[1] / SR,
            "DURATION": mixed.size / SR,
        })

    truth = pd.DataFrame(records)
    truth.to_csv(args.output_dir / "truth.csv", index=False)
    for split in ("train", "dev"):
        truth[truth.SPLIT == split].to_csv(
            args.output_dir / f"truth_{split}.csv", index=False
        )
    print(truth.groupby(["SPLIT", "MIX_MODE", "COMPONENT_CASE"]).size().to_string())
    print(f"voice sources={len(voices)}, music sources={len(music)}")
    print(f"Built {len(truth)} files at {args.output_dir}")


if __name__ == "__main__":
    main()
