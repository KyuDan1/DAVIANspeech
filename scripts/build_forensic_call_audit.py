#!/usr/bin/env python3
"""Build a fixed 1,200-file two-party voice-phishing stress audit.

The audit crosses two-speaker authenticity, conversational timing, optional
background/hold music, and real telephony codecs.  Its source pools are locked
evaluation banks; the derived files must never be used for fitting or weight
selection.  The purpose is to expose sparse-fake dilution and channel failures
that one-speaker speech/music mixtures cannot represent.
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import soundfile as sf
from scipy.signal import resample_poly
from tqdm import tqdm


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from telephone_channel import apply_channel  # noqa: E402


SR = 16_000
VOICE_CASES = (
    ("real_real", 0, 0),
    ("fake_real", 1, 0),
    ("real_fake", 0, 1),
    ("fake_fake", 1, 1),
)
CONVERSATION_MODES = ("balanced_turns", "sparse_second_speaker")
MUSIC_CASES = ("absent", "real", "fake")
CHANNELS = (
    "clean", "g711_ulaw", "opus_nb_8k", "g722_wb",
    "transcode_g711_opus",
)


def stable_int(*parts: object) -> int:
    payload = "|".join(map(str, parts)).encode("utf-8")
    return int(hashlib.sha256(payload).hexdigest()[:16], 16)


def load_audio(path: Path) -> np.ndarray:
    audio, rate = sf.read(path, dtype="float32", always_2d=True)
    audio = audio.mean(axis=1)
    if rate != SR:
        divisor = np.gcd(rate, SR)
        audio = resample_poly(audio, SR // divisor, rate // divisor)
    audio = np.nan_to_num(np.asarray(audio, dtype=np.float32))
    if not audio.size:
        raise ValueError(f"Empty source audio: {path}")
    return audio


def find_audio(bank: Path, sample_id: str) -> Path:
    matches = [
        path for path in (bank / "audio").glob(f"{sample_id}.*")
        if path.is_file()
    ]
    if len(matches) != 1:
        raise ValueError(f"Expected one file for {bank.name}/{sample_id}: {matches}")
    return matches[0]


def peak_limit(audio: np.ndarray) -> np.ndarray:
    audio = np.nan_to_num(np.asarray(audio, dtype=np.float32))
    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    return audio * min(1.0, 0.98 / max(peak, 1e-8))


def rms(audio: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(audio), dtype=np.float64) + 1e-10))


def crop_or_tile(audio: np.ndarray, length: int, key: str) -> np.ndarray:
    if audio.size < length:
        audio = np.tile(audio, int(np.ceil(length / audio.size)))
    span = audio.size - length
    start = stable_int(key, "crop") % (span + 1)
    return np.asarray(audio[start:start + length], dtype=np.float32)


def source_pools(eval_root: Path) -> tuple[dict[int, list[dict]], dict[int, list[dict]]]:
    voice_bank = eval_root / "multigen_voice_v2"
    voice_truth = pd.read_csv(voice_bank / "truth.csv", dtype={"ID": str})
    # The locked source split was not used to fit any deployed head.  Using
    # only it also prevents this structural audit from recycling the earlier
    # dev/holdout sources used during exploratory fusion analysis.
    voice_truth = voice_truth.loc[voice_truth.SPLIT.eq("locked")].copy()
    voices: dict[int, list[dict]] = {0: [], 1: []}
    for row in voice_truth.sort_values(["VOICE_FAKE", "GENERATOR", "ID"]).to_dict("records"):
        label = int(row["VOICE_FAKE"])
        voices[label].append({
            "id": row["ID"],
            "generator": row["GENERATOR"],
            "group": str(row.get("GROUP_ID", row["ID"])),
            "audio": load_audio(find_audio(voice_bank, row["ID"])),
        })

    music_bank = eval_root / "source_disjoint_music_v1"
    music_truth = pd.read_csv(music_bank / "truth.csv", dtype={"ID": str})
    music_truth = music_truth.loc[music_truth.SPLIT.eq("prospective")].copy()
    music: dict[int, list[dict]] = {0: [], 1: []}
    for row in music_truth.sort_values(["MUSIC_FAKE", "GENERATOR", "ID"]).to_dict("records"):
        label = int(row["MUSIC_FAKE"])
        music[label].append({
            "id": row["ID"],
            "generator": row["GENERATOR"],
            "group": str(row.get("GROUP_ID", row["ID"])),
            "audio": load_audio(find_audio(music_bank, row["ID"])),
        })
    if any(not values for values in (*voices.values(), *music.values())):
        raise ValueError("A locked real/fake source pool is empty")
    return voices, music


def choose(pool: list[dict], *key: object) -> dict:
    return pool[stable_int(*key) % len(pool)]


def conversation(
    first: np.ndarray, second: np.ndarray, mode: str, key: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return voice waveform plus per-speaker activity masks."""
    if mode == "balanced_turns":
        speakers = (0, 1, 0, 1, 0, 1)
        lengths = [2.4 + (stable_int(key, "len", i) % 17) / 10 for i in range(6)]
    elif mode == "sparse_second_speaker":
        # Speaker 2 is deliberately just 1.2--2.0 s.  Depending on the
        # authenticity cell this is either a sparse fake injection or a short
        # real victim response surrounded by a fake caller.
        speakers = (0, 0, 1, 0, 0, 0)
        lengths = [
            3.0 + (stable_int(key, "len", i) % 21) / 10 for i in range(6)
        ]
        lengths[2] = 1.2 + (stable_int(key, "short") % 9) / 10
    else:
        raise ValueError(mode)

    pieces: list[tuple[int, np.ndarray]] = []
    for index, (speaker, seconds) in enumerate(zip(speakers, lengths)):
        length = int(seconds * SR)
        source = first if speaker == 0 else second
        pieces.append((speaker, crop_or_tile(source, length, f"{key}|turn|{index}")))
    gaps = [int((0.12 + (stable_int(key, "gap", i) % 39) / 100) * SR)
            for i in range(len(pieces) - 1)]
    total = sum(len(audio) for _, audio in pieces) + sum(gaps)
    output = np.zeros(total, dtype=np.float32)
    first_mask = np.zeros(total, dtype=bool)
    second_mask = np.zeros(total, dtype=bool)
    cursor = 0
    for index, (speaker, audio) in enumerate(pieces):
        end = cursor + len(audio)
        output[cursor:end] = audio
        (first_mask if speaker == 0 else second_mask)[cursor:end] = True
        cursor = end + (gaps[index] if index < len(gaps) else 0)
    return peak_limit(output), first_mask, second_mask


def add_music(
    voice: np.ndarray, first_mask: np.ndarray, second_mask: np.ndarray,
    music: np.ndarray | None, mode: str, key: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, str]:
    if music is None:
        return voice, first_mask, second_mask, 0.0, "absent"
    if mode == "balanced_turns":
        segment = crop_or_tile(music, len(voice), f"{key}|background")
        # Voice is 9--15 dB above background music.
        snr_db = 9.0 + stable_int(key, "snr") % 7
        gain = rms(voice[first_mask | second_mask]) / (
            rms(segment) * 10 ** (snr_db / 20)
        )
        return peak_limit(voice + gain * segment), first_mask, second_mask, 1.0, "background"

    # Insert a 5--9 s hold-music interval at the middle turn boundary.  It is
    # sequential, not an overlay, and keeps speaker masks exactly aligned.
    hold_length = int((5 + stable_int(key, "hold") % 5) * SR)
    hold = crop_or_tile(music, hold_length, f"{key}|hold")
    split = len(voice) // 2
    output = np.concatenate([voice[:split], hold, voice[split:]])
    zeros = np.zeros(hold_length, dtype=bool)
    first = np.concatenate([first_mask[:split], zeros, first_mask[split:]])
    second = np.concatenate([second_mask[:split], zeros, second_mask[split:]])
    return peak_limit(output), first, second, hold_length / len(output), "hold"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir", type=Path,
        default=ROOT / "data" / "eval" / "forensic_call_audit_v1",
    )
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260903)
    parser.add_argument(
        "--ffmpeg", type=Path,
        default=(
            ROOT.parent / "conda_envs" / "envs" / "davianspeech"
            / "bin" / "ffmpeg"
        ),
    )
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output_dir}")
    if args.repeats < 1:
        parser.error("--repeats must be positive")

    voices, music_pool = source_pools(ROOT / "data" / "eval")
    audio_dir = args.output_dir / "audio"
    audio_dir.mkdir(parents=True)
    records = []
    cells = [
        (voice_case, first_fake, second_fake, mode, music_case, repeat)
        for voice_case, first_fake, second_fake in VOICE_CASES
        for mode in CONVERSATION_MODES
        for music_case in MUSIC_CASES
        for repeat in range(args.repeats)
    ]
    for cell_index, (
        voice_case, first_fake, second_fake, mode, music_case, repeat,
    ) in enumerate(tqdm(cells, desc="forensic call bases")):
        key = f"{args.seed}|{voice_case}|{mode}|{music_case}|{repeat}"
        first = choose(voices[first_fake], key, "first")
        # Use a different group whenever the pool makes it possible.
        second_candidates = [
            item for item in voices[second_fake] if item["group"] != first["group"]
        ] or voices[second_fake]
        second = choose(second_candidates, key, "second")
        voice, first_mask, second_mask = conversation(
            first["audio"], second["audio"], mode, key
        )
        music_fake = None if music_case == "absent" else int(music_case == "fake")
        music_item = (
            None if music_fake is None
            else choose(music_pool[music_fake], key, "music")
        )
        mixed, first_mask, second_mask, music_fraction, music_layout = add_music(
            voice, first_mask, second_mask,
            None if music_item is None else music_item["audio"], mode, key,
        )
        fake_mask = (
            (first_mask if first_fake else np.zeros_like(first_mask))
            | (second_mask if second_fake else np.zeros_like(second_mask))
        )
        voice_mask = first_mask | second_mask
        fake_voice_fraction = float(fake_mask.sum() / max(voice_mask.sum(), 1))

        for channel_index, channel in enumerate(CHANNELS):
            sample_id = f"forensic_{cell_index:04d}_{channel}"
            transformed = apply_channel(
                mixed, channel, ffmpeg=args.ffmpeg,
                key=stable_int(key, channel),
            )
            sf.write(
                audio_dir / f"{sample_id}.flac", transformed, SR,
                subtype="PCM_16",
            )
            voice_fake = int(first_fake or second_fake)
            file_fake = int(voice_fake or (music_fake or 0))
            records.append({
                "ID": sample_id,
                "FILE_FAKE": file_fake,
                "VOICE_FAKE": voice_fake,
                "MUSIC_FAKE": music_fake,
                "VOICE_PRESENT": 1,
                "MUSIC_PRESENT": int(music_fake is not None),
                "AUDIO_TYPE": "voice" if music_fake is None else "mixed",
                "VOICE_CASE": voice_case,
                "CONVERSATION_MODE": mode,
                "MUSIC_CASE": music_case,
                "MUSIC_LAYOUT": music_layout,
                "CHANNEL": channel,
                "FAKE_VOICE_FRACTION": fake_voice_fraction,
                "MUSIC_FRACTION": music_fraction,
                "FIRST_SOURCE_ID": first["id"],
                "FIRST_GENERATOR": first["generator"],
                "SECOND_SOURCE_ID": second["id"],
                "SECOND_GENERATOR": second["generator"],
                "MUSIC_SOURCE_ID": None if music_item is None else music_item["id"],
                "MUSIC_GENERATOR": None if music_item is None else music_item["generator"],
                "PARENT_ID": f"forensic_{cell_index:04d}",
                "DURATION": len(transformed) / SR,
                "ROLE": "locked_forensic_structure_and_channel_audit",
            })

    result = pd.DataFrame(records)
    expected = len(cells) * len(CHANNELS)
    if len(result) != expected:
        raise AssertionError(f"Expected {expected} rows, got {len(result)}")
    result.to_csv(args.output_dir / "truth.csv", index=False)
    sample = pd.DataFrame({"ID": result.ID})
    for column in (
        "FILE_FAKE_PROB", "VOICE_FAKE_PROB", "MUSIC_FAKE_PROB",
        "VOICE_PRESENT_PROB", "MUSIC_PRESENT_PROB",
    ):
        sample[column] = 0.5
    sample.to_csv(args.output_dir / "sample_submission.csv", index=False)
    print(result.groupby(["CHANNEL", "FILE_FAKE"]).size().unstack(fill_value=0))
    print(result.groupby(["VOICE_CASE", "MUSIC_CASE"]).size().unstack(fill_value=0))
    print(result.groupby("CONVERSATION_MODE").FAKE_VOICE_FRACTION.describe())
    print(f"Built {len(result)} files in {args.output_dir}")


if __name__ == "__main__":
    main()
