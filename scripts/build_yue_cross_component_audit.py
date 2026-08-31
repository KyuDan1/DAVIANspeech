#!/usr/bin/env python3
"""Build an untuned RF/FR audit set from native YuE instrumental tracks."""

from __future__ import annotations

import argparse
import hashlib
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


def stable_int(value: str) -> int:
    return int(hashlib.sha256(value.encode()).hexdigest()[:16], 16)


def component_case(voice_fake: int | None, music_fake: int | None) -> str:
    if voice_fake is None:
        return f"music_{'fake' if music_fake else 'real'}"
    voice = "fake" if voice_fake else "real"
    music = "fake" if music_fake else "real"
    return f"voice_{voice}__music_{music}"


def load_audio(path: Path) -> np.ndarray:
    audio, _ = librosa.load(path, sr=SR, mono=True, dtype=np.float32)
    return audio[:60 * SR]


def find_audio(directory: Path, sample_id: str) -> Path:
    matches = list(directory.glob(f"{sample_id}.*"))
    if len(matches) != 1:
        raise ValueError(f"Expected one file for {sample_id}, found {matches}")
    return matches[0]


def rms(audio: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(audio), dtype=np.float64) + 1e-10))


def crop_or_tile(audio: np.ndarray, length: int, key: str) -> np.ndarray:
    if audio.size < length:
        audio = np.tile(audio, int(np.ceil(length / max(audio.size, 1))))
    start = stable_int(key) % (audio.size - length + 1)
    return audio[start:start + length].copy()


def peak_limit(audio: np.ndarray) -> np.ndarray:
    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    if peak > 0.98:
        audio = audio * (0.98 / peak)
    return audio.astype(np.float32)


def concurrent(voice: np.ndarray, music: np.ndarray, key: str) -> np.ndarray:
    length = min(max(voice.size, 8 * SR), 12 * SR)
    voice = crop_or_tile(voice, length, key + "voice")
    music = crop_or_tile(music, length, key + "music")
    music *= rms(voice) / rms(music)
    return peak_limit(voice + music)


def partial(voice: np.ndarray, music: np.ndarray, key: str) -> np.ndarray:
    voice = crop_or_tile(voice, 10 * SR, key + "voice")
    music = crop_or_tile(music, 8 * SR, key + "music")
    music *= rms(voice) / rms(music)
    canvas = np.zeros(14 * SR, dtype=np.float32)
    canvas[:10 * SR] += voice
    canvas[6 * SR:14 * SR] += music
    return peak_limit(canvas)


def sequential(voice: np.ndarray, music: np.ndarray, key: str) -> np.ndarray:
    voice = crop_or_tile(voice, 8 * SR, key + "voice")
    music = crop_or_tile(music, 8 * SR, key + "music")
    gap = np.zeros(SR // 4, dtype=np.float32)
    parts = [voice, gap, music] if stable_int(key) % 2 else [music, gap, voice]
    return peak_limit(np.concatenate(parts))


def choose_rows(frame: pd.DataFrame, count: int, fake: int) -> list[pd.Series]:
    selected = frame[pd.to_numeric(frame.FILE_FAKE) == fake].copy()
    if not fake and "GENRE" in selected and selected.GENRE.eq("Pop").any():
        selected = selected[selected.GENRE.eq("Pop")]
    selected = selected.sort_values("ID").reset_index(drop=True)
    if selected.empty:
        raise ValueError(f"No label={fake} rows available")
    return [selected.iloc[index % len(selected)] for index in range(count)]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--yue-root", type=Path, required=True)
    parser.add_argument("--voice-bank", type=Path, required=True)
    parser.add_argument("--music-bank", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    yue_tracks = sorted(args.yue_root.glob("seed_*/vocoder/stems/itrack.mp3"))
    if not yue_tracks:
        raise FileNotFoundError(f"No YuE instrumental tracks under {args.yue_root}")
    voice_truth = pd.read_csv(args.voice_bank / "truth.csv", dtype={"ID": str})
    music_truth = pd.read_csv(args.music_bank / "truth.csv", dtype={"ID": str})
    real_voices = choose_rows(voice_truth, len(yue_tracks), 0)
    fake_voices = choose_rows(voice_truth, len(yue_tracks), 1)
    real_music = choose_rows(music_truth, len(yue_tracks), 0)

    audio_dir = args.output_dir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    rows = []

    def write_sample(
        sample_id: str, audio: np.ndarray, voice_fake: int | None,
        music_fake: int | None, mode: str, voice_row: pd.Series | None,
        music_id: str, music_generator: str,
    ) -> None:
        case = component_case(voice_fake, music_fake)
        sf.write(audio_dir / f"{sample_id}.flac", audio, SR, subtype="PCM_16")
        rows.append({
            "ID": sample_id,
            "FILE_FAKE": int(bool(voice_fake) or bool(music_fake)),
            "VOICE_FAKE": "" if voice_fake is None else voice_fake,
            "MUSIC_FAKE": "" if music_fake is None else music_fake,
            "VOICE_PRESENT": int(voice_fake is not None),
            "MUSIC_PRESENT": int(music_fake is not None),
            "AUDIO_TYPE": "mixed" if voice_fake is not None
                          and music_fake is not None else (
                              "voice" if music_fake is None else "music"
                          ),
            "MIX_MODE": mode,
            "COMPONENT_CASE": case,
            "EVAL_CELL": f"{mode}__{case}",
            "SPLIT": "audit_yue",
            "VOICE_SOURCE_ID": "" if voice_row is None else voice_row.ID,
            "MUSIC_SOURCE_ID": music_id,
            "VOICE_GENERATOR": "" if voice_row is None else voice_row.GENERATOR,
            "MUSIC_GENERATOR": music_generator,
            "CONDITION": "clean_separator_free",
        })

    for index, (yue_path, real_voice_row, fake_voice_row, real_music_row) in enumerate(
        zip(yue_tracks, real_voices, fake_voices, real_music)
    ):
        real_voice = load_audio(find_audio(args.voice_bank / "audio", real_voice_row.ID))
        fake_voice = load_audio(find_audio(args.voice_bank / "audio", fake_voice_row.ID))
        real_track = load_audio(find_audio(args.music_bank / "audio", real_music_row.ID))
        fake_track = load_audio(yue_path)
        yue_vocal_path = yue_path.with_name("vtrack.mp3")
        yue_vocal = load_audio(yue_vocal_path)
        key = f"yue_{index:03d}"
        write_sample(
            key + "_music_real", crop_or_tile(real_track, 12 * SR, key), None, 0,
            "music_only", None, real_music_row.ID, "real",
        )
        write_sample(
            key + "_music_fake", crop_or_tile(fake_track, 12 * SR, key), None, 1,
            "music_only", None, yue_path.parent.parent.parent.name, "YuE",
        )
        for mode, mixer in (
            ("concurrent", concurrent), ("partial_overlap", partial),
            ("sequential", sequential),
        ):
            for voice_fake, music_fake, voice_row, voice, music_id, generator, music in (
                (0, 0, real_voice_row, real_voice, real_music_row.ID, "real", real_track),
                (1, 0, fake_voice_row, fake_voice, real_music_row.ID, "real", real_track),
                (0, 1, real_voice_row, real_voice, yue_path.parent.parent.parent.name,
                 "YuE", fake_track),
                (1, 1, fake_voice_row, fake_voice, yue_path.parent.parent.parent.name,
                 "YuE", fake_track),
            ):
                case = f"v{voice_fake}_m{music_fake}"
                write_sample(
                    f"{key}_{mode}_{case}", mixer(voice, music, key + mode + case),
                    voice_fake, music_fake, mode, voice_row, music_id, generator,
                )
        # YuE natively emits separate vocal and instrumental code streams.
        # Keep non-silent generated vocals as an additional singing-voice
        # stress slice; they are not used to balance the primary TTS factors.
        if rms(yue_vocal) >= 0.01:
            native_voice = pd.Series({
                "ID": yue_vocal_path.parent.parent.parent.name + "_vtrack",
                "GENERATOR": "YuE",
            })
            write_sample(
                key + "_native_vocal_real_music",
                concurrent(yue_vocal, real_track, key + "native_fr"),
                1, 0, "native_yue_vocal", native_voice,
                real_music_row.ID, "real",
            )
            write_sample(
                key + "_native_vocal_fake_music",
                concurrent(yue_vocal, fake_track, key + "native_ff"),
                1, 1, "native_yue_vocal", native_voice,
                yue_path.parent.parent.parent.name, "YuE",
            )

    truth = pd.DataFrame(rows)
    truth.to_csv(args.output_dir / "truth.csv", index=False)
    submission = truth[["ID"]].copy()
    for column in PREDICTION_COLUMNS:
        submission[column] = 0.5
    submission.to_csv(args.output_dir / "sample_submission.csv", index=False)
    print(f"Built {len(truth)} files from {len(yue_tracks)} native YuE tracks")
    print(truth.groupby(["MIX_MODE", "VOICE_FAKE", "MUSIC_FAKE"],
                        dropna=False).size().to_string())


if __name__ == "__main__":
    main()
