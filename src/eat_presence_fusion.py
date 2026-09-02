"""Add EAT AudioSet evidence to PANNs presence and refresh the file gate."""

from __future__ import annotations

import csv
import gc
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

try:  # package import in tests; flat import in the offline submission
    from .eat_presence import EatPresence, fuse_presence
    from .pipeline import find_audio_files, load_audio, order_by_submission
except ImportError:  # pragma: no cover - exercised by script.py
    from eat_presence import EatPresence, fuse_presence
    from pipeline import find_audio_files, load_audio, order_by_submission


def combine_with_gate(voice: float, music: float, voice_present: float,
                      music_present: float, gate: float = 0.60) -> float:
    active = []
    if voice_present >= gate:
        active.append(voice)
    if music_present >= gate:
        active.append(music)
    if active:
        return float(max(active))
    return float(voice if voice_present >= music_present else music)


def apply_eat_presence_fusion(
    test_dir: Path,
    submission_path: Path,
    eat_dir: Path,
    labels_dir: Path,
    device: str = "cuda",
    voice_weight: float = 0.30,
    music_weight: float = 0.90,
    file_gate: float = 0.60,
) -> None:
    """Rewrite presence ranks and the anchor File score, one file at a time."""
    with submission_path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        columns = list(reader.fieldnames or [])
        rows = list(reader)
    required = {
        "ID", "FILE_FAKE_PROB", "VOICE_FAKE_PROB", "MUSIC_FAKE_PROB",
        "VOICE_PRESENT_PROB", "MUSIC_PRESENT_PROB",
    }
    if not required.issubset(columns):
        raise ValueError(f"Submission is missing {sorted(required - set(columns))}")
    for value in (voice_weight, music_weight, file_gate):
        if not 0 <= value <= 1:
            raise ValueError("fusion weights and gate must lie in [0, 1]")

    audio_files = order_by_submission(find_audio_files(test_dir), rows)
    gc.collect()
    torch.cuda.empty_cache()
    detector = EatPresence(eat_dir, labels_dir, device=device)
    for row, path in zip(rows, tqdm(audio_files, desc="EAT presence")):
        eat_voice, eat_music = detector.predict(load_audio(path))
        voice_present, music_present = fuse_presence(
            float(row["VOICE_PRESENT_PROB"]),
            float(row["MUSIC_PRESENT_PROB"]), eat_voice, eat_music,
            voice_weight=voice_weight, music_weight=music_weight,
        )
        row["FILE_FAKE_PROB"] = round(combine_with_gate(
            float(row["VOICE_FAKE_PROB"]), float(row["MUSIC_FAKE_PROB"]),
            voice_present, music_present, gate=file_gate,
        ), 10)
        row["VOICE_PRESENT_PROB"] = round(voice_present, 10)
        row["MUSIC_PRESENT_PROB"] = round(music_present, 10)
    del detector
    gc.collect()
    torch.cuda.empty_cache()

    temporary = submission_path.with_suffix(".tmp.csv")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(submission_path)
