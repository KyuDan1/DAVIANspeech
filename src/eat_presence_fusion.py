"""Add EAT AudioSet evidence to PANNs presence.

Presence and authenticity are deliberately decoupled by default.  Updating a
file-authenticity score through a hard presence threshold can propagate a small
component-presence error into ADS, while the competition scores presence
directly through CPS.
"""

from __future__ import annotations

import csv
import gc
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

try:  # package import in tests; flat import in the offline submission
    from .eat_presence import EatPresence, fuse_music_probe, fuse_presence
    from .pipeline import find_audio_files, load_audio, order_by_submission
except ImportError:  # pragma: no cover - exercised by script.py
    from eat_presence import EatPresence, fuse_music_probe, fuse_presence
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
    voice_weight: float = 0.35,
    music_weight: float = 0.90,
    file_gate: float = 0.60,
    update_file_score: bool = False,
    presence_head_path: Path | None = None,
    music_probe_weight: float = 0.40,
) -> None:
    """Rewrite presence ranks and optionally refresh File score.

    ``update_file_score=False`` is the safe competition default: it preserves
    the verified ADS path exactly and prevents hard-gate error propagation.
    The opt-in path remains available for controlled ablations.
    """
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
    for value in (voice_weight, music_weight, file_gate, music_probe_weight):
        if not 0 <= value <= 1:
            raise ValueError("fusion weights and gate must lie in [0, 1]")

    audio_files = order_by_submission(find_audio_files(test_dir), rows)
    gc.collect()
    torch.cuda.empty_cache()
    detector = EatPresence(
        eat_dir, labels_dir, device=device,
        presence_head_path=presence_head_path,
    )
    for row, path in zip(rows, tqdm(audio_files, desc="EAT presence")):
        eat_voice, eat_music, probe_music = detector.predict_with_probe(load_audio(path))
        voice_present, music_present = fuse_presence(
            float(row["VOICE_PRESENT_PROB"]),
            float(row["MUSIC_PRESENT_PROB"]), eat_voice, eat_music,
            voice_weight=voice_weight, music_weight=music_weight,
        )
        if probe_music is not None:
            music_present = fuse_music_probe(
                music_present, probe_music, music_probe_weight
            )
        if update_file_score:
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
