"""Conservative File consistency from the final component probabilities.

The competition defines ``FILE_FAKE`` as the logical OR of the authenticity
labels of the present components.  This postprocessor intentionally runs last:
it reads the *final* Voice/Music probabilities already written by any upstream
specialists and softly transfers their strongest evidence to File.

Presence is used only as a high-recall mixed-content gate.  It prevents an
obviously absent component from changing File, while avoiding the much less
stable presence-logit reweighting of authenticity evidence.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np


def _logit(value: np.ndarray | float) -> np.ndarray:
    value = np.clip(np.asarray(value, dtype=np.float64), 1e-5, 1 - 1e-5)
    return np.log(value) - np.log1p(-value)


def _sigmoid(value: np.ndarray | float) -> np.ndarray:
    return np.exp(-np.logaddexp(0.0, -np.asarray(value, dtype=np.float64)))


def component_consistent_file_probability(
    file_fake: np.ndarray | float,
    voice_fake: np.ndarray | float,
    music_fake: np.ndarray | float,
    voice_present: np.ndarray | float,
    music_present: np.ndarray | float,
    *,
    file_weight: float = 0.20,
    voice_presence_threshold: float = 0.10,
    music_presence_threshold: float = 0.20,
) -> np.ndarray:
    """Return File probabilities with a gated final-component residual.

    The maximum is a conservative soft surrogate for the label OR.  A noisy-OR
    was also evaluated, but its joint inflation transferred more correlated
    component errors and was less consistent on held-out mixed corpora.
    """
    if not 0.0 <= file_weight <= 1.0:
        raise ValueError("file_weight must be in [0, 1]")
    for name, value in (
        ("voice_presence_threshold", voice_presence_threshold),
        ("music_presence_threshold", music_presence_threshold),
    ):
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"{name} must be in [0, 1]")

    file_fake = np.asarray(file_fake, dtype=np.float64)
    voice_fake = np.asarray(voice_fake, dtype=np.float64)
    music_fake = np.asarray(music_fake, dtype=np.float64)
    voice_present = np.asarray(voice_present, dtype=np.float64)
    music_present = np.asarray(music_present, dtype=np.float64)
    arrays = np.broadcast_arrays(
        file_fake, voice_fake, music_fake, voice_present, music_present,
    )
    if not all(np.isfinite(value).all() for value in arrays):
        raise ValueError("component consistency inputs must be finite")
    if not all(((value >= 0.0) & (value <= 1.0)).all() for value in arrays):
        raise ValueError("component consistency inputs must be probabilities")

    file_fake, voice_fake, music_fake, voice_present, music_present = arrays
    mixed = (
        (voice_present >= voice_presence_threshold)
        & (music_present >= music_presence_threshold)
    )
    component_max = np.maximum(voice_fake, music_fake)
    weight = file_weight * mixed.astype(np.float64)
    result = _sigmoid(
        (1.0 - weight) * _logit(file_fake)
        + weight * _logit(component_max)
    )
    return np.asarray(result, dtype=np.float64)


def apply_component_consistent_file_fusion(
    submission_path: Path,
    *,
    file_weight: float = 0.20,
    voice_presence_threshold: float = 0.10,
    music_presence_threshold: float = 0.20,
) -> None:
    """Update only ``FILE_FAKE_PROB`` in a submission CSV, atomically."""
    submission_path = Path(submission_path)
    with submission_path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        columns = list(reader.fieldnames or [])
        rows = list(reader)
    required = {
        "ID", "FILE_FAKE_PROB", "VOICE_FAKE_PROB", "MUSIC_FAKE_PROB",
        "VOICE_PRESENT_PROB", "MUSIC_PRESENT_PROB",
    }
    missing = required.difference(columns)
    if missing:
        raise ValueError(f"Submission is missing columns: {sorted(missing)}")
    ids = [row["ID"] for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("Submission contains duplicate IDs")

    values = np.asarray([
        [
            float(row["FILE_FAKE_PROB"]),
            float(row["VOICE_FAKE_PROB"]),
            float(row["MUSIC_FAKE_PROB"]),
            float(row["VOICE_PRESENT_PROB"]),
            float(row["MUSIC_PRESENT_PROB"]),
        ]
        for row in rows
    ], dtype=np.float64)
    updated = component_consistent_file_probability(
        values[:, 0], values[:, 1], values[:, 2], values[:, 3], values[:, 4],
        file_weight=file_weight,
        voice_presence_threshold=voice_presence_threshold,
        music_presence_threshold=music_presence_threshold,
    )
    for row, probability in zip(rows, updated):
        row["FILE_FAKE_PROB"] = round(float(probability), 10)

    temporary = submission_path.with_suffix(".tmp.csv")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(submission_path)
