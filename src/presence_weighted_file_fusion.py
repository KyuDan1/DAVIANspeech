"""File-local logical consistency using soft component-presence evidence.

The competition label is the OR of fake labels for the components that are
actually present.  This postprocessor uses only the five probabilities of the
same file.  Presence is a soft reliability term rather than a hard gate, so a
small presence error cannot abruptly switch a component on or off.
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


def component_file_evidence(
    voice_fake: np.ndarray | float,
    music_fake: np.ndarray | float,
    voice_present: np.ndarray | float,
    music_present: np.ndarray | float,
    presence_logit_weight: float = 0.5,
) -> np.ndarray:
    """Return a soft OR over the two presence-adjusted fake logits."""
    if not 0 <= presence_logit_weight <= 1:
        raise ValueError("presence_logit_weight must be in [0, 1]")
    voice = _logit(voice_fake) + presence_logit_weight * _logit(voice_present)
    music = _logit(music_fake) + presence_logit_weight * _logit(music_present)
    return _sigmoid(np.maximum(voice, music))


def apply_presence_weighted_file_fusion(
    submission_path: Path,
    file_weight: float = 0.5,
    presence_logit_weight: float = 0.5,
) -> None:
    """Blend File with the soft component OR without changing other columns."""
    if not 0 <= file_weight <= 1:
        raise ValueError("file_weight must be in [0, 1]")
    if not 0 <= presence_logit_weight <= 1:
        raise ValueError("presence_logit_weight must be in [0, 1]")
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

    for row in rows:
        evidence = component_file_evidence(
            float(row["VOICE_FAKE_PROB"]), float(row["MUSIC_FAKE_PROB"]),
            float(row["VOICE_PRESENT_PROB"]), float(row["MUSIC_PRESENT_PROB"]),
            presence_logit_weight,
        )
        combined = _sigmoid(
            (1 - file_weight) * _logit(float(row["FILE_FAKE_PROB"]))
            + file_weight * _logit(evidence)
        )
        row["FILE_FAKE_PROB"] = round(float(combined), 10)

    temporary = submission_path.with_suffix(".tmp.csv")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(submission_path)
