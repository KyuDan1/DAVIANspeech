"""Separation-free Music-only residual over cached component-query features."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

try:  # package imports in tests; flat imports in offline submissions.
    from .component_query_mhfa_inference import score_component_query_mhfa
except ImportError:  # pragma: no cover - exercised by script.py
    from component_query_mhfa_inference import score_component_query_mhfa


def _logit(values) -> np.ndarray:
    values = np.clip(np.asarray(values, dtype=np.float64), 1e-5, 1 - 1e-5)
    return np.log(values) - np.log1p(-values)


def _sigmoid(values) -> np.ndarray:
    return np.exp(-np.logaddexp(0, -np.asarray(values, dtype=np.float64)))


def logit_ensemble(probabilities: list[np.ndarray]) -> np.ndarray:
    """Average member evidence in logit space without fitting a stacker."""
    if not probabilities:
        raise ValueError("at least one Music expert is required")
    values = [np.asarray(item, dtype=np.float64) for item in probabilities]
    if any(item.shape != values[0].shape for item in values[1:]):
        raise ValueError("Music expert shapes differ")
    return _sigmoid(np.mean([_logit(item) for item in values], axis=0))


def fuse_music_logits(anchor, expert, weight: float) -> np.ndarray:
    """Return a bounded convex Music-only logit mixture."""
    if not 0 <= weight <= 1:
        raise ValueError("Music residual weight must lie in [0,1]")
    anchor = np.asarray(anchor, dtype=np.float64)
    expert = np.asarray(expert, dtype=np.float64)
    if anchor.shape != expert.shape:
        raise ValueError("anchor and expert Music probabilities differ in shape")
    return _sigmoid((1 - weight) * _logit(anchor) + weight * _logit(expert))


def apply_music_probability_fusion(
    submission_path: Path,
    expert_ids: np.ndarray,
    expert_probability: np.ndarray,
    *,
    music_weight: float,
) -> None:
    """Update only ``MUSIC_FAKE_PROB`` after strict ID alignment."""
    expert_ids = np.asarray(expert_ids).astype(str)
    expert_probability = np.asarray(expert_probability, dtype=np.float64)
    if expert_probability.shape != (len(expert_ids),):
        raise ValueError("Music expert must return one probability per ID")
    if len(set(expert_ids)) != len(expert_ids):
        raise ValueError("Music expert contains duplicate IDs")
    with Path(submission_path).open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        columns = list(reader.fieldnames or [])
        rows = list(reader)
    row_ids = [row["ID"] for row in rows]
    if len(set(row_ids)) != len(row_ids):
        raise ValueError("submission contains duplicate IDs")
    if set(row_ids) != set(expert_ids):
        raise ValueError("submission and Music expert IDs differ")
    position = {item: index for index, item in enumerate(expert_ids)}
    ordered = expert_probability[[position[item] for item in row_ids]]
    anchor = np.asarray(
        [float(row["MUSIC_FAKE_PROB"]) for row in rows], dtype=np.float64
    )
    fused = fuse_music_logits(anchor, ordered, music_weight)
    for row, value in zip(rows, fused):
        row["MUSIC_FAKE_PROB"] = round(float(value), 10)
    temporary = Path(submission_path).with_suffix(".tmp.csv")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(submission_path)


def apply_component_query_music_residual(
    submission_path: Path,
    eat_statistics_path: Path,
    spear_statistics_path: Path,
    checkpoint_paths: list[Path],
    *,
    device: str = "cuda",
    music_weight: float,
) -> None:
    """Score fixed heads from existing caches and change Music only."""
    member_ids: np.ndarray | None = None
    members: list[np.ndarray] = []
    for checkpoint_path in checkpoint_paths:
        ids, probability = score_component_query_mhfa(
            eat_statistics_path,
            spear_statistics_path,
            checkpoint_path,
            device=device,
        )
        ids = ids.astype(str)
        if member_ids is None:
            member_ids = ids
        elif not np.array_equal(member_ids, ids):
            raise ValueError("Music checkpoint ID order differs")
        members.append(probability[:, 1])
    if member_ids is None:
        raise ValueError("at least one Music checkpoint is required")
    apply_music_probability_fusion(
        submission_path,
        member_ids,
        logit_ensemble(members),
        music_weight=music_weight,
    )
