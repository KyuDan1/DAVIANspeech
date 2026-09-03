"""File-local consistency for temporally separated voice/music mixtures.

The competition defines a file as fake whenever either present component is
fake.  The correction here is intentionally narrow: it only acts when both
components are predicted present, their stem energies look temporally
localized, and the telephone router does not identify a narrow-band channel.
Stem energy controls routing only; it is never used as authenticity evidence.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np


EPS = 1e-10


def _logit(value: np.ndarray | float) -> np.ndarray:
    value = np.clip(np.asarray(value, dtype=np.float64), 1e-5, 1 - 1e-5)
    return np.log(value) - np.log1p(-value)


def _sigmoid(value: np.ndarray | float) -> np.ndarray:
    return np.exp(-np.logaddexp(0.0, -np.asarray(value, dtype=np.float64)))


def _frame_rms(
    values: np.ndarray, frame: int = 8_000, hop: int = 4_000,
) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32).reshape(-1)
    if len(values) < frame:
        values = np.pad(values, (0, frame - len(values)))
    starts = list(range(0, max(len(values) - frame + 1, 1), hop))
    last = max(len(values) - frame, 0)
    if starts[-1] != last:
        starts.append(last)
    return np.asarray([
        np.sqrt(np.mean(np.square(values[start:start + frame], dtype=np.float64)))
        for start in starts
    ])


def stem_layout_score(
    original: np.ndarray, voice: np.ndarray, music: np.ndarray,
) -> float:
    """Return a scale-free cue for sequential/localized component activity.

    For each stem, compare its 90th- and 50th-percentile frame RMS relative to
    the original mixture.  A large gap means that one component is concentrated
    in only part of the file, as in sequential speech/music or hold music.
    """
    length = min(len(original), len(voice), len(music))
    if length < 1:
        return 0.0
    original_rms = _frame_rms(original[:length])
    voice_rms = _frame_rms(voice[:length])
    music_rms = _frame_rms(music[:length])
    count = min(len(original_rms), len(voice_rms), len(music_rms))
    original_rms = original_rms[:count]
    ratios = (
        voice_rms[:count] / (original_rms + EPS),
        music_rms[:count] / (original_rms + EPS),
    )
    gaps = [
        np.log1p(np.quantile(values, 0.90))
        - np.log1p(np.quantile(values, 0.50))
        for values in ratios
    ]
    result = float(max(gaps))
    if not np.isfinite(result):
        raise ValueError("non-finite stem layout score")
    return result


def save_layout_scores(path: Path, ids: list[str], scores: list[float]) -> None:
    """Write one deterministic layout score for each independently read file."""
    if len(ids) != len(scores) or len(ids) != len(set(ids)):
        raise ValueError("layout IDs must be unique and match score count")
    values = np.asarray(scores, dtype=np.float32)
    if not np.isfinite(values).all():
        raise ValueError("layout scores must be finite")
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, ids=np.asarray(ids), scores=values)


def apply_mixed_component_consistency(
    submission_path: Path,
    layout_statistics_path: Path,
    telephone_ids_path: Path | None = None,
    layout_threshold: float = 0.35,
    file_weight: float = 0.70,
    presence_threshold: float = 0.50,
) -> None:
    """Blend File toward component OR for routed non-telephone mixtures."""
    if not 0.0 <= file_weight <= 1.0:
        raise ValueError("file_weight must be in [0, 1]")
    if layout_threshold < 0.0:
        raise ValueError("layout_threshold must be non-negative")
    if not 0.0 <= presence_threshold <= 1.0:
        raise ValueError("presence_threshold must be in [0, 1]")

    state = np.load(layout_statistics_path, allow_pickle=False)
    ids = state["ids"].astype(str)
    scores = state["scores"].astype(np.float64)
    if len(ids) != len(scores) or len(ids) != len(set(ids)):
        raise ValueError("layout statistic IDs must be unique")
    if not np.isfinite(scores).all():
        raise ValueError("layout scores must be finite")
    layout = dict(zip(ids, scores))

    telephone_ids: set[str] = set()
    if telephone_ids_path is not None and Path(telephone_ids_path).is_file():
        routed = np.load(telephone_ids_path, allow_pickle=False)["ids"].astype(str)
        telephone_ids = set(routed)
        if len(telephone_ids) != len(routed):
            raise ValueError("telephone IDs must be unique")

    with Path(submission_path).open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        columns = list(reader.fieldnames or [])
        rows = list(reader)
    row_ids = {row["ID"] for row in rows}
    if row_ids != set(layout):
        raise ValueError("submission/layout statistic IDs differ")
    if not telephone_ids.issubset(row_ids):
        raise ValueError("telephone IDs are not a submission subset")

    changed = 0
    for row in rows:
        item_id = row["ID"]
        mixed = (
            float(row["VOICE_PRESENT_PROB"]) >= presence_threshold
            and float(row["MUSIC_PRESENT_PROB"]) >= presence_threshold
        )
        routed = layout[item_id] >= layout_threshold
        if not (mixed and routed and item_id not in telephone_ids):
            continue
        component_or = max(
            float(row["VOICE_FAKE_PROB"]), float(row["MUSIC_FAKE_PROB"]),
        )
        combined = _sigmoid(
            (1 - file_weight) * _logit(float(row["FILE_FAKE_PROB"]))
            + file_weight * _logit(component_or)
        )
        row["FILE_FAKE_PROB"] = round(float(combined), 10)
        changed += 1

    temporary = Path(submission_path).with_suffix(".tmp.csv")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(submission_path)
    print(f"mixed component consistency routed {changed}/{len(rows)} files")
