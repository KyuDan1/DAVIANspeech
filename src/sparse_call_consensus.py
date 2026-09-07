"""Temporal consensus between released XLS-R and Spectra anti-spoof heads."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np


def _logit(value: np.ndarray) -> np.ndarray:
    value = np.clip(np.asarray(value, dtype=np.float64), 1e-5, 1 - 1e-5)
    return np.log(value) - np.log1p(-value)


def _sigmoid(value: np.ndarray) -> np.ndarray:
    return np.exp(-np.logaddexp(0.0, -np.asarray(value, dtype=np.float64)))


def _segment_starts(num_samples: int, window: int) -> np.ndarray:
    if num_samples <= window:
        return np.asarray([0], dtype=np.int64)
    last = num_samples - window
    starts = list(range(0, last + 1, window))
    if starts[-1] != last:
        starts.append(last)
    return np.asarray(starts, dtype=np.int64)


def local_features(
    xlsr_probability: np.ndarray, spectra_margin: np.ndarray,
) -> np.ndarray:
    """Four compact agreement/disagreement features for aligned windows."""
    xlsr = _logit(xlsr_probability)
    spectra = np.asarray(spectra_margin, dtype=np.float64)
    return np.column_stack((
        xlsr, spectra, xlsr * spectra, np.abs(xlsr - spectra),
    ))


def aligned_features(
    xlsr: np.lib.npyio.NpzFile,
    spectra: np.lib.npyio.NpzFile,
    xlsr_index: int,
    spectra_index: int,
) -> np.ndarray:
    """Align each XLS-R window to the closest Spectra window in one file."""
    x_begin, x_end = xlsr["offsets"][xlsr_index:xlsr_index + 2]
    s_begin, s_end = spectra["offsets"][spectra_index:spectra_index + 2]
    if "starts" in xlsr.files:
        x_starts = xlsr["starts"][x_begin:x_end]
    else:
        x_starts = _segment_starts(
            int(round(float(xlsr["durations"][xlsr_index]) * 16_000)),
            int(xlsr["window"]),
        )
    if "starts" in spectra.files:
        s_starts = spectra["starts"][s_begin:s_end]
    else:
        s_starts = _segment_starts(
            int(round(float(spectra["durations"][spectra_index]) * 16_000)),
            int(spectra["window"]),
        )
    if not len(x_starts) or not len(s_starts):
        raise ValueError("Every valid audio file must have at least one window")
    nearest = np.abs(
        x_starts[:, None].astype(np.int64) - s_starts[None].astype(np.int64)
    ).argmin(axis=1)
    return local_features(
        xlsr["scores"][x_begin:x_end],
        spectra["fake_margins"][s_begin:s_end][nearest],
    )


def _logmeanexp(values: np.ndarray, temperature: float) -> float:
    scaled = temperature * np.asarray(values, dtype=np.float64)
    peak = float(scaled.max())
    return float(
        (peak + np.log(np.exp(scaled - peak).mean())) / temperature
    )


def score_statistics(
    xlsr_path: Path, spectra_path: Path, head_path: Path,
) -> tuple[np.ndarray, np.ndarray]:
    """Return IDs and calibrated file-level consensus probabilities."""
    xlsr = np.load(xlsr_path, allow_pickle=False)
    spectra = np.load(spectra_path, allow_pickle=False)
    head = np.load(head_path, allow_pickle=False)
    x_ids = xlsr["ids"].astype(str)
    s_ids = spectra["ids"].astype(str)
    if len(x_ids) != len(set(x_ids)) or len(s_ids) != len(set(s_ids)):
        raise ValueError("Statistic IDs must be unique")
    if set(x_ids) != set(s_ids):
        raise ValueError("XLS-R/Spectra statistic IDs differ")
    s_index = {item: index for index, item in enumerate(s_ids)}
    mean = head["feature_mean"].astype(np.float64)
    scale = head["feature_scale"].astype(np.float64)
    weight = head["local_weight"].astype(np.float64)
    bias = float(head["local_bias"])
    temperature = float(head["pool_temperature"])
    bag_weight = float(head["bag_weight"])
    bag_bias = float(head["bag_bias"])
    probabilities = []
    for index, item_id in enumerate(x_ids):
        if "valid" in spectra.files and not bool(
            spectra["valid"][s_index[item_id]]
        ):
            probabilities.append(np.nan)
            continue
        features = aligned_features(xlsr, spectra, index, s_index[item_id])
        standardized = (features - mean) / scale
        local_margin = standardized @ weight + bias
        pooled = _logmeanexp(local_margin, temperature)
        probabilities.append(_sigmoid(bag_weight * pooled + bag_bias))
    return x_ids, np.asarray(probabilities, dtype=np.float64)


def apply_sparse_call_consensus(
    submission_path: Path,
    xlsr_statistics_path: Path,
    spectra_statistics_path: Path,
    head_path: Path,
    voice_weight: float = 0.60,
    file_voice_only_weight: float = 0.0,
    voice_presence_gate: float = 0.50,
    music_absence_gate: float = 0.50,
) -> None:
    """Blend temporal Voice evidence and optionally restore File consistency.

    The file correction is deliberately restricted to confident voice-only
    predictions.  In that condition the competition's logical definition is
    simply ``FILE_FAKE == VOICE_FAKE``.  Later file-specific experts can put
    those two rankings on incompatible scales, especially after telephone
    transmission, so a conservative logit blend restores consistency without
    touching mixed or music-only audio.
    """
    for name, value in (
        ("voice_weight", voice_weight),
        ("file_voice_only_weight", file_voice_only_weight),
    ):
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"{name} must be in [0, 1]")
    ids, probabilities = score_statistics(
        xlsr_statistics_path, spectra_statistics_path, head_path
    )
    expert = dict(zip(ids, probabilities))
    with Path(submission_path).open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        columns = list(reader.fieldnames or [])
        rows = list(reader)
    if {row["ID"] for row in rows} != set(expert):
        raise ValueError("Submission/consensus statistic IDs differ")
    for row in rows:
        old_voice = float(row["VOICE_FAKE_PROB"])
        vote = float(expert[row["ID"]])
        if not np.isfinite(vote):
            continue
        combined_voice = _sigmoid(
            (1 - voice_weight) * _logit(old_voice)
            + voice_weight * _logit(vote)
        )
        row["VOICE_FAKE_PROB"] = round(float(combined_voice), 10)
        voice_only = (
            float(row["VOICE_PRESENT_PROB"]) >= voice_presence_gate
            and float(row["MUSIC_PRESENT_PROB"]) < music_absence_gate
        )
        if file_voice_only_weight > 0.0 and voice_only:
            old_file = float(row["FILE_FAKE_PROB"])
            combined_file = _sigmoid(
                (1 - file_voice_only_weight) * _logit(old_file)
                + file_voice_only_weight * _logit(combined_voice)
            )
            row["FILE_FAKE_PROB"] = round(float(combined_file), 10)
    temporary = Path(submission_path).with_suffix(".tmp.csv")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(submission_path)
