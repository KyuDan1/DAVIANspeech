"""Compact temporal-bin features from frozen SPEAR hidden states.

The competition audio can contain speech and music concurrently or in
different time intervals.  File-level pooling loses that layout.  These
helpers keep a small sequence of statistics from the *original mixture*;
they never run source separation or modify the encoder representation.
"""

from __future__ import annotations

import numpy as np
import torch
from torch.nn import functional as F

try:  # package import in tests; flat import in the offline submission
    from .dual_domain_stats import sequence_statistics
except ImportError:  # pragma: no cover - exercised by script.py
    from dual_domain_stats import sequence_statistics


DEFAULT_LAYERS = (0, 1, 2, 3)
DEFAULT_BINS = 8
DEFAULT_WIDTH = 64
DEFAULT_SEED = 20260903


def random_projection(
    dimension: int = 1280,
    width: int = DEFAULT_WIDTH,
    seed: int = DEFAULT_SEED,
) -> np.ndarray:
    """Return a deterministic, label-free Gaussian JL projection."""
    if dimension <= 0 or width <= 0:
        raise ValueError("projection dimensions must be positive")
    generator = np.random.default_rng(seed)
    return (
        generator.standard_normal((dimension, width), dtype=np.float32)
        / np.sqrt(dimension)
    ).astype(np.float32)


def temporal_bin_boundaries(frames: int, bins: int = DEFAULT_BINS) -> list[tuple[int, int]]:
    """Split encoder frames into non-empty, deterministic contiguous bins."""
    if frames <= 0 or bins <= 0:
        raise ValueError("frames and bins must be positive")
    if frames < bins:
        raise ValueError("there must be at least one encoder frame per bin")
    edges = np.linspace(0, frames, bins + 1, dtype=np.int64)
    return [(int(edges[index]), int(edges[index + 1])) for index in range(bins)]


@torch.inference_mode()
def projected_temporal_bins(
    hidden_states: list[torch.Tensor] | tuple[torch.Tensor, ...],
    projection: torch.Tensor,
    layers: tuple[int, ...] = DEFAULT_LAYERS,
    bins: int = DEFAULT_BINS,
) -> torch.Tensor:
    """Pool hidden states to ``[batch, bins, layers, stats, width]``.

    Per-statistic LayerNorm makes the fixed projection less sensitive to
    codec-dependent absolute scale.  The four statistics are defined in
    :mod:`dual_domain_stats` and retain local level, variation, modulation,
    and nonlinear energy cues.
    """
    if not layers:
        raise ValueError("at least one SPEAR layer is required")
    if min(layers) < 0 or max(layers) >= len(hidden_states):
        raise ValueError("requested SPEAR layer is unavailable")
    values = torch.stack([hidden_states[index] for index in layers], dim=1).float()
    if values.ndim != 4:
        raise ValueError("hidden states must have [batch, time, channels] shape")
    if projection.ndim != 2 or projection.shape[0] != values.shape[-1]:
        raise ValueError("projection has an incompatible input dimension")
    features = []
    for start, end in temporal_bin_boundaries(values.shape[-2], bins):
        statistics = sequence_statistics(values[:, :, start:end, :])
        normalized = F.layer_norm(statistics, (statistics.shape[-1],))
        features.append(normalized @ projection)
    return torch.stack(features, dim=1)


def audio_bin_ranges(
    num_samples: int,
    view_start: int,
    crop_samples: int,
    bins: int = DEFAULT_BINS,
) -> tuple[np.ndarray, np.ndarray]:
    """Return global sample ranges and validity for the bins in one crop."""
    if num_samples <= 0 or crop_samples <= 0 or bins <= 0:
        raise ValueError("audio, crop, and bin sizes must be positive")
    edges = np.linspace(0, crop_samples, bins + 1, dtype=np.int64)
    ranges = np.zeros((bins, 2), dtype=np.int32)
    mask = np.zeros(bins, dtype=bool)
    for index in range(bins):
        start = min(num_samples, view_start + int(edges[index]))
        end = min(num_samples, view_start + int(edges[index + 1]))
        ranges[index] = (start, end)
        mask[index] = end > start
    return ranges, mask
