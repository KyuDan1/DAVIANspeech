"""Compact temporal statistics for separation-free audio deepfake detection.

The frozen SSL encoders produce hundreds of frame tokens per crop.  Keeping
only their mean discards short-lived synthesis artefacts, while storing every
token makes repeated experiments unnecessarily expensive.  This module keeps
four complementary per-channel statistics that are cheap enough to cache and
deploy:

* mean: semantic/acoustic location;
* standard deviation: within-crop variability;
* mean absolute temporal delta: high-rate modulation/codec discontinuities;
* log Teager--Kaiser energy: local signal texture.

No source separation is involved.  All statistics are computed on tokens from
the original mixture.
"""

from __future__ import annotations

import numpy as np
import torch


STAT_NAMES = ("mean", "std", "delta", "tkeo")


def temporal_starts(num_samples: int, crop_samples: int, max_views: int = 3) -> list[int]:
    """Return deterministic start/middle/end crop positions."""
    if crop_samples <= 0:
        raise ValueError("crop_samples must be positive")
    if max_views <= 0:
        raise ValueError("max_views must be positive")
    if num_samples <= crop_samples:
        return [0]
    last = num_samples - crop_samples
    candidates = np.linspace(0, last, max_views, dtype=np.int64)
    return sorted({int(value) for value in candidates})


def interval_view_targets(
    num_samples: int,
    crop_samples: int,
    voice_interval: tuple[int, int],
    music_interval: tuple[int, int],
    voice_fake: int,
    music_fake: int,
    max_views: int = 3,
    minimum_overlap_samples: int = 1_600,
) -> tuple[np.ndarray, np.ndarray]:
    """Create local voice/music/file labels for deterministic crop views.

    A component contributes to a view only when at least 100 ms overlaps by
    default. This prevents a one-frame boundary touch from turning an entire
    6--10 second crop positive.
    """
    starts = temporal_starts(num_samples, crop_samples, max_views)
    targets = np.zeros((max_views, 3), dtype=np.float32)
    mask = np.zeros(max_views, dtype=bool)
    intervals = (voice_interval, music_interval)
    labels = (int(voice_fake), int(music_fake))
    for view, start in enumerate(starts):
        end = min(start + crop_samples, num_samples)
        mask[view] = True
        local = []
        for (component_start, component_end), label in zip(intervals, labels):
            overlap = max(0, min(end, component_end) - max(start, component_start))
            local.append(float(label and overlap >= minimum_overlap_samples))
        targets[view, 0], targets[view, 1] = local
        targets[view, 2] = max(local)
    return targets, mask


def crop_or_pad(audio: np.ndarray, start: int, samples: int) -> np.ndarray:
    """Take one crop and zero-pad short inputs without repeating artefacts."""
    audio = np.asarray(audio, dtype=np.float32)
    crop = audio[start:start + samples]
    if crop.size < samples:
        crop = np.pad(crop, (0, samples - crop.size))
    return crop.astype(np.float32, copy=False)


def sequence_statistics(sequence: torch.Tensor) -> torch.Tensor:
    """Return ``[..., 4, channels]`` statistics for ``[..., time, channels]``."""
    if sequence.ndim < 2:
        raise ValueError("sequence must have time and channel dimensions")
    x = sequence.float()
    mean = x.mean(dim=-2)
    std = x.std(dim=-2, unbiased=False)
    if x.shape[-2] < 2:
        delta = torch.zeros_like(mean)
    else:
        delta = (x[..., 1:, :] - x[..., :-1, :]).abs().mean(dim=-2)
    if x.shape[-2] < 3:
        tkeo = torch.zeros_like(mean)
    else:
        centre = x[..., 1:-1, :]
        energy = centre.square() - x[..., :-2, :] * x[..., 2:, :]
        tkeo = torch.log1p(energy.abs().mean(dim=-2))
    return torch.stack((mean, std, delta, tkeo), dim=-2)


def pad_views(
    views: list[torch.Tensor], max_views: int, shape: tuple[int, ...]
) -> tuple[np.ndarray, np.ndarray]:
    """Pad a list of statistic tensors and return data plus a validity mask."""
    result = np.zeros((max_views, *shape), dtype=np.float16)
    mask = np.zeros(max_views, dtype=bool)
    for index, value in enumerate(views[:max_views]):
        result[index] = value.detach().cpu().numpy().astype(np.float16)
        mask[index] = True
    return result, mask
