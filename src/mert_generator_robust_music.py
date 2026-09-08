"""Dual-scale MERT Music-authenticity head and deployment-safe inference.

The input is the start/middle/end, per-layer mean/std tensor emitted by the
same MERT forward pass used by the existing SOFIA branch.  A fixed random
projection prevents a small training set from learning coordinate-specific
generator shortcuts.  Shallow/local and deep/long-horizon summaries are kept
as separate towers until the final Music logit.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


def make_projection(width: int, seed: int) -> torch.Tensor:
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    return torch.randn(768, width, generator=generator) / np.sqrt(768)


@torch.inference_mode()
def dual_scale_features(
    statistics: np.ndarray | torch.Tensor,
    projection: torch.Tensor,
    feature_version: str = "rich_v2",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return local fakeprint and long-horizon rhythm/harmony summaries.

    ``statistics`` must be ``[batch, 3 views, 13 layers, mean/std, 768]``.
    Local summaries emphasize shallow within-window dispersion, adjacent-layer
    changes and short temporal changes.  Long summaries preserve deep-layer
    start/middle/end direction and curvature rather than averaging the song.
    """
    values = torch.as_tensor(statistics, device=projection.device, dtype=torch.float32)
    if values.ndim != 5 or tuple(values.shape[1:]) != (3, 13, 2, 768):
        raise ValueError(f"unexpected MERT statistics shape: {tuple(values.shape)}")
    if projection.ndim != 2 or projection.shape[0] != 768:
        raise ValueError("MERT projection must have shape [768, width]")
    projected = F.layer_norm(values, (768,)) @ projection

    shallow = projected[:, :, :6]
    if feature_version == "compressed_v1":
        shallow_mean = shallow[:, :, :, 0]
        local = torch.cat((
            shallow_mean.mean(dim=(1, 2)),
            shallow[:, :, :, 1].mean(dim=(1, 2)),
            shallow_mean.std(dim=1, unbiased=False).mean(dim=1),
            (shallow_mean[:, :, 1:] - shallow_mean[:, :, :-1]).abs().mean(dim=(1, 2)),
            (shallow_mean[:, 1:] - shallow_mean[:, :-1]).abs().mean(dim=(1, 2)),
        ), dim=1)
        deep = projected[:, :, 6:]
        deep_mean = deep[:, :, :, 0]
        view_mean = deep_mean.mean(dim=2)
        long = torch.cat((
            view_mean.mean(dim=1),
            view_mean[:, 1] - view_mean[:, 0],
            view_mean[:, 2] - view_mean[:, 1],
            view_mean[:, 2] - 2 * view_mean[:, 1] + view_mean[:, 0],
            view_mean.std(dim=1, unbiased=False),
            (deep_mean[:, :, -1] - deep_mean[:, :, 0]).mean(dim=1),
            deep[:, :, :, 1].mean(dim=(1, 2)),
        ), dim=1)
        return local.float(), long.float()
    if feature_version != "rich_v2":
        raise ValueError(f"unknown MERT feature version: {feature_version}")
    # Preserve layer/stat identity. The compressed prototype averaged these
    # axes and lost shallow fakeprints that can transfer across generators.
    local = torch.cat((
        shallow.mean(dim=1).flatten(1),
        shallow.std(dim=1, unbiased=False).flatten(1),
    ), dim=1)

    deep = projected[:, :, 6:]
    deep_mean = deep[:, :, :, 0]
    long = torch.cat((
        deep.mean(dim=1).flatten(1),
        (deep_mean[:, 1] - deep_mean[:, 0]).flatten(1),
        (deep_mean[:, 2] - deep_mean[:, 1]).flatten(1),
        (deep_mean[:, 2] - 2 * deep_mean[:, 1] + deep_mean[:, 0]).flatten(1),
        deep_mean.std(dim=1, unbiased=False).flatten(1),
    ), dim=1)
    return local.float(), long.float()


class MertGeneratorRobustMusicHead(nn.Module):
    """Two low-capacity towers for local and compositional MERT evidence."""

    def __init__(
        self, local_dim: int, long_dim: int, hidden: int = 96,
        dropout: float = .15,
    ) -> None:
        super().__init__()
        self.local_tower = nn.Sequential(
            nn.LayerNorm(local_dim), nn.Linear(local_dim, hidden),
            nn.GELU(), nn.Dropout(dropout),
        )
        self.long_tower = nn.Sequential(
            nn.LayerNorm(long_dim), nn.Linear(long_dim, hidden),
            nn.GELU(), nn.Dropout(dropout),
        )
        self.local_output = nn.Linear(hidden, 1)
        self.long_output = nn.Linear(hidden, 1)
        self.fusion = nn.Sequential(
            nn.LayerNorm(hidden * 2), nn.Linear(hidden * 2, hidden),
            nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden, 1),
        )

    def forward(
        self, local: torch.Tensor, long: torch.Tensor, mode: str = "dual",
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        local_hidden = self.local_tower(local)
        long_hidden = self.long_tower(long)
        local_logit = self.local_output(local_hidden).squeeze(-1)
        long_logit = self.long_output(long_hidden).squeeze(-1)
        if mode == "local":
            logit = local_logit
            representation = local_hidden
        elif mode == "long":
            logit = long_logit
            representation = long_hidden
        elif mode == "dual":
            representation = torch.cat((local_hidden, long_hidden), dim=1)
            logit = self.fusion(representation).squeeze(-1)
        else:
            raise ValueError(f"unknown MERT feature mode: {mode}")
        return logit, local_logit, long_logit, representation


def asymmetric_codec_kl(
    codec_logits: torch.Tensor, clean_teacher_logits: torch.Tensor,
) -> torch.Tensor:
    """Clean-teacher to codec-student Bernoulli KL in stable fp32."""
    teacher = clean_teacher_logits.detach().float().sigmoid().clamp(1e-6, 1 - 1e-6)
    cross_entropy = F.binary_cross_entropy_with_logits(
        codec_logits.float(), teacher, reduction="none",
    )
    entropy = -(teacher * teacher.log() + (1 - teacher) * (1 - teacher).log())
    return (cross_entropy - entropy).mean()


def logit_residual(anchor, expert, weight: float) -> np.ndarray:
    """Convex, per-file Music-only fusion used for the fixed residual audit."""
    if not 0 <= weight <= 1:
        raise ValueError("weight must be in [0, 1]")
    anchor = np.clip(np.asarray(anchor, dtype=np.float64), 1e-5, 1 - 1e-5)
    expert = np.clip(np.asarray(expert, dtype=np.float64), 1e-5, 1 - 1e-5)
    anchor_logit = np.log(anchor) - np.log1p(-anchor)
    expert_logit = np.log(expert) - np.log1p(-expert)
    fused = (1 - weight) * anchor_logit + weight * expert_logit
    return np.exp(-np.logaddexp(0, -fused)).astype(np.float32)


def load_statistics_archive(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load runtime statistics written by ``SofiaMertDetector``."""
    with np.load(path, allow_pickle=False) as archive:
        required = {"ids", "statistics"}
        if not required.issubset(archive.files):
            raise ValueError(f"missing MERT statistic fields in {path}")
        ids = archive["ids"].astype(str)
        statistics = archive["statistics"].astype(np.float16, copy=True)
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate MERT statistic IDs")
    return ids, statistics


@torch.inference_mode()
def predict_mert_generator_robust_music(
    statistics: np.ndarray,
    checkpoint_path: Path,
    *,
    device: str = "cuda",
    batch_size: int = 128,
) -> np.ndarray:
    """Score a frozen head from an already-computed MERT temporal tensor."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint.get("model_type") != "mert_generator_robust_music":
        raise ValueError("not a generator-robust MERT Music checkpoint")
    target = torch.device(device)
    projection = torch.as_tensor(
        checkpoint["projection"], device=target, dtype=torch.float32,
    )
    model = MertGeneratorRobustMusicHead(**checkpoint["config"])
    model.load_state_dict(checkpoint["model"])
    model.to(target).eval()
    local_mean = torch.as_tensor(checkpoint["local_mean"], device=target)
    local_std = torch.as_tensor(checkpoint["local_std"], device=target)
    long_mean = torch.as_tensor(checkpoint["long_mean"], device=target)
    long_std = torch.as_tensor(checkpoint["long_std"], device=target)
    scores = []
    for start in range(0, len(statistics), batch_size):
        expected_local = int(checkpoint["config"]["local_dim"])
        feature_version = (
            "compressed_v1" if expected_local == 5 * projection.shape[1]
            else "rich_v2"
        )
        local, long = dual_scale_features(
            statistics[start:start + batch_size], projection, feature_version,
        )
        local = ((local - local_mean) / local_std).clamp_(-8, 8)
        long = ((long - long_mean) / long_std).clamp_(-8, 8)
        logit, _, _, _ = model(local, long, str(checkpoint["mode"]))
        scores.append(logit.sigmoid().float().cpu().numpy())
    return np.concatenate(scores)
