"""Joint voice/music MIL head for separation-free SPEAR temporal bins."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class SpearTemporalJointHead(nn.Module):
    """Predict local component activity and authenticity with a shared trunk."""

    def __init__(
        self, feature_dimension: int, mean: torch.Tensor, std: torch.Tensor,
        hidden: int = 96, dropout: float = .15, temperature: float = 5.0,
        minimum_presence_weight: float = .05,
    ) -> None:
        super().__init__()
        if feature_dimension <= 0 or hidden <= 0 or temperature <= 0:
            raise ValueError("feature/hidden dimensions and temperature must be positive")
        self.feature_dimension = int(feature_dimension)
        self.temperature = float(temperature)
        self.minimum_presence_weight = float(minimum_presence_weight)
        self.register_buffer("mean", mean.reshape(feature_dimension).float())
        self.register_buffer("std", std.reshape(feature_dimension).float())
        self.network = nn.Sequential(
            nn.Linear(feature_dimension, hidden), nn.LayerNorm(hidden), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(hidden, 4),
        )

    def bin_logits(self, features: torch.Tensor) -> torch.Tensor:
        normalized = ((features.float() - self.mean) / self.std).clamp_(-8, 8)
        return self.network(normalized)

    def _lme(self, probability: torch.Tensor, weight: torch.Tensor,
             mask: torch.Tensor) -> torch.Tensor:
        weight = weight * mask.to(weight.dtype)
        scaled = self.temperature * probability
        scaled = scaled.masked_fill(~mask, -1e4)
        return (
            (
                torch.logsumexp(scaled + weight.clamp_min(1e-8).log(), dim=1)
                - weight.sum(dim=1).clamp_min(1e-8).log()
            ) / self.temperature
        ).clamp(1e-5, 1 - 1e-5)

    def aggregate(self, logits: torch.Tensor, mask: torch.Tensor):
        if logits.shape[:-1] != mask.shape or logits.shape[-1] != 4:
            raise ValueError("joint logits and mask have incompatible shapes")
        local = logits.sigmoid()
        voice_presence = self._lme(
            local[..., 0], torch.ones_like(local[..., 0]), mask
        )
        music_presence = self._lme(
            local[..., 1], torch.ones_like(local[..., 1]), mask
        )
        voice_weight = self.minimum_presence_weight + (
            1 - self.minimum_presence_weight
        ) * local[..., 0].detach()
        music_weight = self.minimum_presence_weight + (
            1 - self.minimum_presence_weight
        ) * local[..., 1].detach()
        voice_fake = self._lme(local[..., 2], voice_weight, mask)
        music_fake = self._lme(local[..., 3], music_weight, mask)
        voice_evidence = voice_presence * voice_fake
        music_evidence = music_presence * music_fake
        file_fake = (
            1 - (1 - voice_evidence) * (1 - music_evidence)
        ).clamp(1e-5, 1 - 1e-5)
        return file_fake, voice_fake, music_fake, voice_presence, music_presence

    def forward(self, features: torch.Tensor, mask: torch.Tensor):
        batch = features.shape[0]
        flat = features.reshape(batch, -1, features.shape[-1])
        flat_mask = mask.reshape(batch, -1)
        logits = self.bin_logits(flat)
        return logits, self.aggregate(logits, flat_mask)


def joint_temporal_loss(
    logits: torch.Tensor,
    outputs: tuple[torch.Tensor, ...],
    mask: torch.Tensor,
    local_voice_presence: torch.Tensor,
    local_music_presence: torch.Tensor,
    local_voice_fake: torch.Tensor,
    local_music_fake: torch.Tensor,
    file_fake: torch.Tensor,
    voice_fake: torch.Tensor,
    music_fake: torch.Tensor,
    voice_present: torch.Tensor,
    music_present: torch.Tensor,
    sample_weight: torch.Tensor,
    local_weight: float = .25,
    presence_weight: float = .15,
    file_weight: float = .50,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Balanced local and file-level multitask objective."""
    predicted_file, predicted_voice, predicted_music, predicted_vp, predicted_mp = outputs
    valid = mask.bool()
    presence = .5 * (
        F.binary_cross_entropy_with_logits(
            logits[..., 0][valid], local_voice_presence[valid]
        )
        + F.binary_cross_entropy_with_logits(
            logits[..., 1][valid], local_music_presence[valid]
        )
    )
    voice_bins = valid & local_voice_presence.bool()
    music_bins = valid & local_music_presence.bool()
    local = .5 * (
        F.binary_cross_entropy_with_logits(
            logits[..., 2][voice_bins], local_voice_fake[voice_bins]
        )
        + F.binary_cross_entropy_with_logits(
            logits[..., 3][music_bins], local_music_fake[music_bins]
        )
    )

    def weighted_bce(prediction, target, selected=None):
        loss = F.binary_cross_entropy(prediction, target, reduction="none")
        weights = sample_weight
        if selected is not None:
            loss, weights = loss[selected], weights[selected]
        return (loss * weights).sum() / weights.sum().clamp_min(1e-8)

    component = .5 * (
        weighted_bce(predicted_voice, voice_fake, voice_present.bool())
        + weighted_bce(predicted_music, music_fake, music_present.bool())
    )
    file_loss = weighted_bce(predicted_file, file_fake)
    global_presence = .5 * (
        weighted_bce(predicted_vp, voice_present)
        + weighted_bce(predicted_mp, music_present)
    )
    total = (
        component + file_weight * file_loss + local_weight * local
        + presence_weight * (presence + global_presence)
    )
    return total, {
        "component": component.detach(), "file": file_loss.detach(),
        "local": local.detach(), "presence": presence.detach(),
        "global_presence": global_presence.detach(),
    }
