"""Speech-presence-aware Voice authenticity MIL on original-mixture SPEAR bins."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class SpearVoiceMILHead(nn.Module):
    """Predict local speech presence and fake evidence before weighted pooling."""

    def __init__(
        self,
        feature_dimension: int,
        mean: torch.Tensor,
        std: torch.Tensor,
        *,
        hidden: int = 96,
        dropout: float = 0.15,
        temperature: float = 2.0,
        minimum_presence_weight: float = 0.05,
    ) -> None:
        super().__init__()
        if feature_dimension <= 0 or hidden <= 0 or temperature <= 0:
            raise ValueError("feature dimensions and temperature must be positive")
        if not 0 <= minimum_presence_weight <= 1:
            raise ValueError("minimum presence weight must lie in [0, 1]")
        self.feature_dimension = int(feature_dimension)
        self.temperature = float(temperature)
        self.minimum_presence_weight = float(minimum_presence_weight)
        self.register_buffer("mean", mean.reshape(feature_dimension).float())
        self.register_buffer("std", std.reshape(feature_dimension).float())
        self.network = nn.Sequential(
            nn.Linear(feature_dimension, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 2),
        )

    def local_logits(self, features: torch.Tensor) -> torch.Tensor:
        normalized = ((features.float() - self.mean) / self.std).clamp(-8, 8)
        return self.network(normalized)

    def aggregate(
        self, logits: torch.Tensor, mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return file Voice probability and differentiable presence weights."""
        if logits.shape[:-1] != mask.shape or logits.shape[-1] != 2:
            raise ValueError("Voice MIL logits and mask have incompatible shapes")
        mask = mask.bool()
        if not mask.any(dim=1).all():
            raise ValueError("every Voice MIL item needs at least one valid bin")
        presence = logits[..., 0].sigmoid()
        weight = self.minimum_presence_weight + (
            1 - self.minimum_presence_weight
        ) * presence
        weight = weight * mask.to(weight.dtype)
        fake_logit = logits[..., 1]
        temperature = self.temperature
        pooled_logit = (
            torch.logsumexp(
                temperature * fake_logit + weight.clamp_min(1e-8).log(), dim=1,
            )
            - weight.sum(dim=1).clamp_min(1e-8).log()
        ) / temperature
        return pooled_logit.sigmoid(), weight

    def forward(
        self, features: torch.Tensor, mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if features.ndim < 3:
            raise ValueError("Voice MIL features must include batch and segment axes")
        batch = features.shape[0]
        flattened = features.reshape(batch, -1, features.shape[-1])
        flat_mask = mask.reshape(batch, -1)
        logits = self.local_logits(flattened)
        probability, weights = self.aggregate(logits, flat_mask)
        return probability, logits, weights


def spear_voice_mil_loss(
    probability: torch.Tensor,
    logits: torch.Tensor,
    mask: torch.Tensor,
    local_presence: torch.Tensor,
    local_fake: torch.Tensor,
    voice_fake: torch.Tensor,
    sample_weight: torch.Tensor,
    *,
    local_presence_weight: float = 0.20,
    local_fake_weight: float = 0.40,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Balanced file loss with explicitly supervised local speech routing."""
    if local_presence_weight < 0 or local_fake_weight < 0:
        raise ValueError("Voice MIL loss weights must be non-negative")
    mask = mask.bool()
    weights = sample_weight / sample_weight.sum().clamp_min(1e-8)
    bag = (
        F.binary_cross_entropy(probability, voice_fake, reduction="none") * weights
    ).sum()
    presence = F.binary_cross_entropy_with_logits(
        logits[..., 0][mask], local_presence[mask]
    )
    speech_bins = mask & local_presence.bool()
    if speech_bins.any():
        local_fake_loss = F.binary_cross_entropy_with_logits(
            logits[..., 1][speech_bins], local_fake[speech_bins]
        )
    else:
        local_fake_loss = bag.new_zeros(())
    total = (
        bag + local_presence_weight * presence + local_fake_weight * local_fake_loss
    )
    return total, {
        "bag": bag.detach(),
        "local_presence": presence.detach(),
        "local_fake": local_fake_loss.detach(),
    }
