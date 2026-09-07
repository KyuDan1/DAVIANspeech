"""Small separation-free MIL head for SPEAR temporal-bin features."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class SpearTemporalBinHead(nn.Module):
    """Shared bin classifier with presence-weighted soft-max aggregation."""

    def __init__(
        self,
        feature_dimension: int,
        mean: torch.Tensor,
        std: torch.Tensor,
        hidden: int = 0,
        dropout: float = 0.10,
        temperature: float = 5.0,
        minimum_presence_weight: float = 0.05,
    ) -> None:
        super().__init__()
        if feature_dimension <= 0 or temperature <= 0:
            raise ValueError("feature dimension and temperature must be positive")
        if not 0 <= minimum_presence_weight <= 1:
            raise ValueError("minimum presence weight must lie in [0, 1]")
        self.feature_dimension = feature_dimension
        self.temperature = float(temperature)
        self.minimum_presence_weight = float(minimum_presence_weight)
        self.register_buffer("mean", mean.reshape(feature_dimension).float())
        self.register_buffer("std", std.reshape(feature_dimension).float())
        if hidden > 0:
            self.network = nn.Sequential(
                nn.Linear(feature_dimension, hidden), nn.GELU(),
                nn.Dropout(dropout), nn.Linear(hidden, 2),
            )
        else:
            self.network = nn.Linear(feature_dimension, 2)

    def bin_logits(self, features: torch.Tensor) -> torch.Tensor:
        if features.shape[-1] != self.feature_dimension:
            raise ValueError("temporal-bin feature dimension is incompatible")
        normalized = ((features.float() - self.mean) / self.std).clamp_(-8, 8)
        return self.network(normalized)

    def aggregate(
        self, logits: torch.Tensor, mask: torch.Tensor,
    ) -> torch.Tensor:
        """Return one music-fake probability for each file.

        Presence probabilities softly suppress speech-only/silent bins.  They
        are detached for aggregation during training so the authenticity loss
        cannot obtain an easy shortcut by corrupting the presence router.
        """
        if logits.shape[:-1] != mask.shape or logits.shape[-1] != 2:
            raise ValueError("logits and temporal mask have incompatible shapes")
        presence = logits[..., 0].sigmoid().detach()
        weight = (
            self.minimum_presence_weight
            + (1 - self.minimum_presence_weight) * presence
        ) * mask.to(logits.dtype)
        fake_probability = logits[..., 1].sigmoid()
        scaled = self.temperature * fake_probability
        scaled = scaled.masked_fill(~mask, -1e4)
        log_weight = weight.clamp_min(1e-8).log()
        numerator = torch.logsumexp(scaled + log_weight, dim=1)
        denominator = weight.sum(dim=1).clamp_min(1e-8).log()
        return ((numerator - denominator) / self.temperature).clamp(1e-5, 1 - 1e-5)

    def forward(self, features: torch.Tensor, mask: torch.Tensor):
        batch = features.shape[0]
        flat = features.reshape(batch, -1, features.shape[-1])
        flat_mask = mask.reshape(batch, -1)
        logits = self.bin_logits(flat)
        return logits, self.aggregate(logits, flat_mask)


def temporal_bin_loss(
    logits: torch.Tensor,
    file_probability: torch.Tensor,
    valid_mask: torch.Tensor,
    music_present_targets: torch.Tensor,
    music_fake_targets: torch.Tensor,
    file_music_fake: torch.Tensor,
    file_music_present: torch.Tensor,
    sample_weight: torch.Tensor,
    presence_weight: float = 0.20,
    local_fake_weight: float = 0.50,
    environments: list[tuple[torch.Tensor, torch.Tensor]] | None = None,
    group_dro_temperature: float = 0.0,
    consistency_pairs: torch.Tensor | None = None,
    consistency_weight: float = 0.0,
    balance_local_losses: bool = False,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Joint local-presence, local-authenticity, and file MIL objective."""
    selected = valid_mask.bool()
    if balance_local_losses:
        raw_presence = F.binary_cross_entropy_with_logits(
            logits[..., 0], music_present_targets, reduction="none"
        )
        presence_per_file = (
            (raw_presence * selected).sum(dim=1)
            / selected.sum(dim=1).clamp_min(1)
        )
        presence_loss = (
            presence_per_file * sample_weight
        ).sum() / sample_weight.sum().clamp_min(1e-8)
    else:
        presence_loss = F.binary_cross_entropy_with_logits(
            logits[..., 0][selected], music_present_targets[selected]
        )
    fake_selected = selected & music_present_targets.bool()
    if not fake_selected.any():
        raise ValueError("batch has no music-present temporal bins")
    if balance_local_losses:
        raw_local_fake = F.binary_cross_entropy_with_logits(
            logits[..., 1], music_fake_targets, reduction="none"
        )
        local_per_file = (
            (raw_local_fake * fake_selected).sum(dim=1)
            / fake_selected.sum(dim=1).clamp_min(1)
        )
        local_file_selected = fake_selected.any(dim=1)
        local_weight = sample_weight[local_file_selected]
        local_fake_loss = (
            local_per_file[local_file_selected] * local_weight
        ).sum() / local_weight.sum().clamp_min(1e-8)
    else:
        local_fake_loss = F.binary_cross_entropy_with_logits(
            logits[..., 1][fake_selected], music_fake_targets[fake_selected]
        )
    file_selected = file_music_present.bool()
    per_file = F.binary_cross_entropy(
        file_probability[file_selected], file_music_fake[file_selected], reduction="none"
    )
    weighted = sample_weight[file_selected]
    file_loss = (per_file * weighted).sum() / weighted.sum().clamp_min(1e-8)
    if environments and group_dro_temperature > 0:
        full_loss = F.binary_cross_entropy(
            file_probability, file_music_fake, reduction="none"
        )
        environment_loss = torch.stack([
            0.5 * full_loss[positive].mean() + 0.5 * full_loss[negative].mean()
            for positive, negative in environments
        ])
        temperature = float(group_dro_temperature)
        file_loss = temperature * (
            torch.logsumexp(environment_loss / temperature, dim=0)
            - torch.log(environment_loss.new_tensor(float(len(environment_loss))))
        )
    consistency = logits.new_zeros(())
    if (
        consistency_pairs is not None and consistency_pairs.numel()
        and consistency_weight > 0
    ):
        probability_logit = torch.logit(file_probability, eps=1e-5)
        consistency = F.smooth_l1_loss(
            probability_logit[consistency_pairs[:, 0]],
            probability_logit[consistency_pairs[:, 1]],
        )
    total = (
        file_loss + local_fake_weight * local_fake_loss
        + presence_weight * presence_loss + consistency_weight * consistency
    )
    return total, {
        "file": file_loss.detach(), "local_fake": local_fake_loss.detach(),
        "presence": presence_loss.detach(), "consistency": consistency.detach(),
    }
