"""Dedicated original-mixture Music authenticity head.

Unlike the three-task invariant head, this module has one attentive query and
one primary representation for Music.  Voice/File logits are low-weight
training auxiliaries only; they are never fused into competition outputs.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

try:
    from .dual_domain_head import MultiTaskAttentiveStats, SwiGLUProjection
except ImportError:  # pragma: no cover - offline flat-module package
    from dual_domain_head import MultiTaskAttentiveStats, SwiGLUProjection


class ChannelInvariantMusicHead(nn.Module):
    """Single-query EAT/SPEAR head with auxiliary Voice/File stabilization."""

    def __init__(
        self,
        width: int = 128,
        heads: int = 4,
        dropout: float = .25,
        stream_dropout: float = .05,
    ) -> None:
        super().__init__()
        self.width = int(width)
        self.stream_dropout = float(stream_dropout)
        self.eat_projection = SwiGLUProjection(768, width, dropout)
        self.spear_projection = SwiGLUProjection(1280, width, dropout)
        self.stat_embedding = nn.Parameter(torch.empty(4, width))
        self.spear_layer_embedding = nn.Parameter(torch.empty(13, width))
        self.view_embedding = nn.Parameter(torch.empty(3, width))
        self.stream_embedding = nn.Parameter(torch.empty(2, width))
        for parameter in (
            self.stat_embedding, self.spear_layer_embedding,
            self.view_embedding, self.stream_embedding,
        ):
            nn.init.normal_(parameter, std=.02)
        self.music_pool = MultiTaskAttentiveStats(width, tasks=1, heads=heads)
        self.music_mlp = nn.Sequential(
            nn.LayerNorm(width * 2),
            nn.Linear(width * 2, width),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.music_output = nn.Linear(width, 1)
        # Auxiliary outputs regularize the shared representation, but Music is
        # not averaged with either of them at inference time.
        self.auxiliary_output = nn.Linear(width, 2)  # Voice, File

    def _stream_masks(
        self, eat_mask: torch.Tensor, spear_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not self.training or self.stream_dropout <= 0:
            return eat_mask, spear_mask
        random = torch.rand(eat_mask.shape[0], 2, device=eat_mask.device)
        drop_eat = random[:, 0] < self.stream_dropout
        drop_spear = (random[:, 1] < self.stream_dropout) & ~drop_eat
        return (
            eat_mask & ~drop_eat[:, None],
            spear_mask & ~drop_spear[:, None],
        )

    def representation(
        self,
        eat: torch.Tensor,
        spear: torch.Tensor,
        eat_mask: torch.Tensor,
        spear_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Return one Music-specific hidden representation per file."""
        eat_mask, spear_mask = self._stream_masks(eat_mask, spear_mask)
        batch, views = eat.shape[:2]
        eat_tokens = (
            self.eat_projection(eat)
            + self.stat_embedding[None, None, :, :]
            + self.view_embedding[None, :views, None, :]
            + self.stream_embedding[0]
        ).reshape(batch, views * 4, self.width)
        eat_token_mask = eat_mask[:, :, None].expand(-1, -1, 4).reshape(batch, -1)

        spear_tokens = (
            self.spear_projection(spear)
            + self.spear_layer_embedding[None, None, :, None, :]
            + self.stat_embedding[None, None, None, :, :]
            + self.view_embedding[None, :views, None, None, :]
            + self.stream_embedding[1]
        ).reshape(batch, views * 13 * 4, self.width)
        spear_token_mask = spear_mask[:, :, None, None].expand(
            -1, -1, 13, 4,
        ).reshape(batch, -1)
        tokens = torch.cat((eat_tokens, spear_tokens), dim=1)
        mask = torch.cat((eat_token_mask, spear_token_mask), dim=1)
        pooled = self.music_pool(tokens, mask)[:, 0]
        return self.music_mlp(pooled)

    def forward(
        self,
        eat: torch.Tensor,
        spear: torch.Tensor,
        eat_mask: torch.Tensor,
        spear_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return primary Music logit and auxiliary ``[Voice, File]`` logits."""
        hidden = self.representation(eat, spear, eat_mask, spear_mask)
        return self.music_output(hidden).squeeze(-1), self.auxiliary_output(hidden)


def pairwise_music_rank_loss(
    music_logits: torch.Tensor, music_targets: torch.Tensor,
) -> torch.Tensor:
    """Logistic all-positive/all-negative rank loss inside one balanced batch."""
    positive = music_logits[music_targets >= .5]
    negative = music_logits[music_targets < .5]
    if not len(positive) or not len(negative):
        return music_logits.new_zeros(())
    return F.softplus(-(positive[:, None] - negative[None, :])).mean()


def asymmetric_bernoulli_consistency(
    codec_logits: torch.Tensor,
    clean_teacher_logits: torch.Tensor,
    pair_mask: torch.Tensor,
) -> torch.Tensor:
    """Teacher-to-codec Bernoulli KL on verified clean/channel pairs."""
    # Explicit fp32 is required outside autocast: bf16 rounds ``1-1e-6`` to
    # one and would make the entropy's ``0 * log(0)`` produce NaN.
    teacher_probability = (
        clean_teacher_logits.detach().float().sigmoid().clamp(1e-6, 1 - 1e-6)
    )
    cross_entropy = F.binary_cross_entropy_with_logits(
        codec_logits.float(), teacher_probability, reduction="none",
    )
    entropy = -(
        teacher_probability * teacher_probability.log()
        + (1 - teacher_probability) * (1 - teacher_probability).log()
    )
    divergence = cross_entropy - entropy
    return (divergence * pair_mask).sum() / pair_mask.sum().clamp_min(1.0)


def dedicated_music_loss(
    music_logits: torch.Tensor,
    auxiliary_logits: torch.Tensor,
    music_targets: torch.Tensor,
    voice_targets: torch.Tensor,
    file_targets: torch.Tensor,
    voice_present: torch.Tensor,
    *,
    rank_weight: float = .20,
    auxiliary_weight: float = .05,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Music BCE+rank with small Voice/File stabilization auxiliaries."""
    music_bce = F.binary_cross_entropy_with_logits(music_logits, music_targets)
    rank = pairwise_music_rank_loss(music_logits, music_targets)
    voice_raw = F.binary_cross_entropy_with_logits(
        auxiliary_logits[:, 0], voice_targets, reduction="none",
    )
    voice_bce = (
        (voice_raw * voice_present).sum() / voice_present.sum().clamp_min(1.0)
    )
    file_bce = F.binary_cross_entropy_with_logits(
        auxiliary_logits[:, 1], file_targets,
    )
    auxiliary = .5 * (voice_bce + file_bce)
    total = music_bce + rank_weight * rank + auxiliary_weight * auxiliary
    return total, {
        "music_bce": music_bce,
        "rank": rank,
        "auxiliary": auxiliary,
    }
