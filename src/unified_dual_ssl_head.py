"""Unified EAT/SPEAR token head for separation-free component detection.

The two frozen encoders expose different temporal resolutions.  EAT keeps
layer/statistic tokens for broad acoustic evidence, while SPEAR keeps compact
time-bin tokens for vocal and local evidence.  They are concatenated as tokens
after task-specific layer fusion; no frame-to-frame alignment or source
separation is required.
"""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


class UnifiedDualSSLHead(nn.Module):
    """Joint Voice/Music/File head over compact EAT and SPEAR tokens."""

    TASKS = 3  # Voice fake, Music fake, File fake.

    def __init__(
        self,
        eat_layers: int = 12,
        eat_stats: int = 5,
        eat_dimension: int = 128,
        spear_layers: int = 4,
        spear_stats: int = 4,
        spear_dimension: int = 64,
        width: int = 96,
        heads: int = 4,
        context_layers: int = 1,
        maximum_views: int = 3,
        maximum_bins: int = 8,
        dropout: float = 0.15,
        file_component_weight: float = 0.10,
    ) -> None:
        super().__init__()
        if width % heads:
            raise ValueError("width must be divisible by heads")
        if not 0 <= file_component_weight <= 1:
            raise ValueError("file_component_weight must lie in [0, 1]")
        self.width = int(width)
        self.maximum_views = int(maximum_views)
        self.maximum_bins = int(maximum_bins)
        self.eat_layers = int(eat_layers)
        self.spear_layers = int(spear_layers)
        self.file_component_weight = float(file_component_weight)

        self.eat_norm = nn.LayerNorm(eat_dimension)
        self.spear_norm = nn.LayerNorm(spear_dimension)
        self.eat_projection = nn.Linear(eat_dimension, width)
        self.spear_projection = nn.Linear(spear_dimension, width)

        # One authenticity task can prefer different SSL depths from another.
        self.eat_layer_logits = nn.Parameter(torch.zeros(self.TASKS, eat_layers))
        self.spear_layer_logits = nn.Parameter(torch.zeros(self.TASKS, spear_layers))
        self.task_embedding = nn.Parameter(torch.empty(self.TASKS, width))
        self.stream_embedding = nn.Parameter(torch.empty(2, width))
        self.view_embedding = nn.Parameter(torch.empty(maximum_views, width))
        self.bin_embedding = nn.Parameter(torch.empty(maximum_bins, width))
        self.eat_stat_embedding = nn.Parameter(torch.empty(eat_stats, width))
        self.spear_stat_embedding = nn.Parameter(torch.empty(spear_stats, width))
        for parameter in (
            self.task_embedding,
            self.stream_embedding,
            self.view_embedding,
            self.bin_embedding,
            self.eat_stat_embedding,
            self.spear_stat_embedding,
        ):
            nn.init.normal_(parameter, std=0.02)

        layer = nn.TransformerEncoderLayer(
            d_model=width,
            nhead=heads,
            dim_feedforward=3 * width,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.context = nn.TransformerEncoder(
            layer, num_layers=context_layers, norm=nn.LayerNorm(width)
        )
        self.attention_query = nn.Parameter(torch.empty(self.TASKS, width))
        nn.init.normal_(self.attention_query, std=0.02)
        self.classifier = nn.Sequential(
            nn.LayerNorm(3 * width),
            nn.Linear(3 * width, width),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(width, 1),
        )
        self.presence_head = nn.Sequential(
            nn.LayerNorm(3 * width), nn.Linear(3 * width, 1)
        )
        self.joint_head = nn.Sequential(
            nn.LayerNorm(3 * width), nn.Linear(3 * width, width), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(width, 4),
        )

    def _tokens(
        self,
        eat: torch.Tensor,
        spear: torch.Tensor,
        eat_mask: torch.Tensor,
        spear_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if eat.ndim != 5:
            raise ValueError("EAT must have shape [batch, view, layer, stat, dim]")
        if spear.ndim != 6:
            raise ValueError(
                "SPEAR must have shape [batch, view, bin, layer, stat, dim]"
            )
        batch, views, eat_layers, eat_stats, _ = eat.shape
        sbatch, sviews, bins, spear_layers, spear_stats, _ = spear.shape
        if (batch, views) != (sbatch, sviews):
            raise ValueError("EAT and SPEAR batch/view dimensions differ")
        if eat_mask.shape != (batch, views):
            raise ValueError("EAT mask has incompatible shape")
        if spear_mask.shape != (batch, views, bins):
            raise ValueError("SPEAR mask has incompatible shape")
        if eat_layers != self.eat_layers or spear_layers != self.spear_layers:
            raise ValueError("cached layer count differs from model configuration")
        if views > self.maximum_views or bins > self.maximum_bins:
            raise ValueError("cached view/bin count exceeds positional embeddings")

        eat_hidden = self.eat_projection(self.eat_norm(eat.float()))
        eat_layer_weight = self.eat_layer_logits.softmax(dim=-1)
        # [B,Q,V,S,W]
        eat_hidden = torch.einsum(
            "bvlsd,ql->bqvsd", eat_hidden, eat_layer_weight
        )
        eat_hidden = (
            eat_hidden
            + self.task_embedding[None, :, None, None]
            + self.stream_embedding[0]
            + self.view_embedding[None, None, :views, None]
            + self.eat_stat_embedding[None, None, None, :eat_stats]
        )
        eat_hidden = eat_hidden.reshape(batch, self.TASKS, views * eat_stats, -1)
        eat_token_mask = eat_mask[:, :, None].expand(
            -1, -1, eat_stats
        ).reshape(batch, views * eat_stats)

        spear_hidden = self.spear_projection(self.spear_norm(spear.float()))
        spear_layer_weight = self.spear_layer_logits.softmax(dim=-1)
        # [B,Q,V,N,S,W]
        spear_hidden = torch.einsum(
            "bvmlsd,ql->bqvmsd", spear_hidden, spear_layer_weight
        )
        spear_hidden = (
            spear_hidden
            + self.task_embedding[None, :, None, None, None]
            + self.stream_embedding[1]
            + self.view_embedding[None, None, :views, None, None]
            + self.bin_embedding[None, None, None, :bins, None]
            + self.spear_stat_embedding[None, None, None, None, :spear_stats]
        )
        spear_hidden = spear_hidden.reshape(
            batch, self.TASKS, views * bins * spear_stats, -1
        )
        spear_token_mask = spear_mask[:, :, :, None].expand(
            -1, -1, -1, spear_stats
        ).reshape(batch, views * bins * spear_stats)

        tokens = torch.cat((eat_hidden, spear_hidden), dim=2)
        mask = torch.cat((eat_token_mask, spear_token_mask), dim=1)
        return tokens, mask

    def forward(
        self,
        eat: torch.Tensor,
        spear: torch.Tensor,
        eat_mask: torch.Tensor,
        spear_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        tokens, mask = self._tokens(eat, spear, eat_mask, spear_mask)
        batch, tasks, count, width = tokens.shape
        expanded_mask = mask[:, None].expand(-1, tasks, -1).reshape(batch * tasks, count)
        hidden = self.context(
            tokens.reshape(batch * tasks, count, width),
            src_key_padding_mask=~expanded_mask,
        ).reshape(batch, tasks, count, width)

        scores = torch.einsum(
            "bqtd,qd->bqt", hidden, self.attention_query
        ) / math.sqrt(width)
        scores = scores.masked_fill(~mask[:, None], -1e4)
        attention = scores.softmax(dim=-1)
        mean = torch.einsum("bqt,bqtd->bqd", attention, hidden)
        second = torch.einsum("bqt,bqtd->bqd", attention, hidden.square())
        std = (second - mean.square()).clamp_min(1e-5).sqrt()
        maximum = hidden.masked_fill(~mask[:, None, :, None], -1e4).max(dim=2).values
        pooled = torch.cat((mean, std, maximum), dim=-1)
        task_logits = self.classifier(pooled).squeeze(-1)
        presence_logits = self.presence_head(pooled[:, :2]).squeeze(-1)
        joint_logits = self.joint_head(pooled[:, 2])
        return task_logits, presence_logits, joint_logits

    def probabilities(self, task_logits: torch.Tensor) -> torch.Tensor:
        direct = task_logits.sigmoid()
        component_or = 1 - (1 - direct[:, 0]) * (1 - direct[:, 1])
        file_probability = (
            (1 - self.file_component_weight) * direct[:, 2]
            + self.file_component_weight * component_or
        )
        return torch.stack((direct[:, 0], direct[:, 1], file_probability), dim=-1)


def unified_multitask_loss(
    model: UnifiedDualSSLHead,
    task_logits: torch.Tensor,
    presence_logits: torch.Tensor,
    joint_logits: torch.Tensor,
    fake_targets: torch.Tensor,
    presence_targets: torch.Tensor,
    sample_weight: torch.Tensor,
    presence_weight: float = 0.15,
    joint_weight: float = 0.20,
    consistency_weight: float = 0.05,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute component-conditional balanced loss and auxiliary objectives."""

    def weighted_bce(logits, targets, weights):
        raw = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        return (raw * weights).sum() / weights.sum().clamp_min(1e-8)

    voice_selected = presence_targets[:, 0].bool()
    music_selected = presence_targets[:, 1].bool()
    voice = weighted_bce(
        task_logits[voice_selected, 0], fake_targets[voice_selected, 0],
        sample_weight[voice_selected],
    )
    music = weighted_bce(
        task_logits[music_selected, 1], fake_targets[music_selected, 1],
        sample_weight[music_selected],
    )
    file_loss = weighted_bce(
        task_logits[:, 2], fake_targets[:, 2], sample_weight
    )
    presence = F.binary_cross_entropy_with_logits(
        presence_logits, presence_targets
    )
    joint_target = (
        fake_targets[:, 0].long() * 2 + fake_targets[:, 1].long()
    )
    joint = F.cross_entropy(joint_logits, joint_target)
    probability = model.probabilities(task_logits)
    component_or = 1 - (1 - probability[:, 0]) * (1 - probability[:, 1])
    consistency = F.smooth_l1_loss(probability[:, 2], component_or)
    component = 0.5 * (voice + music)
    total = (
        component + 0.5 * file_loss + presence_weight * presence
        + joint_weight * joint + consistency_weight * consistency
    )
    return total, {
        "component": component.detach(), "voice": voice.detach(),
        "music": music.detach(), "file": file_loss.detach(),
        "presence": presence.detach(), "joint": joint.detach(),
        "consistency": consistency.detach(),
    }
