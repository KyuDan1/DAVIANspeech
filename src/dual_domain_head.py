"""Lightweight multi-task fusion head for frozen EAT and SPEAR statistics."""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


class SwiGLUProjection(nn.Module):
    def __init__(self, input_dim: int, width: int, dropout: float) -> None:
        super().__init__()
        self.projection = nn.Linear(input_dim, width * 2)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(width)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        gate, value = self.projection(inputs).chunk(2, dim=-1)
        return self.norm(self.dropout(F.silu(gate) * value))


class MultiTaskAttentiveStats(nn.Module):
    """Task/head-specific attentive mean and standard deviation pooling."""

    def __init__(self, width: int, tasks: int, heads: int) -> None:
        super().__init__()
        if width % heads:
            raise ValueError("width must be divisible by heads")
        self.tasks = tasks
        self.heads = heads
        self.head_dim = width // heads
        self.query = nn.Parameter(torch.empty(tasks, heads, self.head_dim))
        self.bias = nn.Parameter(torch.zeros(tasks, heads, 1))
        nn.init.normal_(self.query, std=0.02)

    def forward(self, tokens: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        batch, count, _ = tokens.shape
        values = tokens.reshape(batch, count, self.heads, self.head_dim)
        scores = torch.einsum("bthd,qhd->bqht", values, self.query)
        scores = scores / math.sqrt(self.head_dim) + self.bias[None]
        scores = scores.masked_fill(~mask[:, None, None, :], -1e4)
        attention = scores.softmax(dim=-1)
        mean = torch.einsum("bqht,bthd->bqhd", attention, values)
        second = torch.einsum("bqht,bthd->bqhd", attention, values.square())
        deviation = (second - mean.square()).clamp_min(1e-5).sqrt()
        return torch.cat((mean.flatten(2), deviation.flatten(2)), dim=-1)


class DualDomainHead(nn.Module):
    """Fuse original-mixture EAT/SPEAR cues into component and joint scores."""

    TASKS = 3  # voice fake, music fake, file fake

    def __init__(
        self, width: int = 128, heads: int = 4, dropout: float = 0.25,
        stream_dropout: float = 0.10,
    ) -> None:
        super().__init__()
        self.width = width
        self.stream_dropout = stream_dropout
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
            nn.init.normal_(parameter, std=0.02)
        self.pool = MultiTaskAttentiveStats(width, self.TASKS, heads)
        self.task_mlp = nn.Sequential(
            nn.LayerNorm(width * 2),
            nn.Linear(width * 2, width),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.task_weight = nn.Parameter(torch.empty(self.TASKS, width))
        self.task_bias = nn.Parameter(torch.zeros(self.TASKS))
        nn.init.normal_(self.task_weight, std=0.02)
        self.joint_head = nn.Sequential(
            nn.LayerNorm(width * 2),
            nn.Linear(width * 2, width),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(width, 4),
        )

    def _stream_masks(
        self, eat_mask: torch.Tensor, spear_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not self.training or self.stream_dropout <= 0:
            return eat_mask, spear_mask
        batch = eat_mask.shape[0]
        random = torch.rand(batch, 2, device=eat_mask.device)
        drop_eat = random[:, 0] < self.stream_dropout
        drop_spear = random[:, 1] < self.stream_dropout
        # Never remove both sources from one example.
        drop_spear &= ~drop_eat
        return eat_mask & ~drop_eat[:, None], spear_mask & ~drop_spear[:, None]

    def forward(
        self,
        eat: torch.Tensor,
        spear: torch.Tensor,
        eat_mask: torch.Tensor,
        spear_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Accept ``[B,V,4,768]`` and ``[B,V,13,4,1280]`` statistics."""
        eat_mask, spear_mask = self._stream_masks(eat_mask, spear_mask)
        batch, views = eat.shape[:2]

        eat_tokens = self.eat_projection(eat)
        eat_tokens = (
            eat_tokens
            + self.stat_embedding[None, None, :, :]
            + self.view_embedding[None, :views, None, :]
            + self.stream_embedding[0]
        ).reshape(batch, views * 4, self.width)
        eat_token_mask = eat_mask[:, :, None].expand(-1, -1, 4).reshape(batch, -1)

        spear_tokens = self.spear_projection(spear)
        spear_tokens = (
            spear_tokens
            + self.spear_layer_embedding[None, None, :, None, :]
            + self.stat_embedding[None, None, None, :, :]
            + self.view_embedding[None, :views, None, None, :]
            + self.stream_embedding[1]
        ).reshape(batch, views * 13 * 4, self.width)
        spear_token_mask = spear_mask[:, :, None, None].expand(
            -1, -1, 13, 4
        ).reshape(batch, -1)

        tokens = torch.cat((eat_tokens, spear_tokens), dim=1)
        mask = torch.cat((eat_token_mask, spear_token_mask), dim=1)
        pooled = self.pool(tokens, mask)
        hidden = self.task_mlp(pooled)
        task_logits = (hidden * self.task_weight[None]).sum(dim=-1) + self.task_bias
        # The file-query representation also learns the complete RR/RF/FR/FF
        # posterior, retaining interactions between the component labels.
        joint_logits = self.joint_head(pooled[:, 2])
        return task_logits, joint_logits

    @staticmethod
    def probabilities(
        task_logits: torch.Tensor, joint_logits: torch.Tensor
    ) -> torch.Tensor:
        direct = task_logits.sigmoid()
        joint = joint_logits.softmax(dim=-1)
        joint_voice = joint[:, 2] + joint[:, 3]
        joint_music = joint[:, 1] + joint[:, 3]
        joint_file = 1 - joint[:, 0]
        marginals = torch.stack((joint_voice, joint_music, joint_file), dim=-1)
        return 0.5 * direct + 0.5 * marginals


def multitask_loss(
    task_logits: torch.Tensor,
    joint_logits: torch.Tensor,
    component_targets: torch.Tensor,
    joint_targets: torch.Tensor,
) -> torch.Tensor:
    """Balanced component/file BCE plus joint-class and consistency losses."""
    direct = F.binary_cross_entropy_with_logits(task_logits, component_targets)
    joint = F.cross_entropy(joint_logits, joint_targets)
    probabilities = DualDomainHead.probabilities(task_logits, joint_logits)
    consistent_file = 1 - (1 - probabilities[:, 0]) * (1 - probabilities[:, 1])
    consistency = F.mse_loss(probabilities[:, 2], consistent_file)
    return direct + 0.5 * joint + 0.1 * consistency
