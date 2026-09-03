"""Temporal multiple-instance head for partial and mixed audio deepfakes."""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

try:
    from .dual_domain_head import MultiTaskAttentiveStats, SwiGLUProjection
except ImportError:  # pragma: no cover - offline flat module import
    from dual_domain_head import MultiTaskAttentiveStats, SwiGLUProjection


class TemporalDualDomainHead(nn.Module):
    """Fuse global context with stream-specific local authenticity evidence."""

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
        self.task_embedding = nn.Parameter(torch.empty(self.TASKS, width))
        for parameter in (
            self.stat_embedding, self.spear_layer_embedding,
            self.view_embedding, self.stream_embedding, self.task_embedding,
        ):
            nn.init.normal_(parameter, std=0.02)

        # A small sample-adaptive residual is added to a stable task-specific
        # prior. This follows the useful part of SSL layer-gating MoE without
        # introducing large experts that overfit the synthetic generators.
        self.spear_layer_gate = nn.Parameter(torch.zeros(self.TASKS, 13))
        self.spear_gate_context = nn.Linear(width, self.TASKS, bias=False)
        self.eat_view_mlp = nn.Sequential(
            nn.LayerNorm(width), nn.Linear(width, width), nn.GELU(), nn.Dropout(dropout)
        )
        self.spear_view_mlp = nn.Sequential(
            nn.LayerNorm(width), nn.Linear(width, width), nn.GELU(), nn.Dropout(dropout)
        )
        self.eat_view_weight = nn.Parameter(torch.empty(self.TASKS, width))
        self.spear_view_weight = nn.Parameter(torch.empty(self.TASKS, width))
        self.eat_view_bias = nn.Parameter(torch.zeros(self.TASKS))
        self.spear_view_bias = nn.Parameter(torch.zeros(self.TASKS))
        nn.init.normal_(self.eat_view_weight, std=0.02)
        nn.init.normal_(self.spear_view_weight, std=0.02)

        self.global_pool = MultiTaskAttentiveStats(width, self.TASKS, heads)
        self.global_mlp = nn.Sequential(
            nn.LayerNorm(width * 2), nn.Linear(width * 2, width),
            nn.GELU(), nn.Dropout(dropout),
        )
        self.global_weight = nn.Parameter(torch.empty(self.TASKS, width))
        self.global_bias = nn.Parameter(torch.zeros(self.TASKS))
        nn.init.normal_(self.global_weight, std=0.02)
        self.joint_head = nn.Sequential(
            nn.LayerNorm(width * 2), nn.Linear(width * 2, width),
            nn.GELU(), nn.Dropout(dropout), nn.Linear(width, 4),
        )
        self.mil_temperature_raw = nn.Parameter(torch.full((self.TASKS,), 3.98))
        self.mil_mix_logit = nn.Parameter(torch.full((self.TASKS,), -0.62))

    def _stream_masks(
        self, eat_mask: torch.Tensor, spear_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not self.training or self.stream_dropout <= 0:
            return eat_mask, spear_mask
        random = torch.rand(eat_mask.shape[0], 2, device=eat_mask.device)
        drop_eat = random[:, 0] < self.stream_dropout
        drop_spear = random[:, 1] < self.stream_dropout
        drop_spear &= ~drop_eat
        return eat_mask & ~drop_eat[:, None], spear_mask & ~drop_spear[:, None]

    @staticmethod
    def _masked_lme(
        logits: torch.Tensor, mask: torch.Tensor, temperature: torch.Tensor,
    ) -> torch.Tensor:
        """Log-mean-exp over stream/view instances, independently per task."""
        scaled = logits * temperature[None, None, :]
        scaled = scaled.masked_fill(~mask[:, :, None], -1e4)
        count = mask.sum(dim=1).clamp_min(1).to(logits.dtype)
        pooled = torch.logsumexp(scaled, dim=1) - count.log()[:, None]
        return pooled / temperature[None, :]

    def forward(
        self,
        eat: torch.Tensor,
        spear: torch.Tensor,
        eat_mask: torch.Tensor,
        spear_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        eat_mask, spear_mask = self._stream_masks(eat_mask, spear_mask)
        batch, views = eat.shape[:2]

        eat_tokens = self.eat_projection(eat)
        eat_tokens = (
            eat_tokens + self.stat_embedding[None, None, :, :]
            + self.view_embedding[None, :views, None, :]
            + self.stream_embedding[0]
        )
        eat_view = eat_tokens.mean(dim=2)
        eat_view = self.eat_view_mlp(
            eat_view[:, None] + self.task_embedding[None, :, None]
        )
        eat_view_logits = (
            eat_view * self.eat_view_weight[None, :, None]
        ).sum(dim=-1) + self.eat_view_bias[None, :, None]
        eat_view_logits = eat_view_logits.transpose(1, 2)

        spear_tokens = self.spear_projection(spear)
        spear_tokens = (
            spear_tokens + self.spear_layer_embedding[None, None, :, None, :]
            + self.stat_embedding[None, None, None, :, :]
            + self.view_embedding[None, :views, None, None, :]
            + self.stream_embedding[1]
        )
        spear_layers = spear_tokens.mean(dim=3)
        gate_context = self.spear_gate_context(spear_layers)
        valid = spear_mask[:, :, None, None].to(gate_context.dtype)
        gate_context = (gate_context * valid).sum(dim=1) / valid.sum(dim=1).clamp_min(1)
        gate_logits = self.spear_layer_gate[None] + gate_context.transpose(1, 2)
        layer_weights = gate_logits.softmax(dim=-1)
        spear_view = torch.einsum("bvld,bql->bqvd", spear_layers, layer_weights)
        spear_view = self.spear_view_mlp(
            spear_view + self.task_embedding[None, :, None]
        )
        spear_view_logits = (
            spear_view * self.spear_view_weight[None, :, None]
        ).sum(dim=-1) + self.spear_view_bias[None, :, None]
        spear_view_logits = spear_view_logits.transpose(1, 2)

        flat_eat = eat_tokens.reshape(batch, views * 4, self.width)
        flat_eat_mask = eat_mask[:, :, None].expand(-1, -1, 4).reshape(batch, -1)
        flat_spear = spear_tokens.reshape(batch, views * 13 * 4, self.width)
        flat_spear_mask = spear_mask[:, :, None, None].expand(
            -1, -1, 13, 4
        ).reshape(batch, -1)
        tokens = torch.cat((flat_eat, flat_spear), dim=1)
        token_mask = torch.cat((flat_eat_mask, flat_spear_mask), dim=1)
        pooled = self.global_pool(tokens, token_mask)
        hidden = self.global_mlp(pooled)
        global_logits = (
            hidden * self.global_weight[None]
        ).sum(dim=-1) + self.global_bias

        instance_logits = torch.cat((eat_view_logits, spear_view_logits), dim=1)
        instance_mask = torch.cat((eat_mask, spear_mask), dim=1)
        temperature = F.softplus(self.mil_temperature_raw) + 1.0
        mil_logits = self._masked_lme(instance_logits, instance_mask, temperature)
        mixture = self.mil_mix_logit.sigmoid()[None]
        task_logits = (1 - mixture) * global_logits + mixture * mil_logits
        joint_logits = self.joint_head(pooled[:, 2])
        return task_logits, joint_logits, eat_view_logits, spear_view_logits

    @staticmethod
    def probabilities(task_logits: torch.Tensor, joint_logits: torch.Tensor) -> torch.Tensor:
        direct = task_logits.sigmoid()
        joint = joint_logits.softmax(dim=-1)
        marginals = torch.stack(
            (joint[:, 2] + joint[:, 3], joint[:, 1] + joint[:, 3], 1 - joint[:, 0]),
            dim=-1,
        )
        return 0.5 * direct + 0.5 * marginals


def temporal_multitask_loss(
    task_logits: torch.Tensor,
    joint_logits: torch.Tensor,
    eat_view_logits: torch.Tensor,
    spear_view_logits: torch.Tensor,
    component_targets: torch.Tensor,
    joint_targets: torch.Tensor,
    eat_view_targets: torch.Tensor,
    spear_view_targets: torch.Tensor,
    eat_view_mask: torch.Tensor,
    spear_view_mask: torch.Tensor,
    auxiliary_weight: float = 0.5,
) -> torch.Tensor:
    direct = F.binary_cross_entropy_with_logits(task_logits, component_targets)
    joint = F.cross_entropy(joint_logits, joint_targets)
    probabilities = TemporalDualDomainHead.probabilities(task_logits, joint_logits)
    consistent_file = 1 - (1 - probabilities[:, 0]) * (1 - probabilities[:, 1])
    consistency = F.mse_loss(probabilities[:, 2], consistent_file)

    auxiliary = task_logits.new_zeros(())
    terms = 0
    for logits, targets, mask in (
        (eat_view_logits, eat_view_targets, eat_view_mask),
        (spear_view_logits, spear_view_targets, spear_view_mask),
    ):
        selected = mask[:, :, None].expand_as(logits)
        if selected.any():
            auxiliary = auxiliary + F.binary_cross_entropy_with_logits(
                logits[selected], targets[selected]
            )
            terms += 1
    if terms:
        auxiliary = auxiliary / terms
    return direct + 0.5 * joint + 0.1 * consistency + auxiliary_weight * auxiliary
