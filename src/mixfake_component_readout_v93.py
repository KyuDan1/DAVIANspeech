"""Component-conditioned readout for frozen original-mixture WPT features.

The model is intentionally a small residual on the already trained WPT task
logits.  It learns component-specific temporal attention without changing the
XLS-R/AASIST representation, so it can reuse the WPT pass that is already in
the current submission and cannot erase the pretrained forensic signal.
"""
from __future__ import annotations

import math

import torch
from torch import nn


def lme(logits: torch.Tensor, temperature: float = 5.0) -> torch.Tensor:
    if logits.ndim != 3 or logits.shape[1] <= 0 or temperature <= 0:
        raise ValueError("expected [batch, views, tasks] and positive temperature")
    return (
        torch.logsumexp(temperature * logits.float(), dim=1) / temperature
        - math.log(logits.shape[1]) / temperature
    )


class ComponentReadoutV93(nn.Module):
    TASKS = ("VOICE", "MUSIC", "FILE")

    def __init__(
        self, embedding_dim: int = 160, width: int = 128,
        dropout: float = 0.15, residual_limit: float = 4.0,
        temperature: float = 5.0,
    ) -> None:
        super().__init__()
        if min(embedding_dim, width) <= 0 or not 0 <= dropout < 1:
            raise ValueError("invalid model dimensions")
        if residual_limit <= 0 or temperature <= 0:
            raise ValueError("positive residual limit and temperature required")
        self.embedding_dim = int(embedding_dim)
        self.width = int(width)
        self.residual_limit = float(residual_limit)
        self.temperature = float(temperature)
        self.register_buffer("feature_mean", torch.zeros(embedding_dim))
        self.register_buffer("feature_std", torch.ones(embedding_dim))
        self.trunk = nn.Sequential(
            nn.Linear(embedding_dim, width),
            nn.LayerNorm(width),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(width, width),
            nn.GELU(),
        )
        self.attention = nn.Linear(width, len(self.TASKS))
        # Per-task input: attended view, mean, standard deviation, maximum,
        # all three base bag logits, and own base-view mean/std/max.
        context = 4 * width + 6
        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(context),
                nn.Linear(context, width),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(width, 1),
            ) for _ in self.TASKS
        ])
        # Exact base-WPT initialization avoids an untrained regression.
        for head in self.heads:
            nn.init.zeros_(head[-1].weight)
            nn.init.zeros_(head[-1].bias)

    def set_normalization(self, mean: torch.Tensor, std: torch.Tensor) -> None:
        if mean.shape != (self.embedding_dim,) or std.shape != mean.shape:
            raise ValueError("normalization shape mismatch")
        if not torch.isfinite(mean).all() or not torch.isfinite(std).all():
            raise ValueError("normalization must be finite")
        self.feature_mean.copy_(mean)
        self.feature_std.copy_(std.clamp_min(1e-5))

    def forward(
        self, embeddings: torch.Tensor, base_view_logits: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if embeddings.ndim != 3 or embeddings.shape[-1] != self.embedding_dim:
            raise ValueError("embeddings must be [batch, views, embedding_dim]")
        if base_view_logits.shape != (*embeddings.shape[:2], len(self.TASKS)):
            raise ValueError("base logits must be [batch, views, 3]")
        values = (embeddings.float() - self.feature_mean) / self.feature_std
        hidden = self.trunk(values)
        attention = self.attention(hidden).transpose(1, 2).softmax(dim=-1)
        attended = torch.einsum("btv,bvw->btw", attention, hidden)
        mean = hidden.mean(dim=1)
        std = hidden.std(dim=1, unbiased=False)
        maximum = hidden.amax(dim=1)
        base_bag = lme(base_view_logits, self.temperature)
        base_mean = base_view_logits.float().mean(dim=1)
        base_std = base_view_logits.float().std(dim=1, unbiased=False)
        base_max = base_view_logits.float().amax(dim=1)
        residuals = []
        global_hidden = torch.cat((mean, std, maximum), dim=-1)
        for task, head in enumerate(self.heads):
            scalar = torch.cat((
                base_bag,
                base_mean[:, task:task + 1],
                base_std[:, task:task + 1],
                base_max[:, task:task + 1],
            ), dim=-1)
            context = torch.cat((attended[:, task], global_hidden, scalar), dim=-1)
            residuals.append(head(context).squeeze(-1))
        residual = self.residual_limit * torch.tanh(
            torch.stack(residuals, dim=-1) / self.residual_limit
        )
        return base_bag + residual, residual
