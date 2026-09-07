"""Conservative latent router for genuinely complementary audio experts."""

from __future__ import annotations

import torch
from torch import nn


class BoundedLatentExpertRouter(nn.Module):
    """Route per-task expert logits using an intermediate audio representation.

    The learned gate is a residual around a validated fixed MoE prior.  Even a
    badly calibrated router therefore cannot remove an expert: every weight is
    lower-bounded by ``(1 - strength) * prior``.  This is deliberately safer
    than a hard switch under unseen generators and channel transformations.
    """

    def __init__(
        self,
        latent_width: int,
        prior: torch.Tensor,
        model_width: int = 48,
        heads: int = 4,
        layers: int = 1,
        dropout: float = .15,
        strength: float = .25,
        temperature: float = 1.5,
        expert_latent_widths: tuple[int, ...] | None = None,
        global_latent_width: int = 0,
    ) -> None:
        super().__init__()
        prior = torch.as_tensor(prior, dtype=torch.float32)
        if prior.ndim != 2 or prior.shape[0] < 1 or prior.shape[1] < 2:
            raise ValueError("prior must have shape [tasks, experts>=2]")
        if (prior <= 0).any():
            raise ValueError("all prior expert weights must be positive")
        if not torch.allclose(prior.sum(-1), torch.ones(prior.shape[0]), atol=1e-6):
            raise ValueError("each task prior must sum to one")
        if not 0 <= strength <= 1 or temperature <= 0:
            raise ValueError("invalid router strength or temperature")
        if model_width % heads:
            raise ValueError("model width must be divisible by attention heads")
        self.tasks, self.experts = prior.shape
        self.strength = float(strength)
        self.temperature = float(temperature)
        self.register_buffer("prior", prior)
        self.expert_latent_widths = (
            tuple(int(value) for value in expert_latent_widths)
            if expert_latent_widths is not None else None
        )
        self.global_latent_width = int(global_latent_width)
        if self.expert_latent_widths is not None:
            if (
                len(self.expert_latent_widths) != self.experts
                or any(value <= 0 for value in self.expert_latent_widths)
                or self.global_latent_width < 0
                or sum(self.expert_latent_widths) + self.global_latent_width
                != latent_width
            ):
                raise ValueError("expert/global latent widths do not match input")
            self.expert_latent_projection = nn.ModuleList([
                nn.Sequential(
                    nn.LayerNorm(width), nn.Linear(width, model_width),
                    nn.GELU(), nn.Dropout(dropout),
                )
                for width in self.expert_latent_widths
            ])
            self.global_latent_projection = (
                nn.Sequential(
                    nn.LayerNorm(self.global_latent_width),
                    nn.Linear(self.global_latent_width, model_width),
                    nn.GELU(), nn.Dropout(dropout),
                )
                if self.global_latent_width else None
            )
            self.latent_projection = None
        else:
            if self.global_latent_width:
                raise ValueError(
                    "global_latent_width requires expert_latent_widths"
                )
            self.latent_projection = nn.Sequential(
                nn.LayerNorm(latent_width),
                nn.Linear(latent_width, model_width),
                nn.GELU(),
                nn.Dropout(dropout),
            )
            self.expert_latent_projection = None
            self.global_latent_projection = None
        # Current logit, confidence, deviation from the fixed-MoE logit, and
        # each expert's cross-task mean/std form observable evidence only.
        self.evidence_projection = nn.Sequential(
            nn.Linear(5, model_width), nn.GELU(),
        )
        self.task_embedding = nn.Parameter(torch.empty(self.tasks, model_width))
        self.expert_embedding = nn.Parameter(torch.empty(self.experts, model_width))
        layer = nn.TransformerEncoderLayer(
            d_model=model_width, nhead=heads,
            dim_feedforward=2 * model_width, dropout=dropout,
            activation="gelu", batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=layers)
        self.score = nn.Sequential(nn.LayerNorm(model_width), nn.Linear(model_width, 1))
        nn.init.normal_(self.task_embedding, std=.02)
        nn.init.normal_(self.expert_embedding, std=.02)
        # Starting exactly at the fixed MoE makes the comparison controlled.
        nn.init.zeros_(self.score[-1].weight)
        nn.init.zeros_(self.score[-1].bias)

    def forward(
        self, latent: torch.Tensor, expert_logits: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return fused logits and weights.

        ``latent`` is ``[B,H]`` and ``expert_logits`` is ``[B,E,T]``.
        """
        if latent.ndim != 2:
            raise ValueError("latent must have shape [batch, width]")
        if expert_logits.shape != (len(latent), self.experts, self.tasks):
            raise ValueError("expert logits do not match router dimensions")
        values = expert_logits.float().permute(0, 2, 1)  # [B,T,E]
        fixed = (values * self.prior[None]).sum(-1, keepdim=True)
        cross_mean = expert_logits.float().mean(-1)[:, None, :].expand(-1, self.tasks, -1)
        cross_std = expert_logits.float().std(-1, unbiased=False)[:, None, :].expand(
            -1, self.tasks, -1
        )
        evidence = torch.stack((
            values, values.abs(), values - fixed, cross_mean, cross_std,
        ), dim=-1)
        if self.expert_latent_widths is None:
            latent_tokens = self.latent_projection(latent)[:, None, :].expand(
                -1, self.experts, -1
            )
            global_token = 0
        else:
            blocks = torch.split(
                latent,
                (*self.expert_latent_widths, self.global_latent_width)
                if self.global_latent_width else self.expert_latent_widths,
                dim=-1,
            )
            latent_tokens = torch.stack([
                projection(blocks[index])
                for index, projection in enumerate(
                    self.expert_latent_projection
                )
            ], dim=1)
            global_token = (
                self.global_latent_projection(blocks[-1])[:, None, :]
                if self.global_latent_projection is not None else 0
            )
        base_tokens = latent_tokens[:, None, :, :]
        if torch.is_tensor(global_token):
            base_tokens = base_tokens + global_token[:, None, :, :]
        tokens = (
            base_tokens
            + self.evidence_projection(evidence)
            + self.task_embedding[None, :, None, :]
            + self.expert_embedding[None, None, :, :]
        )
        encoded = self.encoder(tokens.reshape(-1, self.experts, tokens.shape[-1]))
        adjustment = self.score(encoded).squeeze(-1).reshape(
            len(latent), self.tasks, self.experts
        )
        adaptive = torch.softmax(
            self.prior.log()[None] + adjustment / self.temperature, dim=-1
        )
        weights = (1 - self.strength) * self.prior[None] + self.strength * adaptive
        return (weights * values).sum(-1), weights


def pairwise_ranking_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    masks: torch.Tensor,
) -> torch.Tensor:
    """Task-conditional logistic AUC surrogate aligned with EER ranking."""
    if logits.shape != targets.shape or masks.shape != targets.shape:
        raise ValueError("logits, targets, and masks must share [batch, tasks]")
    terms = []
    for task in range(logits.shape[1]):
        valid = masks[:, task].bool()
        positive = logits[valid & targets[:, task].eq(1), task]
        negative = logits[valid & targets[:, task].eq(0), task]
        if len(positive) and len(negative):
            terms.append(torch.nn.functional.softplus(
                -(positive[:, None] - negative[None, :])
            ).mean())
    return torch.stack(terms).mean() if terms else logits.sum() * 0
