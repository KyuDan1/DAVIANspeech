"""Bounded attention gate over complementary deepfake detector experts."""

from __future__ import annotations

import torch
from torch import nn


def probability_logit(values: torch.Tensor) -> torch.Tensor:
    values = values.float().clamp(1e-5, 1 - 1e-5)
    return values.log() - (-values).log1p()


class BoundedAttentionRouter(nn.Module):
    """Softly reweight expert predictions from their internal embeddings.

    The learned gate cannot fully suppress an expert.  ``strength`` controls
    how far it may move away from uniform voting, which limits damage from an
    unseen generator or channel on which the router itself is miscalibrated.
    """

    def __init__(
        self,
        experts: int,
        tasks: int,
        expert_width: int,
        model_width: int = 32,
        heads: int = 4,
        layers: int = 1,
        dropout: float = 0.20,
        strength: float = 0.50,
        temperature: float = 1.50,
    ) -> None:
        super().__init__()
        if experts < 2 or tasks < 1:
            raise ValueError("router needs at least two experts and one task")
        if not 0 <= strength <= 1 or temperature <= 0:
            raise ValueError("invalid router strength or temperature")
        if model_width % heads:
            raise ValueError("model_width must be divisible by heads")
        self.experts = int(experts)
        self.tasks = int(tasks)
        self.strength = float(strength)
        self.temperature = float(temperature)
        self.projections = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(expert_width + 1),
                nn.Linear(expert_width + 1, model_width),
                nn.GELU(),
            )
            for _ in range(experts)
        ])
        self.expert_embedding = nn.Parameter(
            torch.empty(experts, model_width)
        )
        self.task_embedding = nn.Parameter(torch.empty(tasks, model_width))
        layer = nn.TransformerEncoderLayer(
            d_model=model_width,
            nhead=heads,
            dim_feedforward=2 * model_width,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=layers)
        self.score = nn.Sequential(
            nn.LayerNorm(model_width), nn.Linear(model_width, 1)
        )
        nn.init.normal_(self.expert_embedding, std=0.02)
        nn.init.normal_(self.task_embedding, std=0.02)

    def forward(
        self, hidden: torch.Tensor, probabilities: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return fused logits and weights.

        ``hidden`` is ``[B,E,T,H]`` and ``probabilities`` is ``[B,E,T]``.
        """
        if hidden.ndim != 4 or probabilities.shape != hidden.shape[:3]:
            raise ValueError("router expects hidden [B,E,T,H] and probabilities [B,E,T]")
        batch, experts, tasks, _ = hidden.shape
        if experts != self.experts or tasks != self.tasks:
            raise ValueError("expert/task count differs from router configuration")
        logits = probability_logit(probabilities)
        tokens = []
        for index, projection in enumerate(self.projections):
            item = torch.cat(
                (hidden[:, index], logits[:, index, :, None]), dim=-1
            )
            tokens.append(projection(item))
        # [B,T,E,D]: each task attends only across its expert evidence.
        tokens = torch.stack(tokens, dim=2)
        tokens = (
            tokens
            + self.expert_embedding[None, None, :, :]
            + self.task_embedding[None, :, None, :]
        )
        encoded = self.encoder(tokens.reshape(batch * tasks, experts, -1))
        raw = self.score(encoded).squeeze(-1) / self.temperature
        raw = raw.softmax(dim=-1).reshape(batch, tasks, experts)
        uniform = 1.0 / experts
        weights = uniform + self.strength * (raw - uniform)
        fused_probability = (
            weights * probabilities.float().permute(0, 2, 1)
        ).sum(dim=-1)
        return probability_logit(fused_probability), weights
