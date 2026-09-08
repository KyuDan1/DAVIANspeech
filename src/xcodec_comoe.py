"""Compact lower/upper residual-codebook experts for AI-music detection."""

from __future__ import annotations

import torch
from torch import nn


class TokenBranch(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, dropout: float) -> None:
        super().__init__()
        self.projection = nn.Linear(2 * width, width)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=width, nhead=heads, dim_feedforward=4 * width,
            dropout=dropout, activation="gelu", batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=layers, norm=nn.LayerNorm(width)
        )
        self.score = nn.Linear(width, 1)

    def forward(self, left: torch.Tensor, right: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.projection(torch.cat((left, right), dim=-1))
        hidden = self.encoder(hidden)
        pooled = hidden.mean(dim=1)
        return pooled, self.score(pooled).squeeze(-1)


class XCodecCoMoE(nn.Module):
    """Two complementary token experts plus a learned fixed fusion head."""

    def __init__(
        self, vocab: int = 1024, width: int = 128, layers: int = 3,
        heads: int = 4, dropout: float = 0.15,
    ) -> None:
        super().__init__()
        self.embeddings = nn.ModuleList([
            nn.Embedding(vocab, width) for _ in range(4)
        ])
        self.lower = TokenBranch(width, layers, heads, dropout)
        self.upper = TokenBranch(width, layers, heads, dropout)
        self.fusion = nn.Sequential(
            nn.LayerNorm(2 * width), nn.Dropout(dropout), nn.Linear(2 * width, 1)
        )

    def forward(self, tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        embedded = [layer(tokens[:, index]) for index, layer in enumerate(self.embeddings)]
        lower, lower_logit = self.lower(embedded[0], embedded[1])
        upper, upper_logit = self.upper(embedded[2], embedded[3])
        fused = self.fusion(torch.cat((lower, upper), dim=-1)).squeeze(-1)
        return fused, lower_logit, upper_logit
