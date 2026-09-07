"""Long-range music-structure head over uniformly sampled EAT segments."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class SegmentalEatMusicHead(nn.Module):
    """Combine segment content with a track self-similarity pathway.

    Each segment first pools the all-layer EAT statistics into one embedding.
    A content Transformer models the ordered embeddings, while a second
    Transformer sees their self-similarity rows.  The structure pathway can
    therefore use repetition and section changes without depending on an
    absolute generator timbre fingerprint.
    """

    def __init__(
        self,
        mean: torch.Tensor,
        std: torch.Tensor,
        max_views: int = 8,
        heads: int = 4,
        layer_depth: int = 1,
        segment_depth: int = 1,
        dropout: float = 0.20,
        branch_mode: str = "both",
    ) -> None:
        super().__init__()
        mean = torch.as_tensor(mean, dtype=torch.float32)
        std = torch.as_tensor(std, dtype=torch.float32)
        if mean.ndim != 3 or mean.shape != std.shape:
            raise ValueError("mean/std must have shape [layers, statistics, width]")
        if torch.any(std <= 0):
            raise ValueError("normalization standard deviations must be positive")
        if max_views <= 0 or layer_depth <= 0 or segment_depth <= 0:
            raise ValueError("view and Transformer depths must be positive")
        if branch_mode not in {"content", "structure", "both"}:
            raise ValueError("branch_mode must be content, structure, or both")
        layers, statistics, width = mean.shape
        if width % heads:
            raise ValueError("feature width must be divisible by attention heads")

        self.layers = layers
        self.statistics = statistics
        self.width = width
        self.max_views = max_views
        self.branch_mode = branch_mode
        self.register_buffer("normalization_mean", mean)
        self.register_buffer("normalization_std", std)
        self.layer_embedding = nn.Parameter(torch.zeros(layers, width))
        self.statistic_embedding = nn.Parameter(torch.zeros(statistics, width))
        self.position_embedding = nn.Parameter(torch.zeros(max_views, width))

        def encoder(depth: int) -> nn.TransformerEncoder:
            block = nn.TransformerEncoderLayer(
                d_model=width,
                nhead=heads,
                dim_feedforward=4 * width,
                dropout=dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            return nn.TransformerEncoder(block, num_layers=depth)

        self.layer_encoder = encoder(layer_depth)
        self.layer_key = nn.Linear(width, 1, bias=False)
        self.layer_value = nn.Linear(width, width)
        self.content_encoder = encoder(segment_depth)
        self.structure_projection = nn.Sequential(
            nn.LayerNorm(max_views), nn.Linear(max_views, width), nn.GELU(),
        )
        self.structure_encoder = encoder(segment_depth)
        self.content_key = nn.Linear(width, 1, bias=False)
        self.structure_key = nn.Linear(width, 1, bias=False)
        branches = 2 if branch_mode == "both" else 1
        self.classifier = nn.Sequential(
            nn.LayerNorm(branches * width),
            nn.Linear(branches * width, width),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(width, 1),
        )
        for embedding in (
            self.layer_embedding, self.statistic_embedding, self.position_embedding
        ):
            nn.init.normal_(embedding, std=0.02)

    @staticmethod
    def _pool(
        values: torch.Tensor, mask: torch.Tensor, key: nn.Linear
    ) -> torch.Tensor:
        attention = key(values).squeeze(-1).masked_fill(~mask, -1e4)
        attention = attention.softmax(dim=1)
        return torch.einsum("bt,btw->bw", attention, values)

    def forward(self, values: torch.Tensor, view_mask: torch.Tensor) -> torch.Tensor:
        if values.ndim != 5:
            raise ValueError(
                "values must have shape [batch, views, layers, statistics, width]"
            )
        batch, views, layers, statistics, width = values.shape
        if (views, layers, statistics, width) != (
            self.max_views, self.layers, self.statistics, self.width
        ):
            raise ValueError("segmental EAT tensor has incompatible shape")
        if view_mask.shape != (batch, views):
            raise ValueError("view mask has incompatible shape")
        view_mask = view_mask.bool()
        if not torch.all(view_mask.any(dim=1)):
            raise ValueError("every item needs at least one valid segment")

        values = (
            (values.float() - self.normalization_mean[None, None])
            / self.normalization_std[None, None]
        ).clamp_(-8, 8)
        values = values.reshape(batch * views, layers * statistics, width)
        identity = (
            self.layer_embedding[:, None, :]
            + self.statistic_embedding[None, :, :]
        ).reshape(1, layers * statistics, width)
        values = self.layer_encoder(values + identity)
        weights = self.layer_key(values).softmax(dim=1)
        segments = (weights * self.layer_value(values)).sum(dim=1)
        segments = segments.reshape(batch, views, width)
        segments = segments.masked_fill(~view_mask.unsqueeze(-1), 0)

        content = self.content_encoder(
            segments + self.position_embedding[None],
            src_key_padding_mask=~view_mask,
        )
        outputs = []
        if self.branch_mode in {"content", "both"}:
            outputs.append(self._pool(content, view_mask, self.content_key))

        if self.branch_mode in {"structure", "both"}:
            normalized = F.normalize(segments, dim=-1, eps=1e-6)
            similarity = normalized @ normalized.transpose(1, 2)
            pair_mask = view_mask[:, :, None] & view_mask[:, None, :]
            similarity = similarity.masked_fill(~pair_mask, 0)
            structure = self.structure_projection(similarity)
            structure = self.structure_encoder(
                structure + self.position_embedding[None],
                src_key_padding_mask=~view_mask,
            )
            outputs.append(self._pool(structure, view_mask, self.structure_key))
        return self.classifier(torch.cat(outputs, dim=-1)).squeeze(-1)
