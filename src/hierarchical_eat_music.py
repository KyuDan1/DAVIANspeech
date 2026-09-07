"""Music-authenticity head over all intermediate EAT layers."""

from __future__ import annotations

import torch
from torch import nn


class HierarchicalEatMusicHead(nn.Module):
    """Attend over view/layer/statistic tokens from an original mixture.

    Distribution-statistic uncertainty is applied only while training.  It
    perturbs per-example feature style while preserving relative token
    structure, discouraging generator- and channel-specific shortcuts.
    """

    def __init__(
        self,
        mean: torch.Tensor,
        std: torch.Tensor,
        max_views: int = 3,
        heads: int = 4,
        pool_heads: int = 8,
        depth: int = 1,
        dropout: float = 0.20,
        dsu_probability: float = 0.5,
        dsu_scale: float = 1.0,
    ) -> None:
        super().__init__()
        mean = torch.as_tensor(mean, dtype=torch.float32)
        std = torch.as_tensor(std, dtype=torch.float32)
        if mean.ndim != 3 or mean.shape != std.shape:
            raise ValueError("mean/std must have shape [layers, statistics, width]")
        if torch.any(std <= 0):
            raise ValueError("normalization standard deviations must be positive")
        layers, statistics, width = mean.shape
        if width % heads:
            raise ValueError("feature width must be divisible by attention heads")
        if max_views <= 0 or pool_heads <= 0 or depth <= 0:
            raise ValueError("view/head/depth counts must be positive")
        if not 0 <= dsu_probability <= 1 or dsu_scale < 0:
            raise ValueError("invalid DSU configuration")

        self.layers = layers
        self.statistics = statistics
        self.width = width
        self.max_views = max_views
        self.dsu_probability = float(dsu_probability)
        self.dsu_scale = float(dsu_scale)
        self.register_buffer("normalization_mean", mean)
        self.register_buffer("normalization_std", std)
        self.view_embedding = nn.Parameter(torch.zeros(max_views, width))
        self.layer_embedding = nn.Parameter(torch.zeros(layers, width))
        self.statistic_embedding = nn.Parameter(torch.zeros(statistics, width))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=width,
            nhead=heads,
            dim_feedforward=4 * width,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.key = nn.Linear(width, pool_heads, bias=False)
        self.value = nn.Linear(width, width)
        self.classifier = nn.Sequential(
            nn.LayerNorm(pool_heads * width),
            nn.Linear(pool_heads * width, width),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(width, 1),
        )
        nn.init.normal_(self.view_embedding, std=0.02)
        nn.init.normal_(self.layer_embedding, std=0.02)
        nn.init.normal_(self.statistic_embedding, std=0.02)

    def _domain_statistic_uncertainty(
        self, values: torch.Tensor, valid: torch.Tensor
    ) -> torch.Tensor:
        if (
            not self.training
            or self.dsu_probability <= 0
            or self.dsu_scale <= 0
            or values.shape[0] < 2
        ):
            return values
        apply = torch.rand(values.shape[0], 1, 1, device=values.device)
        apply = apply < self.dsu_probability
        weights = valid.unsqueeze(-1).to(values.dtype)
        count = weights.sum(dim=1, keepdim=True).clamp_min(1)
        location = (values * weights).sum(dim=1, keepdim=True) / count
        variance = ((values - location).square() * weights).sum(
            dim=1, keepdim=True
        ) / count
        scale = variance.clamp_min(1e-6).sqrt()
        location_uncertainty = location.squeeze(1).std(
            dim=0, unbiased=False, keepdim=True
        ).unsqueeze(1)
        scale_uncertainty = scale.squeeze(1).std(
            dim=0, unbiased=False, keepdim=True
        ).unsqueeze(1)
        sampled_location = location + self.dsu_scale * torch.randn_like(location) \
            * location_uncertainty
        sampled_scale = (
            scale + self.dsu_scale * torch.randn_like(scale) * scale_uncertainty
        ).clamp_min(1e-3)
        perturbed = (values - location) / scale * sampled_scale + sampled_location
        return torch.where(apply, perturbed, values)

    def forward(self, values: torch.Tensor, view_mask: torch.Tensor) -> torch.Tensor:
        if values.ndim != 5:
            raise ValueError(
                "values must have shape [batch, views, layers, statistics, width]"
            )
        batch, views, layers, statistics, width = values.shape
        if (views, layers, statistics, width) != (
            self.max_views, self.layers, self.statistics, self.width
        ):
            raise ValueError("hierarchical EAT tensor has incompatible shape")
        if view_mask.shape != (batch, views):
            raise ValueError("view mask has incompatible shape")
        if not torch.all(view_mask.any(dim=1)):
            raise ValueError("every item needs at least one valid view")

        values = (
            (values.float() - self.normalization_mean[None, None])
            / self.normalization_std[None, None]
        ).clamp_(-8, 8)
        valid = view_mask[:, :, None, None].expand(
            -1, -1, layers, statistics
        ).reshape(batch, -1)
        values = values.reshape(batch, -1, width)
        values = self._domain_statistic_uncertainty(values, valid)
        embeddings = (
            self.view_embedding[:, None, None, :]
            + self.layer_embedding[None, :, None, :]
            + self.statistic_embedding[None, None, :, :]
        ).reshape(1, -1, width)
        values = values + embeddings
        values = self.encoder(values, src_key_padding_mask=~valid)
        attention = self.key(values).masked_fill(~valid.unsqueeze(-1), -1e4)
        attention = attention.softmax(dim=1)
        content = self.value(values)
        pooled = torch.einsum("bth,btw->bhw", attention, content)
        return self.classifier(pooled.flatten(1)).squeeze(-1)
