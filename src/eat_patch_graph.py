"""Patch-preserving EAT graph features and a compact multitask backend.

The older EAT heads reduce every six-second view to utterance statistics.  That
is useful for presence detection but discards the local time/frequency layout
where generator traces live.  This module retains two complementary node sets
from selected frozen EAT layers: time nodes summarize frequency patches and
frequency nodes summarize time patches.  A small shared graph Transformer then
models both node types without source separation or another SSL backbone.
"""

from __future__ import annotations

import math

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

try:  # package import in tests; flat import in the offline submission
    from .dual_domain_stats import sequence_statistics
except ImportError:  # pragma: no cover
    from dual_domain_stats import sequence_statistics


DEFAULT_LAYERS = (1, 3, 5, 7, 9, 11)
AXIS_STATISTICS = 2  # mean and standard deviation across the other axis.


def gaussian_patch_projection(
    input_width: int = 768,
    output_width: int = 128,
    seed: int = 20260904,
) -> torch.Tensor:
    """Return a deterministic, label-free projection for frozen patch tokens."""
    if input_width <= 0 or output_width <= 0:
        raise ValueError("projection widths must be positive")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    return (
        torch.randn(input_width, output_width, generator=generator)
        / np.sqrt(input_width)
    )


def _axis_statistics(values: torch.Tensor, axis: int) -> torch.Tensor:
    """Return mean/std nodes with shape ``[batch, 2, nodes, width]``."""
    mean = values.mean(dim=axis)
    variance = values.float().var(dim=axis, unbiased=False).clamp_min(1e-6)
    return torch.stack((mean.float(), variance.sqrt()), dim=1)


@torch.inference_mode()
def hierarchical_patch_graph_features(
    model,
    features: torch.Tensor,
    projection: torch.Tensor,
    layers: tuple[int, ...] = DEFAULT_LAYERS,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Extract standard all-layer summaries and patch-preserving graph nodes.

    The first result is the unprojected ``[B,all_layers,5,D]`` tensor used by
    the established EAT presence/dual-domain path.  The other results are
    temporal ``[B,L,2,T,P]`` and spectral ``[B,L,2,F,P]`` graph tensors.  One
    encoder traversal therefore serves both the anchor and the new expert.
    """
    core = getattr(model, "model", model)
    required = (
        "local_encoder", "extra_tokens", "pre_norm", "pos_drop", "blocks",
    )
    missing = [name for name in required if not hasattr(core, name)]
    if missing:
        raise TypeError(f"EAT core is missing attributes: {missing}")
    if features.ndim != 4:
        raise ValueError("EAT features must have shape [batch, channels, time, mel]")
    selected = tuple(int(index) for index in layers)
    if not selected or tuple(sorted(set(selected))) != selected:
        raise ValueError("layers must be nonempty, unique, and increasing")
    if selected[0] < 0 or selected[-1] >= len(core.blocks):
        raise ValueError("selected EAT layer is out of range")

    patch_module = getattr(core.local_encoder, "proj", None)
    if patch_module is None:
        raise TypeError("EAT local encoder does not expose its patch projection")
    grid = patch_module(features)
    if grid.ndim != 4:
        raise RuntimeError("EAT patch projection returned an invalid grid")
    batch, width, time_nodes, frequency_nodes = grid.shape
    values = grid.flatten(2).transpose(1, 2)
    positional = getattr(core, "fixed_positional_encoder", None)
    if positional is not None:
        values = values + positional(values, None)[:, :values.size(1), :]
    values = torch.cat((core.extra_tokens.expand(batch, -1, -1), values), dim=1)
    values = core.pos_drop(core.pre_norm(values))

    matrix = projection.to(device=values.device, dtype=torch.float32)
    if matrix.ndim != 2 or matrix.shape[0] != width:
        raise ValueError("projection has incompatible shape")
    hierarchical, temporal, spectral = [], [], []
    wanted = set(selected)
    for index, block in enumerate(core.blocks):
        values, _ = block(values)
        patch_statistics = sequence_statistics(values[:, 1:])
        hierarchical.append(torch.cat((
            patch_statistics, values[:, :1].float()
        ), dim=-2))
        if index not in wanted:
            continue
        patches = values[:, 1:].reshape(
            batch, time_nodes, frequency_nodes, width
        ).float()
        patches = F.layer_norm(patches, (width,)) @ matrix
        temporal.append(_axis_statistics(patches, axis=2))
        spectral.append(_axis_statistics(patches, axis=1))
    return (
        torch.stack(hierarchical, dim=1),
        torch.stack(temporal, dim=1),
        torch.stack(spectral, dim=1),
    )


@torch.inference_mode()
def patch_graph_features(
    model,
    features: torch.Tensor,
    projection: torch.Tensor,
    layers: tuple[int, ...] = DEFAULT_LAYERS,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Extract selected EAT layers as temporal and spectral graph nodes."""
    _, temporal, spectral = hierarchical_patch_graph_features(
        model, features, projection, layers
    )
    return temporal, spectral


class EatPatchGraphHead(nn.Module):
    """Task-conditioned graph Transformer over EAT time/frequency nodes."""

    TASKS = 3  # Voice fake, Music fake, File fake.

    def __init__(
        self,
        layers: int = len(DEFAULT_LAYERS),
        dimension: int = 128,
        width: int = 96,
        heads: int = 4,
        depth: int = 2,
        maximum_views: int = 3,
        maximum_time_nodes: int = 64,
        maximum_frequency_nodes: int = 16,
        dropout: float = .15,
        temperature: float = 5.,
        file_component_weight: float = .10,
    ) -> None:
        super().__init__()
        if min(layers, dimension, width, heads, depth, maximum_views) <= 0:
            raise ValueError("graph dimensions must be positive")
        if width % heads:
            raise ValueError("width must be divisible by heads")
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        if not 0 <= file_component_weight <= 1:
            raise ValueError("file_component_weight must lie in [0, 1]")
        self.layers = int(layers)
        self.width = int(width)
        self.maximum_views = int(maximum_views)
        self.maximum_time_nodes = int(maximum_time_nodes)
        self.maximum_frequency_nodes = int(maximum_frequency_nodes)
        self.temperature = float(temperature)
        self.file_component_weight = float(file_component_weight)

        self.input_norm = nn.LayerNorm(dimension)
        self.input_projection = nn.Linear(dimension, width)
        self.layer_logits = nn.Parameter(torch.zeros(self.TASKS, layers))
        self.task_embedding = nn.Parameter(torch.empty(self.TASKS, width))
        self.node_type_embedding = nn.Parameter(
            torch.empty(2 * AXIS_STATISTICS, width)
        )
        self.time_position = nn.Parameter(
            torch.empty(maximum_time_nodes, width)
        )
        self.frequency_position = nn.Parameter(
            torch.empty(maximum_frequency_nodes, width)
        )
        self.view_embedding = nn.Parameter(torch.empty(maximum_views, width))
        for parameter in (
            self.task_embedding, self.node_type_embedding, self.time_position,
            self.frequency_position, self.view_embedding,
        ):
            nn.init.normal_(parameter, std=.02)

        layer = nn.TransformerEncoderLayer(
            d_model=width, nhead=heads, dim_feedforward=4 * width,
            dropout=dropout, activation="gelu", batch_first=True,
            norm_first=True,
        )
        self.graph = nn.TransformerEncoder(
            layer, num_layers=depth, norm=nn.LayerNorm(width)
        )
        self.pool_query = nn.Parameter(torch.empty(self.TASKS, width))
        nn.init.normal_(self.pool_query, std=.02)
        self.classifiers = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(3 * width), nn.Linear(3 * width, width), nn.GELU(),
                nn.Dropout(dropout), nn.Linear(width, 1),
            )
            for _ in range(self.TASKS)
        ])

    def _nodes(
        self, temporal: torch.Tensor, spectral: torch.Tensor,
    ) -> torch.Tensor:
        if temporal.ndim != 6 or spectral.ndim != 6:
            raise ValueError(
                "graph features must be [batch, view, layer, stat, node, dim]"
            )
        batch, views, layers, statistics, time_nodes, _ = temporal.shape
        shape = spectral.shape
        if shape[:4] != (batch, views, layers, statistics):
            raise ValueError("temporal/spectral graph dimensions differ")
        frequency_nodes = shape[4]
        if layers != self.layers or statistics != AXIS_STATISTICS:
            raise ValueError("cached graph layer/statistic count differs")
        if views > self.maximum_views:
            raise ValueError("cached views exceed the model configuration")
        if time_nodes > self.maximum_time_nodes:
            raise ValueError("temporal node count exceeds positional embeddings")
        if frequency_nodes > self.maximum_frequency_nodes:
            raise ValueError("spectral node count exceeds positional embeddings")

        weights = self.layer_logits.softmax(dim=-1)
        temporal = self.input_projection(self.input_norm(temporal.float()))
        spectral = self.input_projection(self.input_norm(spectral.float()))
        temporal = torch.einsum("bvlstd,ql->bvqstd", temporal, weights)
        spectral = torch.einsum("bvlstd,ql->bvqstd", spectral, weights)

        temporal = (
            temporal
            + self.task_embedding[None, None, :, None, None]
            + self.view_embedding[None, :views, None, None, None]
            + self.node_type_embedding[None, None, None, :statistics, None]
            + self.time_position[None, None, None, None, :time_nodes]
        ).reshape(batch, views, self.TASKS, statistics * time_nodes, self.width)
        spectral = (
            spectral
            + self.task_embedding[None, None, :, None, None]
            + self.view_embedding[None, :views, None, None, None]
            + self.node_type_embedding[
                None, None, None, statistics:, None
            ]
            + self.frequency_position[
                None, None, None, None, :frequency_nodes
            ]
        ).reshape(
            batch, views, self.TASKS,
            statistics * frequency_nodes, self.width,
        )
        return torch.cat((temporal, spectral), dim=3)

    def view_logits(
        self, temporal: torch.Tensor, spectral: torch.Tensor,
    ) -> torch.Tensor:
        nodes = self._nodes(temporal, spectral)
        batch, views, tasks, count, width = nodes.shape
        hidden = self.graph(nodes.reshape(batch * views * tasks, count, width))
        hidden = hidden.reshape(batch, views, tasks, count, width)
        score = torch.einsum(
            "bvqnd,qd->bvqn", hidden, self.pool_query
        ) / math.sqrt(width)
        attention = score.softmax(dim=-1)
        mean = torch.einsum("bvqn,bvqnd->bvqd", attention, hidden)
        second = torch.einsum(
            "bvqn,bvqnd->bvqd", attention, hidden.square()
        )
        std = (second - mean.square()).clamp_min(1e-5).sqrt()
        maximum = hidden.amax(dim=3)
        pooled = torch.cat((mean, std, maximum), dim=-1)
        return torch.cat([
            classifier(pooled[:, :, task]).unsqueeze(-1)
            for task, classifier in enumerate(self.classifiers)
        ], dim=-1).squeeze(-2)

    def aggregate(
        self, logits: torch.Tensor, view_mask: torch.Tensor,
    ) -> torch.Tensor:
        if logits.ndim != 3 or view_mask.shape != logits.shape[:2]:
            raise ValueError("view logits/mask have incompatible shapes")
        mask = view_mask.bool()
        if not torch.all(mask.any(dim=1)):
            raise ValueError("every item needs at least one valid view")
        scaled = (self.temperature * logits).masked_fill(
            ~mask.unsqueeze(-1), -1e4
        )
        normalizer = mask.sum(dim=1).float().log().unsqueeze(-1)
        return (torch.logsumexp(scaled, dim=1) - normalizer) / self.temperature

    def forward(
        self, temporal: torch.Tensor, spectral: torch.Tensor,
        view_mask: torch.Tensor,
    ) -> torch.Tensor:
        return self.aggregate(self.view_logits(temporal, spectral), view_mask)

    def probabilities(self, logits: torch.Tensor) -> torch.Tensor:
        direct = logits.sigmoid()
        component_or = 1 - (1 - direct[:, 0]) * (1 - direct[:, 1])
        file_probability = (
            (1 - self.file_component_weight) * direct[:, 2]
            + self.file_component_weight * component_or
        )
        return torch.stack((direct[:, 0], direct[:, 1], file_probability), -1)


def patch_graph_loss(
    model: EatPatchGraphHead,
    logits: torch.Tensor,
    targets: torch.Tensor,
    presence: torch.Tensor,
    sample_weight: torch.Tensor,
    task_weights: tuple[float, float, float] = (.20, .35, .45),
    ranking_weight: float = .15,
    ranking_tail_fraction: float = 1.,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Component-conditional BCE plus a batchwise EER ranking surrogate."""
    if logits.shape != targets.shape or logits.shape[1] != 3:
        raise ValueError("logits and targets must have shape [batch, 3]")
    if presence.shape != (len(logits), 2):
        raise ValueError("presence must have shape [batch, 2]")
    if not 0 < ranking_tail_fraction <= 1:
        raise ValueError("ranking tail fraction must lie in (0, 1]")
    masks = (
        presence[:, 0].bool(), presence[:, 1].bool(),
        torch.ones(len(logits), dtype=torch.bool, device=logits.device),
    )
    bce_terms, ranking_terms = [], []
    for task, mask in enumerate(masks):
        raw = F.binary_cross_entropy_with_logits(
            logits[mask, task], targets[mask, task], reduction="none"
        )
        weights = sample_weight[mask]
        bce_terms.append((raw * weights).sum() / weights.sum().clamp_min(1e-8))
        positive = logits[mask & targets[:, task].eq(1), task]
        negative = logits[mask & targets[:, task].eq(0), task]
        if len(positive) and len(negative):
            pair_loss = F.softplus(
                -(positive[:, None] - negative[None, :])
            ).flatten()
            keep = max(1, math.ceil(ranking_tail_fraction * len(pair_loss)))
            ranking_terms.append(pair_loss.topk(keep).values.mean())
    scale = sum(task_weights)
    if scale <= 0 or any(value < 0 for value in task_weights):
        raise ValueError("task weights must be nonnegative with positive sum")
    bce = sum(
        weight * term for weight, term in zip(task_weights, bce_terms)
    ) / scale
    ranking = (
        torch.stack(ranking_terms).mean()
        if ranking_terms else logits.sum() * 0
    )
    total = bce + ranking_weight * ranking
    return total, {
        "bce": bce.detach(), "ranking": ranking.detach(),
        "voice": bce_terms[0].detach(), "music": bce_terms[1].detach(),
        "file": bce_terms[2].detach(),
    }


def group_dro_patch_loss(
    model: EatPatchGraphHead,
    logits: torch.Tensor,
    targets: torch.Tensor,
    presence: torch.Tensor,
    sample_weight: torch.Tensor,
    groups: torch.Tensor,
    group_weights: torch.Tensor,
    step_size: float,
    task_weights: tuple[float, float, float] = (.20, .35, .45),
    ranking_weight: float = .15,
    ranking_tail_fraction: float = 1.,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Apply online GroupDRO over corpus groups in the current mini-batch."""
    if groups.shape != (len(logits),) or groups.dtype != torch.long:
        raise ValueError("groups must be a long tensor with one value per item")
    if group_weights.ndim != 1 or len(group_weights) == 0:
        raise ValueError("group weights must be a nonempty vector")
    if groups.min() < 0 or groups.max() >= len(group_weights):
        raise ValueError("group index lies outside the weight vector")
    if step_size <= 0:
        raise ValueError("GroupDRO step size must be positive")

    present = groups.unique(sorted=True)
    losses = []
    for group in present:
        selected = groups.eq(group)
        current, _ = patch_graph_loss(
            model, logits[selected], targets[selected], presence[selected],
            sample_weight[selected], task_weights=task_weights,
            ranking_weight=ranking_weight,
            ranking_tail_fraction=ranking_tail_fraction,
        )
        losses.append(current)
    stacked = torch.stack(losses)
    with torch.no_grad():
        group_weights[present] *= torch.exp(
            step_size * stacked.detach().clamp(max=10)
        )
        group_weights /= group_weights.sum().clamp_min(1e-12)
    robust_weights = group_weights[present]
    robust_weights = robust_weights / robust_weights.sum().clamp_min(1e-12)
    total = torch.sum(robust_weights * stacked)
    return total, {
        "group_min": stacked.detach().min(),
        "group_mean": stacked.detach().mean(),
        "group_max": stacked.detach().max(),
        "group_weight_max": group_weights.detach().max(),
    }
