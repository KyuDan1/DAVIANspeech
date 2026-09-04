"""Component-query MHFA over cached EAT patches and SPEAR temporal bins.

The model is intentionally separation-free.  Five learned queries read the
same original-mixture evidence, but can select different SSL layers, temporal
regions, frequency regions, and streams for Voice/Music/File authenticity and
Voice/Music presence.  SPEAR tokens include signed first/second differences so
that sequential and partially manipulated boundaries are not reduced to an
unsigned utterance statistic.
"""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


class SwiGLUBlock(nn.Module):
    """Cheap token-local residual block; avoids quadratic token attention."""

    def __init__(self, width: int, expansion: int = 3, dropout: float = .15) -> None:
        super().__init__()
        hidden = width * expansion
        self.norm = nn.LayerNorm(width)
        self.gate = nn.Linear(width, 2 * hidden)
        self.output = nn.Linear(hidden, width)
        self.dropout = nn.Dropout(dropout)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        left, right = self.gate(self.norm(values)).chunk(2, dim=-1)
        update = self.output(F.silu(left) * right)
        return values + self.dropout(update)


class ComponentQueryMHFA(nn.Module):
    """Five-query multi-head factorized attentive-statistics detector."""

    VOICE_FAKE = 0
    MUSIC_FAKE = 1
    FILE_FAKE = 2
    VOICE_PRESENT = 3
    MUSIC_PRESENT = 4
    TASKS = 5

    def __init__(
        self,
        eat_layers: int = 6,
        eat_dimension: int = 128,
        spear_layers: int = 4,
        spear_stats: int = 4,
        spear_dimension: int = 64,
        width: int = 96,
        heads: int = 8,
        depth: int = 2,
        maximum_views: int = 3,
        maximum_time_nodes: int = 64,
        maximum_frequency_nodes: int = 16,
        maximum_bins: int = 8,
        dropout: float = .15,
        file_component_weight: float = .10,
    ) -> None:
        super().__init__()
        if min(eat_layers, eat_dimension, spear_layers, spear_dimension, width,
               heads, depth, maximum_views, maximum_bins) <= 0:
            raise ValueError("all model dimensions must be positive")
        if width % heads:
            raise ValueError("width must be divisible by heads")
        if not 0 <= file_component_weight <= 1:
            raise ValueError("file_component_weight must lie in [0, 1]")
        self.eat_layers = int(eat_layers)
        self.spear_layers = int(spear_layers)
        self.spear_stats = int(spear_stats)
        self.width = int(width)
        self.heads = int(heads)
        self.maximum_views = int(maximum_views)
        self.maximum_time_nodes = int(maximum_time_nodes)
        self.maximum_frequency_nodes = int(maximum_frequency_nodes)
        self.maximum_bins = int(maximum_bins)
        self.file_component_weight = float(file_component_weight)

        self.eat_norm = nn.LayerNorm(eat_dimension)
        self.spear_norm = nn.LayerNorm(3 * spear_dimension)
        self.eat_projection = nn.Linear(eat_dimension, width)
        self.spear_projection = nn.Linear(3 * spear_dimension, width)
        self.eat_layer_logits = nn.Parameter(torch.zeros(self.TASKS, eat_layers))
        self.spear_layer_logits = nn.Parameter(torch.zeros(self.TASKS, spear_layers))
        self.task_embedding = nn.Parameter(torch.empty(self.TASKS, width))
        self.stream_embedding = nn.Parameter(torch.empty(2, width))
        self.view_embedding = nn.Parameter(torch.empty(maximum_views, width))
        self.eat_stat_embedding = nn.Parameter(torch.empty(4, width))
        self.time_position = nn.Parameter(torch.empty(maximum_time_nodes, width))
        self.frequency_position = nn.Parameter(
            torch.empty(maximum_frequency_nodes, width)
        )
        self.spear_stat_embedding = nn.Parameter(torch.empty(spear_stats, width))
        self.bin_embedding = nn.Parameter(torch.empty(maximum_bins, width))
        for parameter in (
            self.task_embedding, self.stream_embedding, self.view_embedding,
            self.eat_stat_embedding, self.time_position,
            self.frequency_position, self.spear_stat_embedding,
            self.bin_embedding,
        ):
            nn.init.normal_(parameter, std=.02)

        self.eat_blocks = nn.ModuleList([
            SwiGLUBlock(width, dropout=dropout) for _ in range(depth)
        ])
        self.spear_blocks = nn.ModuleList([
            SwiGLUBlock(width, dropout=dropout) for _ in range(depth)
        ])
        self.eat_queries = nn.Parameter(torch.empty(self.TASKS, heads, width))
        self.spear_queries = nn.Parameter(torch.empty(self.TASKS, heads, width))
        nn.init.normal_(self.eat_queries, std=.02)
        nn.init.normal_(self.spear_queries, std=.02)
        self.eat_value = nn.Linear(width, width)
        self.spear_value = nn.Linear(width, width)
        pooled_width = 4 * width  # mean/std from each of two SSL streams.
        self.classifiers = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(pooled_width), nn.Linear(pooled_width, width),
                nn.SiLU(), nn.Dropout(dropout), nn.Linear(width, 1),
            )
            for _ in range(self.TASKS)
        ])
        self.joint_head = nn.Sequential(
            nn.LayerNorm(pooled_width), nn.Linear(pooled_width, width),
            nn.SiLU(), nn.Dropout(dropout), nn.Linear(width, 4),
        )

    @staticmethod
    def signed_differences(values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return zero-left-padded directional first and second bin deltas."""
        first = torch.zeros_like(values)
        first[:, :, 1:] = values[:, :, 1:] - values[:, :, :-1]
        second = torch.zeros_like(values)
        second[:, :, 2:] = first[:, :, 2:] - first[:, :, 1:-1]
        return first, second

    def _eat_tokens(
        self, temporal: torch.Tensor, spectral: torch.Tensor,
        view_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if temporal.ndim != 6 or spectral.ndim != 6:
            raise ValueError("EAT axes must be [batch,view,layer,stat,node,dim]")
        batch, views, layers, statistics, time_nodes, _ = temporal.shape
        if statistics != 2 or layers != self.eat_layers:
            raise ValueError("EAT layer/statistic dimensions differ")
        if spectral.shape[:4] != (batch, views, layers, statistics):
            raise ValueError("EAT temporal and spectral axes differ")
        frequency_nodes = spectral.shape[4]
        if view_mask.shape != (batch, views):
            raise ValueError("EAT view mask differs")
        if (views > self.maximum_views or time_nodes > self.maximum_time_nodes
                or frequency_nodes > self.maximum_frequency_nodes):
            raise ValueError("EAT token count exceeds configured maximum")

        layer_weight = self.eat_layer_logits.softmax(dim=-1)
        temporal = self.eat_projection(self.eat_norm(temporal.float()))
        spectral = self.eat_projection(self.eat_norm(spectral.float()))
        temporal = torch.einsum("bvlstd,ql->bvqstd", temporal, layer_weight)
        spectral = torch.einsum("bvlstd,ql->bvqstd", spectral, layer_weight)
        temporal = (
            temporal + self.task_embedding[None, None, :, None, None]
            + self.stream_embedding[0]
            + self.view_embedding[None, :views, None, None, None]
            + self.eat_stat_embedding[None, None, None, :2, None]
            + self.time_position[None, None, None, None, :time_nodes]
        ).permute(0, 2, 1, 3, 4, 5).reshape(
            batch, self.TASKS, views * 2 * time_nodes, self.width
        )
        spectral = (
            spectral + self.task_embedding[None, None, :, None, None]
            + self.stream_embedding[0]
            + self.view_embedding[None, :views, None, None, None]
            + self.eat_stat_embedding[None, None, None, 2:, None]
            + self.frequency_position[None, None, None, None, :frequency_nodes]
        ).permute(0, 2, 1, 3, 4, 5).reshape(
            batch, self.TASKS, views * 2 * frequency_nodes, self.width
        )
        mask_time = view_mask[:, :, None, None].expand(
            -1, -1, 2, time_nodes
        ).reshape(batch, -1)
        mask_frequency = view_mask[:, :, None, None].expand(
            -1, -1, 2, frequency_nodes
        ).reshape(batch, -1)
        values = torch.cat((temporal, spectral), dim=2)
        mask = torch.cat((mask_time, mask_frequency), dim=1).bool()
        for block in self.eat_blocks:
            values = block(values)
        return values, mask

    def _spear_tokens(
        self, spear: torch.Tensor, spear_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if spear.ndim != 6:
            raise ValueError("SPEAR must be [batch,view,bin,layer,stat,dim]")
        batch, views, bins, layers, statistics, _ = spear.shape
        if layers != self.spear_layers or statistics != self.spear_stats:
            raise ValueError("SPEAR layer/statistic dimensions differ")
        if spear_mask.shape != (batch, views, bins):
            raise ValueError("SPEAR temporal mask differs")
        if views > self.maximum_views or bins > self.maximum_bins:
            raise ValueError("SPEAR token count exceeds configured maximum")
        spear = spear.float()
        first, second = self.signed_differences(spear)
        spear = torch.cat((spear, first, second), dim=-1)
        spear = self.spear_projection(self.spear_norm(spear))
        layer_weight = self.spear_layer_logits.softmax(dim=-1)
        spear = torch.einsum("bvmlsd,ql->bvqmsd", spear, layer_weight)
        spear = (
            spear + self.task_embedding[None, None, :, None, None]
            + self.stream_embedding[1]
            + self.view_embedding[None, :views, None, None, None]
            + self.bin_embedding[None, None, None, :bins, None]
            + self.spear_stat_embedding[None, None, None, None, :statistics]
        ).permute(0, 2, 1, 3, 4, 5).reshape(
            batch, self.TASKS, views * bins * statistics, self.width
        )
        mask = spear_mask[:, :, :, None].expand(
            -1, -1, -1, statistics
        ).reshape(batch, -1).bool()
        for block in self.spear_blocks:
            spear = block(spear)
        return spear, mask

    def _pool(
        self, tokens: torch.Tensor, mask: torch.Tensor,
        queries: torch.Tensor, value_layer: nn.Linear,
    ) -> torch.Tensor:
        batch, tasks, count, width = tokens.shape
        scores = torch.einsum("bqnd,qhd->bqhn", tokens, queries) / math.sqrt(width)
        scores = scores.masked_fill(~mask[:, None, None], -1e4)
        attention = scores.softmax(dim=-1)
        head_width = width // self.heads
        values = value_layer(tokens).reshape(
            batch, tasks, count, self.heads, head_width
        ).permute(0, 1, 3, 2, 4)
        mean = torch.einsum("bqhn,bqhnk->bqhk", attention, values)
        second = torch.einsum(
            "bqhn,bqhnk->bqhk", attention, values.square()
        )
        std = (second - mean.square()).clamp_min(1e-5).sqrt()
        return torch.cat((mean.flatten(2), std.flatten(2)), dim=-1)

    def forward_with_embedding(
        self,
        temporal: torch.Tensor,
        spectral: torch.Tensor,
        eat_mask: torch.Tensor,
        spear: torch.Tensor,
        spear_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        eat, eat_token_mask = self._eat_tokens(temporal, spectral, eat_mask)
        spear, spear_token_mask = self._spear_tokens(spear, spear_mask)
        eat_pooled = self._pool(
            eat, eat_token_mask, self.eat_queries, self.eat_value
        )
        spear_pooled = self._pool(
            spear, spear_token_mask, self.spear_queries, self.spear_value
        )
        pooled = torch.cat((eat_pooled, spear_pooled), dim=-1)
        logits = torch.cat([
            classifier(pooled[:, task]).unsqueeze(-1)
            for task, classifier in enumerate(self.classifiers)
        ], dim=-1).squeeze(-2)
        joint = self.joint_head(pooled[:, self.FILE_FAKE])
        return logits[:, :3], logits[:, 3:], joint

    def forward(self, *inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.forward_with_embedding(*inputs)

    def probabilities(self, authenticity_logits: torch.Tensor) -> torch.Tensor:
        direct = authenticity_logits.sigmoid()
        component_or = 1 - (1 - direct[:, 0]) * (1 - direct[:, 1])
        direct = direct.clone()
        direct[:, 2] = (
            (1 - self.file_component_weight) * direct[:, 2]
            + self.file_component_weight * component_or
        )
        return direct


def component_query_loss(
    model: ComponentQueryMHFA,
    authenticity_logits: torch.Tensor,
    presence_logits: torch.Tensor,
    joint_logits: torch.Tensor,
    fake_targets: torch.Tensor,
    presence_targets: torch.Tensor,
    sample_weight: torch.Tensor,
    task_weights: tuple[float, float, float] = (.05, .45, .50),
    presence_weight: float = .05,
    joint_weight: float = .15,
    ranking_weight: float = .15,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Masked component objectives plus mixed-cell and EER rank supervision."""
    if authenticity_logits.shape != fake_targets.shape:
        raise ValueError("authenticity logits and labels must have shape [batch,3]")
    if presence_logits.shape != presence_targets.shape:
        raise ValueError("presence logits and labels must have shape [batch,2]")

    masks = (
        presence_targets[:, 0].bool(), presence_targets[:, 1].bool(),
        torch.ones(len(fake_targets), dtype=torch.bool, device=fake_targets.device),
    )
    bce, ranking = [], []
    for task, mask in enumerate(masks):
        raw = F.binary_cross_entropy_with_logits(
            authenticity_logits[mask, task], fake_targets[mask, task],
            reduction="none",
        )
        weights = sample_weight[mask]
        bce.append((raw * weights).sum() / weights.sum().clamp_min(1e-8))
        positive = authenticity_logits[mask & fake_targets[:, task].eq(1), task]
        negative = authenticity_logits[mask & fake_targets[:, task].eq(0), task]
        if len(positive) and len(negative):
            ranking.append(F.softplus(
                -(positive[:, None] - negative[None, :])
            ).mean())
    authenticity = sum(
        weight * term for weight, term in zip(task_weights, bce)
    ) / sum(task_weights)
    presence = F.binary_cross_entropy_with_logits(
        presence_logits, presence_targets, reduction="none"
    ).mean(dim=1)
    presence = (presence * sample_weight).sum() / sample_weight.sum().clamp_min(1e-8)
    mixed = presence_targets[:, 0].bool() & presence_targets[:, 1].bool()
    if mixed.any():
        target = (
            2 * fake_targets[mixed, 0].long() + fake_targets[mixed, 1].long()
        )
        raw_joint = F.cross_entropy(joint_logits[mixed], target, reduction="none")
        weights = sample_weight[mixed]
        joint = (raw_joint * weights).sum() / weights.sum().clamp_min(1e-8)
    else:
        joint = authenticity_logits.sum() * 0
    rank = torch.stack(ranking).mean() if ranking else authenticity_logits.sum() * 0
    total = (
        authenticity + presence_weight * presence + joint_weight * joint
        + ranking_weight * rank
    )
    return total, {
        "voice": bce[0].detach(), "music": bce[1].detach(),
        "file": bce[2].detach(), "presence": presence.detach(),
        "joint": joint.detach(), "ranking": rank.detach(),
    }
