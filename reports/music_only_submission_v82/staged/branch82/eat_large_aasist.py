"""Separation-free EAT-large to AASIST component-authenticity expert.

The graph layers follow the public AASIST design (spectral graph, temporal
graph, heterogeneous graph and master node), but consume EAT patch tokens
instead of a RawNet2 feature map.  No source separator is exposed here.

Reference implementation: https://github.com/clovaai/aasist (MIT).
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
from torch import nn
from torch.nn import functional as F


def _block_forward(block: nn.Module, values: torch.Tensor) -> torch.Tensor:
    output = block(values)
    return output[0] if isinstance(output, tuple) else output


class BottleneckAdapter(nn.Module):
    """Small zero-initialised residual used in the last EAT blocks."""

    def __init__(self, width: int, bottleneck: int = 32) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(width)
        self.down = nn.Linear(width, bottleneck)
        self.up = nn.Linear(bottleneck, width)
        self.scale = nn.Parameter(torch.tensor(0.1))
        nn.init.normal_(self.down.weight, std=0.02)
        nn.init.zeros_(self.down.bias)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        update = self.up(F.gelu(self.down(self.norm(values))))
        return values + self.scale.tanh() * update


class GraphPool(nn.Module):
    """AASIST top-k graph pooling."""

    def __init__(self, ratio: float, width: int, dropout: float = 0.3) -> None:
        super().__init__()
        if not 0 < ratio <= 1:
            raise ValueError("pool ratio must lie in (0, 1]")
        self.ratio = float(ratio)
        self.score = nn.Linear(width, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, nodes: torch.Tensor) -> torch.Tensor:
        score = self.score(self.dropout(nodes)).sigmoid()
        count = max(1, int(nodes.shape[1] * self.ratio))
        index = score.topk(count, dim=1).indices.expand(-1, -1, nodes.shape[-1])
        return torch.gather(nodes * score, 1, index)


class GraphAttentionLayer(nn.Module):
    """Homogeneous multiplicative graph attention from AASIST."""

    def __init__(
        self, input_width: int, output_width: int, temperature: float = 2.0,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.temperature = float(temperature)
        self.att_projection = nn.Linear(input_width, output_width)
        self.att_weight = nn.Parameter(torch.empty(output_width, 1))
        self.with_attention = nn.Linear(input_width, output_width)
        self.without_attention = nn.Linear(input_width, output_width)
        self.norm = nn.LayerNorm(output_width)
        self.dropout = nn.Dropout(dropout)
        nn.init.xavier_normal_(self.att_weight)

    def forward(self, nodes: torch.Tensor) -> torch.Tensor:
        values = self.dropout(nodes)
        pairwise = values[:, :, None] * values[:, None, :]
        score = torch.tanh(self.att_projection(pairwise)) @ self.att_weight
        attention = (score.squeeze(-1) / self.temperature).softmax(dim=-2)
        update = self.with_attention(attention @ values)
        residual = self.without_attention(values)
        return F.selu(self.norm(update + residual))


class HeterogeneousGraphAttention(nn.Module):
    """AASIST heterogeneous S/T graph attention with a master node."""

    def __init__(
        self, input_width: int, output_width: int, temperature: float = 100.0,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.temperature = float(temperature)
        self.type_s = nn.Linear(input_width, input_width)
        self.type_t = nn.Linear(input_width, input_width)
        self.att_projection = nn.Linear(input_width, output_width)
        self.master_projection = nn.Linear(input_width, output_width)
        self.weights = nn.Parameter(torch.empty(3, output_width, 1))
        self.master_weight = nn.Parameter(torch.empty(output_width, 1))
        self.with_attention = nn.Linear(input_width, output_width)
        self.without_attention = nn.Linear(input_width, output_width)
        self.master_with_attention = nn.Linear(input_width, output_width)
        self.master_without_attention = nn.Linear(input_width, output_width)
        self.norm = nn.LayerNorm(output_width)
        self.dropout = nn.Dropout(dropout)
        nn.init.xavier_normal_(self.weights)
        nn.init.xavier_normal_(self.master_weight)

    def forward(
        self, spectral: torch.Tensor, temporal: torch.Tensor,
        master: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        spectral = self.type_s(spectral)
        temporal = self.type_t(temporal)
        values = torch.cat((spectral, temporal), dim=1)
        spectral_count = spectral.shape[1]
        if master is None:
            master = values.mean(dim=1, keepdim=True)
        values = self.dropout(values)

        pairwise = values[:, :, None] * values[:, None, :]
        projected = torch.tanh(self.att_projection(pairwise))
        board = projected.new_empty((*projected.shape[:-1], 1))
        board[:, :spectral_count, :spectral_count] = (
            projected[:, :spectral_count, :spectral_count] @ self.weights[0]
        )
        board[:, spectral_count:, spectral_count:] = (
            projected[:, spectral_count:, spectral_count:] @ self.weights[1]
        )
        cross = projected[:, :spectral_count, spectral_count:] @ self.weights[2]
        board[:, :spectral_count, spectral_count:] = cross
        board[:, spectral_count:, :spectral_count] = (
            projected[:, spectral_count:, :spectral_count] @ self.weights[2]
        )
        attention = (board.squeeze(-1) / self.temperature).softmax(dim=-2)
        output = self.with_attention(attention @ values) + self.without_attention(values)
        output = self.norm(output)

        master_score = torch.tanh(
            self.master_projection(values * master)
        ) @ self.master_weight
        master_attention = (
            master_score.squeeze(-1) / self.temperature
        ).softmax(dim=-1)
        master_output = self.master_with_attention(
            (master_attention[:, :, None] * values).sum(dim=1, keepdim=True)
        ) + self.master_without_attention(master)
        return (
            output[:, :spectral_count], output[:, spectral_count:], master_output
        )


class HeterogeneousBranch(nn.Module):
    """Two-stage heterogeneous AASIST branch."""

    def __init__(self, width: int = 64, hidden: int = 32) -> None:
        super().__init__()
        self.first = HeterogeneousGraphAttention(width, hidden)
        self.second = HeterogeneousGraphAttention(hidden, hidden)
        self.pool_s = GraphPool(0.5, hidden)
        self.pool_t = GraphPool(0.5, hidden)
        self.dropout = nn.Dropout(0.2)

    def forward(
        self, spectral: torch.Tensor, temporal: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        spectral, temporal, master = self.first(spectral, temporal)
        spectral, temporal = self.pool_s(spectral), self.pool_t(temporal)
        update_s, update_t, update_m = self.second(spectral, temporal, master)
        return (
            self.dropout(spectral + update_s),
            self.dropout(temporal + update_t),
            self.dropout(master + update_m),
        )


class EATTokenAASIST(nn.Module):
    """AASIST graph backend over a weighted mixture of EAT patch layers."""

    TASKS = 3  # Voice, Music, File.

    def __init__(
        self, layer_count: int, input_width: int, graph_width: int = 64,
        graph_hidden: int = 32, branches: int = 2, dropout: float = 0.3,
        file_component_weight: float = 0.5,
    ) -> None:
        super().__init__()
        if layer_count <= 0 or input_width <= 0 or branches <= 0:
            raise ValueError("model dimensions must be positive")
        if not 0 <= file_component_weight <= 1:
            raise ValueError("file component weight must lie in [0, 1]")
        self.layer_count = int(layer_count)
        self.file_component_weight = float(file_component_weight)
        self.layer_logits = nn.Parameter(torch.zeros(layer_count))
        self.input_norm = nn.LayerNorm(input_width)
        self.input_projection = nn.Linear(input_width, graph_width)
        self.time_gate = nn.Linear(graph_width, 1)
        self.frequency_gate = nn.Linear(graph_width, 1)
        self.spectral_graph = GraphAttentionLayer(graph_width, graph_width)
        self.temporal_graph = GraphAttentionLayer(graph_width, graph_width)
        self.pool_s = GraphPool(0.5, graph_width)
        self.pool_t = GraphPool(0.5, graph_width)
        self.branches = nn.ModuleList([
            HeterogeneousBranch(graph_width, graph_hidden)
            for _ in range(branches)
        ])
        embedding_width = 5 * graph_hidden
        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(embedding_width), nn.Dropout(dropout),
                nn.Linear(embedding_width, graph_hidden), nn.SELU(),
                nn.Dropout(dropout), nn.Linear(graph_hidden, 1),
            )
            for _ in range(self.TASKS)
        ])

    def embedding(
        self, layers: torch.Tensor, grid_shape: tuple[int, int],
    ) -> torch.Tensor:
        if layers.ndim != 4:
            raise ValueError("EAT layers must have shape [batch, layer, patch, dim]")
        if layers.shape[1] != self.layer_count:
            raise ValueError("EAT layer count differs from backend configuration")
        time_nodes, frequency_nodes = map(int, grid_shape)
        if layers.shape[2] != time_nodes * frequency_nodes:
            raise ValueError("EAT patch count is incompatible with grid shape")
        weights = self.layer_logits.softmax(dim=0)
        values = torch.einsum("blnd,l->bnd", layers.float(), weights)
        values = self.input_projection(self.input_norm(values))
        grid = values.reshape(
            len(values), time_nodes, frequency_nodes, values.shape[-1]
        )

        time_weight = self.time_gate(grid).softmax(dim=1)
        spectral = (grid * time_weight).sum(dim=1)
        frequency_weight = self.frequency_gate(grid).softmax(dim=2)
        temporal = (grid * frequency_weight).sum(dim=2)
        spectral = self.pool_s(self.spectral_graph(spectral))
        temporal = self.pool_t(self.temporal_graph(temporal))

        branch_values = [branch(spectral, temporal) for branch in self.branches]
        spectral = torch.stack([item[0] for item in branch_values]).amax(dim=0)
        temporal = torch.stack([item[1] for item in branch_values]).amax(dim=0)
        master = torch.stack([item[2] for item in branch_values]).amax(dim=0)
        return torch.cat((
            spectral.abs().amax(dim=1), spectral.mean(dim=1),
            temporal.abs().amax(dim=1), temporal.mean(dim=1),
            master.squeeze(1),
        ), dim=-1)

    def forward(
        self, layers: torch.Tensor, grid_shape: tuple[int, int],
    ) -> torch.Tensor:
        embedding = self.embedding(layers, grid_shape)
        return torch.cat([head(embedding) for head in self.heads], dim=-1)

    def probabilities(self, logits: torch.Tensor) -> torch.Tensor:
        direct = logits.sigmoid()
        component_or = 1 - (1 - direct[:, 0]) * (1 - direct[:, 1])
        direct_file = direct[:, 2]
        file_probability = (
            (1 - self.file_component_weight) * direct_file
            + self.file_component_weight * component_or
        )
        return torch.stack((direct[:, 0], direct[:, 1], file_probability), dim=-1)


class EatLargeAASISTExpert(nn.Module):
    """Frozen EAT-large plus top-block adapters and an AASIST backend."""

    def __init__(
        self, eat_model: nn.Module,
        capture_layers: Sequence[int] = (3, 7, 11, 15, 19, 20, 21, 22, 23),
        adapter_blocks: Sequence[int] = (20, 21, 22, 23),
        bottleneck: int = 32, graph_width: int = 64,
        graph_hidden: int = 32, branches: int = 2, dropout: float = 0.3,
        temperature: float = 5.0, file_component_weight: float = 0.5,
    ) -> None:
        super().__init__()
        core = getattr(eat_model, "model", eat_model)
        required = ("local_encoder", "extra_tokens", "pre_norm", "pos_drop", "blocks")
        missing = [name for name in required if not hasattr(core, name)]
        if missing:
            raise TypeError(f"EAT core is missing attributes: {missing}")
        capture = tuple(int(value) for value in capture_layers)
        adapters = tuple(int(value) for value in adapter_blocks)
        if not capture or tuple(sorted(set(capture))) != capture:
            raise ValueError("capture layers must be increasing and unique")
        if not adapters or tuple(sorted(set(adapters))) != adapters:
            raise ValueError("adapter blocks must be increasing and unique")
        if capture[-1] >= len(core.blocks) or adapters[-1] >= len(core.blocks):
            raise ValueError("EAT block index is out of range")
        if adapters[0] not in capture:
            raise ValueError("first adapter block must be a captured layer")
        if temperature <= 0:
            raise ValueError("temperature must be positive")

        self.eat = eat_model
        self.capture_layers = capture
        self.adapter_blocks = adapters
        self.temperature = float(temperature)
        for parameter in self.eat.parameters():
            parameter.requires_grad_(False)
        width = int(core.extra_tokens.shape[-1])
        self.adapters = nn.ModuleDict({
            str(index): BottleneckAdapter(width, bottleneck)
            for index in adapters
        })
        self.backend = EATTokenAASIST(
            len(capture), width, graph_width, graph_hidden, branches, dropout,
            file_component_weight,
        )

    @property
    def core(self) -> nn.Module:
        return getattr(self.eat, "model", self.eat)

    def train(self, mode: bool = True):
        super().train(mode)
        self.eat.eval()
        return self

    def _layers(
        self, features: torch.Tensor,
    ) -> tuple[torch.Tensor, tuple[int, int]]:
        core = self.core
        patch_module = getattr(core.local_encoder, "proj", None)
        if patch_module is None:
            raise TypeError("EAT local encoder does not expose patch projection")
        first_adapter = self.adapter_blocks[0]
        captured: list[torch.Tensor] = []
        wanted = set(self.capture_layers)

        with torch.no_grad():
            grid = patch_module(features)
            batch, width, time_nodes, frequency_nodes = grid.shape
            values = grid.flatten(2).transpose(1, 2)
            positional = getattr(core, "fixed_positional_encoder", None)
            if positional is not None:
                values = values + positional(values, None)[:, : values.shape[1]]
            values = torch.cat((core.extra_tokens.expand(batch, -1, -1), values), 1)
            values = core.pos_drop(core.pre_norm(values))

        for index, block in enumerate(core.blocks):
            if index < first_adapter:
                with torch.no_grad():
                    values = _block_forward(block, values)
            else:
                values = _block_forward(block, values)
            if str(index) in self.adapters:
                values = self.adapters[str(index)](values)
            if index in wanted:
                # Earlier frozen layers should never retain an autograd graph.
                captured.append(values[:, 1:].float())
        return torch.stack(captured, dim=1), (time_nodes, frequency_nodes)

    def view_logits(self, features: torch.Tensor) -> torch.Tensor:
        layers, grid_shape = self._layers(features)
        return self.backend(layers, grid_shape)

    def aggregate(self, logits: torch.Tensor, view_mask: torch.Tensor) -> torch.Tensor:
        if logits.ndim != 3 or view_mask.shape != logits.shape[:2]:
            raise ValueError("view logits/mask shapes are incompatible")
        mask = view_mask.bool()
        if not torch.all(mask.any(dim=1)):
            raise ValueError("every item needs at least one valid view")
        scaled = (self.temperature * logits).masked_fill(
            ~mask[:, :, None], -1e4
        )
        normalizer = mask.sum(dim=1).float().log()[:, None]
        return (torch.logsumexp(scaled, dim=1) - normalizer) / self.temperature

    def forward(
        self, features: torch.Tensor, view_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if features.ndim == 4:
            return self.view_logits(features)
        if features.ndim != 5:
            raise ValueError("features must be [B,1,T,F] or [B,V,1,T,F]")
        batch, views = features.shape[:2]
        logits = self.view_logits(features.flatten(0, 1)).reshape(batch, views, -1)
        if view_mask is None:
            view_mask = torch.ones(
                batch, views, dtype=torch.bool, device=features.device
            )
        return self.aggregate(logits, view_mask)

    def expert_state_dict(self) -> dict[str, object]:
        return {
            "model_type": "eat_large_aasist_v56",
            "capture_layers": self.capture_layers,
            "adapter_blocks": self.adapter_blocks,
            "temperature": self.temperature,
            "model": {
                name: value.detach().cpu()
                for name, value in self.state_dict().items()
                if not name.startswith("eat.")
            },
        }


def multitask_loss(
    model: EatLargeAASISTExpert,
    logits: torch.Tensor,
    targets: torch.Tensor,
    presence: torch.Tensor,
    sample_weight: torch.Tensor | None = None,
    task_weights: tuple[float, float, float] = (0.20, 0.30, 0.50),
    ranking_weight: float = 0.10,
    component_or_weight: float = 0.05,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Masked multitask BCE with EER ranking and File/component consistency."""
    if logits.shape != targets.shape or logits.ndim != 2 or logits.shape[1] != 3:
        raise ValueError("logits and targets must have shape [batch, 3]")
    if presence.shape != (len(logits), 2):
        raise ValueError("presence must have shape [batch, 2]")
    if sample_weight is None:
        sample_weight = torch.ones(len(logits), device=logits.device)
    masks = (
        presence[:, 0].bool(), presence[:, 1].bool(),
        torch.ones(len(logits), dtype=torch.bool, device=logits.device),
    )
    bce_terms, rank_terms = [], []
    for task, mask in enumerate(masks):
        if not mask.any():
            bce_terms.append(logits.sum() * 0)
            continue
        raw = F.binary_cross_entropy_with_logits(
            logits[mask, task], targets[mask, task], reduction="none"
        )
        weight = sample_weight[mask]
        bce_terms.append((raw * weight).sum() / weight.sum().clamp_min(1e-8))
        positive = logits[mask & targets[:, task].eq(1), task]
        negative = logits[mask & targets[:, task].eq(0), task]
        if len(positive) and len(negative):
            rank_terms.append(F.softplus(
                -(positive[:, None] - negative[None, :])
            ).mean())
    bce = sum(weight * value for weight, value in zip(task_weights, bce_terms))
    ranking = torch.stack(rank_terms).mean() if rank_terms else logits.sum() * 0
    direct = logits.sigmoid()
    component_or = 1 - (1 - direct[:, 0]) * (1 - direct[:, 1])
    consistency = F.smooth_l1_loss(direct[:, 2], component_or)
    total = bce + ranking_weight * ranking + component_or_weight * consistency
    return total, {
        "bce": bce.detach(), "ranking": ranking.detach(),
        "component_or": consistency.detach(),
        "voice": bce_terms[0].detach(), "music": bce_terms[1].detach(),
        "file": bce_terms[2].detach(),
    }

