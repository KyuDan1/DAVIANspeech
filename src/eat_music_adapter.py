"""Parameter-efficient music-forensics adaptation of the bundled EAT model.

The adapter consumes the original mixture.  It deliberately does not expose a
source-separation input: generator traces must survive all the way to EAT.
Only small bottleneck residuals after the final encoder blocks and the pooling
head are trainable; the shared EAT checkpoint remains frozen.
"""

from __future__ import annotations

import math

import torch
from torch import nn


class BottleneckAdapter(nn.Module):
    """Zero-initialized residual adapter preserving the pretrained model at init."""

    def __init__(self, width: int, bottleneck: int, dropout: float = 0.1) -> None:
        super().__init__()
        if width <= 0 or bottleneck <= 0:
            raise ValueError("adapter dimensions must be positive")
        self.norm = nn.LayerNorm(width)
        self.down = nn.Linear(width, bottleneck)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.up = nn.Linear(bottleneck, width)
        self.scale = nn.Parameter(torch.tensor(0.1))
        nn.init.normal_(self.down.weight, std=0.02)
        nn.init.zeros_(self.down.bias)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        update = self.up(self.dropout(self.activation(self.down(self.norm(values)))))
        return values + self.scale.tanh() * update


class EatMusicAdapter(nn.Module):
    """Adapt the last EAT blocks and predict mixture-native Music authenticity.

    Earlier EAT blocks execute without an autograd graph.  Gradients begin at
    the first trainable adapter, flow through the remaining frozen EAT blocks,
    and update only adapters/pooling.  This substantially lowers memory while
    keeping the base checkpoint byte-identical for offline reuse.
    """

    STATISTICS = 4  # CLS, patch mean, patch std, temporal delta.

    def __init__(
        self,
        eat_model: nn.Module,
        adapter_blocks: tuple[int, ...] = (10, 11),
        bottleneck: int = 32,
        pooling_heads: int = 4,
        hidden: int = 192,
        pooling_width: int = 128,
        dropout: float = 0.15,
        temperature: float = 5.0,
    ) -> None:
        super().__init__()
        core = getattr(eat_model, "model", eat_model)
        required = ("local_encoder", "extra_tokens", "pre_norm", "pos_drop", "blocks")
        missing = [name for name in required if not hasattr(core, name)]
        if missing:
            raise TypeError(f"EAT core is missing attributes: {missing}")
        selected = tuple(int(index) for index in adapter_blocks)
        if not selected or tuple(sorted(set(selected))) != selected:
            raise ValueError("adapter blocks must be nonempty, unique, and increasing")
        if selected[0] < 0 or selected[-1] >= len(core.blocks):
            raise ValueError("adapter block index is out of range")
        width = int(core.extra_tokens.shape[-1])
        if pooling_heads <= 0 or pooling_width <= 0:
            raise ValueError("pooling dimensions must be positive")
        if temperature <= 0:
            raise ValueError("temperature must be positive")

        self.eat = eat_model
        self.adapter_blocks = selected
        self.temperature = float(temperature)
        for parameter in self.eat.parameters():
            parameter.requires_grad_(False)
        self.adapters = nn.ModuleDict({
            str(index): BottleneckAdapter(width, bottleneck, dropout)
            for index in selected
        })
        self.token_norm = nn.LayerNorm(width)
        self.token_projection = nn.Linear(width, pooling_width)
        self.layer_logits = nn.Parameter(
            torch.zeros(pooling_heads, len(selected))
        )
        self.cls_layer_logits = nn.Parameter(torch.zeros(len(selected)))
        self.pool_queries = nn.Parameter(torch.empty(pooling_heads, pooling_width))
        nn.init.normal_(self.pool_queries, std=0.02)
        pooled_width = (2 * pooling_heads + 1) * pooling_width
        self.classifier = nn.Sequential(
            nn.LayerNorm(pooled_width),
            nn.Linear(pooled_width, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    @property
    def core(self) -> nn.Module:
        return getattr(self.eat, "model", self.eat)

    def train(self, mode: bool = True):
        super().train(mode)
        # The backbone contains no active stochastic layers in the distributed
        # checkpoint, but keeping it in eval is an explicit reproducibility
        # contract if a future checkpoint changes that assumption.
        self.eat.eval()
        return self

    def _tokens(self, features: torch.Tensor) -> list[torch.Tensor]:
        core = self.core
        batch = features.shape[0]
        with torch.no_grad():
            values = core.local_encoder(features)
            positional = getattr(core, "fixed_positional_encoder", None)
            if positional is not None:
                values = values + positional(values, None)[:, : values.size(1)]
            values = torch.cat((core.extra_tokens.expand(batch, -1, -1), values), 1)
            values = core.pos_drop(core.pre_norm(values))

        outputs = []
        first = self.adapter_blocks[0]
        for index, block in enumerate(core.blocks):
            # Blocks before the first adapter need no graph.  Later frozen
            # blocks retain input gradients so an earlier adapter can learn.
            if index < first:
                with torch.no_grad():
                    values, _ = block(values)
            else:
                values, _ = block(values)
            if str(index) in self.adapters:
                values = self.adapters[str(index)](values)
                outputs.append(values)
        return outputs

    def view_logits(self, features: torch.Tensor) -> torch.Tensor:
        # Preserve every EAT patch until task attention.  This is the critical
        # distinction from earlier shallow heads that collapsed a six-second
        # crop to four global statistics before learning where artifacts live.
        layers = torch.stack(self._tokens(features), dim=1).float()
        cls = self.token_projection(self.token_norm(layers[:, :, 0]))
        patches = self.token_projection(self.token_norm(layers[:, :, 1:]))
        layer_weight = self.layer_logits.softmax(dim=-1)
        values = torch.einsum("blnp,hl->bhnp", patches, layer_weight)
        score = torch.einsum("bhnp,hp->bhn", values, self.pool_queries) / math.sqrt(
            values.shape[-1]
        )
        attention = score.softmax(dim=-1)
        mean = torch.einsum("bhn,bhnp->bhp", attention, values)
        second = torch.einsum("bhn,bhnp->bhp", attention, values.square())
        std = (second - mean.square()).clamp_min(1e-5).sqrt()
        cls = torch.einsum("blp,l->bp", cls, self.cls_layer_logits.softmax(dim=0))
        pooled = torch.cat((mean.flatten(1), std.flatten(1), cls), dim=1)
        return self.classifier(pooled).squeeze(-1)

    def aggregate(self, logits: torch.Tensor, view_mask: torch.Tensor) -> torch.Tensor:
        if logits.ndim != 2 or logits.shape != view_mask.shape:
            raise ValueError("view logits and mask must have shape [batch, views]")
        mask = view_mask.bool()
        if not torch.all(mask.any(dim=1)):
            raise ValueError("every example needs one valid view")
        scaled = (self.temperature * logits).masked_fill(~mask, -1e4)
        normalizer = mask.sum(dim=1).float().log()
        return (torch.logsumexp(scaled, dim=1) - normalizer) / self.temperature

    def forward(
        self, features: torch.Tensor, view_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        if features.ndim == 4:
            return self.view_logits(features)
        if features.ndim != 5:
            raise ValueError("features must be [B,1,T,F] or [B,V,1,T,F]")
        batch, views = features.shape[:2]
        logits = self.view_logits(features.flatten(0, 1)).reshape(batch, views)
        if view_mask is None:
            view_mask = torch.ones(
                batch, views, dtype=torch.bool, device=features.device
            )
        return self.aggregate(logits, view_mask)

    def adapter_state_dict(self) -> dict[str, object]:
        return {
            "model_type": "eat_music_adapter_v55",
            "adapter_blocks": self.adapter_blocks,
            "temperature": self.temperature,
            "model": {
                name: value.detach().cpu()
                for name, value in self.state_dict().items()
                if not name.startswith("eat.")
            },
        }


def pairwise_ranking_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Smooth EER-aligned loss over every positive/negative pair in a batch."""
    positive = logits[targets.eq(1)]
    negative = logits[targets.eq(0)]
    if not len(positive) or not len(negative):
        return logits.new_zeros(())
    return torch.nn.functional.softplus(
        -(positive[:, None] - negative[None, :])
    ).mean()
