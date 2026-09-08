"""Three-stream component-query head that learns bounded anchor residuals.

The frozen anchor remains the owner of presence.  EAT patch statistics, SPEAR
temporal bins, and (optionally) ordered XLS-R window embeddings only predict
three authenticity-logit residuals.  The last linear layer is zero initialized,
so a fresh checkpoint is an exact identity transform of the anchor.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
from torch import nn
from torch.nn import functional as F

try:  # Package import in tests; flat import in offline training scripts.
    from .component_query_mhfa import SwiGLUBlock
except ImportError:  # pragma: no cover
    from component_query_mhfa import SwiGLUBlock


XLSR_WINDOW_DIMENSION = 1920
AUTHENTICITY_TASKS = 3


def noisy_or_logit(voice_logit: torch.Tensor, music_logit: torch.Tensor) -> torch.Tensor:
    """Return the stable logit of ``1-(1-sigmoid(v))*(1-sigmoid(m))``.

    The noisy-OR odds are ``exp(v) + exp(m) + exp(v+m)``.  Expressing the
    result as a log-sum-exp avoids probability clipping and remains finite for
    all finite practical logits.
    """
    if voice_logit.shape != music_logit.shape:
        raise ValueError("Voice and Music logits must have identical shapes")
    return torch.logsumexp(
        torch.stack((voice_logit, music_logit, voice_logit + music_logit), dim=-1),
        dim=-1,
    )


def _require_bool_mask(mask: torch.Tensor, shape: tuple[int, ...], name: str) -> None:
    if mask.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {tuple(mask.shape)}")
    if mask.dtype != torch.bool:
        raise TypeError(f"{name} must use torch.bool")


def _require_prefix_mask(mask: torch.Tensor, name: str) -> None:
    """Require valid temporal entries to form one nonempty prefix when present."""
    if mask.shape[-1] == 0:
        raise ValueError(f"{name} cannot have an empty temporal axis")
    if torch.any(mask[..., 1:] & ~mask[..., :-1]):
        raise ValueError(f"{name} valid entries must form a prefix")


def _require_finite(values: torch.Tensor, name: str) -> None:
    if not torch.is_floating_point(values):
        raise TypeError(f"{name} must be floating point")
    if not torch.isfinite(values).all():
        raise ValueError(f"{name} contains non-finite values")


class TaskInteractionBlock(nn.Module):
    """Let the three component queries exchange evidence after stream pooling."""

    def __init__(self, width: int, heads: int, dropout: float) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(width)
        self.attention = nn.MultiheadAttention(
            width, heads, dropout=dropout, batch_first=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.feed_forward = SwiGLUBlock(width, dropout=dropout)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        normalized = self.norm(values)
        update, _ = self.attention(
            normalized, normalized, normalized, need_weights=False,
        )
        return self.feed_forward(values + self.dropout(update))


class ThreeStreamAnchorResidualHead(nn.Module):
    """Component-query MHFA residual over EAT, SPEAR, and optional XLS-R.

    Input shapes are:

    * EAT temporal/spectral: ``[B,V,L,2,N,D]`` with ``eat_mask [B,V]``.
    * SPEAR bins: ``[B,V,K,L,S,D]`` with ``spear_mask [B,V,K]``.
    * XLS-R windows: ``[B,W,1920]`` with ``xlsr_mask [B,W]``.
    * Anchor authenticity/presence logits: ``[B,3]`` and ``[B,2]``.

    XLS-R is disabled by constructing the head with ``xlsr_dimension=None``;
    in that configuration both XLS-R inputs must be omitted.  Presence logits
    are returned unchanged and are never predicted by this module.
    """

    VOICE_FAKE = 0
    MUSIC_FAKE = 1
    FILE_FAKE = 2
    TASKS = AUTHENTICITY_TASKS
    FILE_DIRECT_WEIGHT = 0.7
    FILE_COMPONENT_WEIGHT = 0.3

    def __init__(
        self,
        eat_layers: int = 6,
        eat_dimension: int = 128,
        spear_layers: int = 4,
        spear_stats: int = 4,
        spear_dimension: int = 64,
        xlsr_dimension: int | None = XLSR_WINDOW_DIMENSION,
        width: int = 96,
        heads: int = 8,
        depth: int = 2,
        maximum_views: int = 3,
        maximum_time_nodes: int = 64,
        maximum_frequency_nodes: int = 16,
        maximum_bins: int = 8,
        maximum_xlsr_windows: int = 16,
        dropout: float = 0.15,
        residual_limits: Sequence[float] = (1.0, 1.0, 1.0),
        task_interaction_depth: int = 0,
    ) -> None:
        super().__init__()
        dimensions = (
            eat_layers, eat_dimension, spear_layers, spear_stats,
            spear_dimension, width, heads, depth, maximum_views,
            maximum_time_nodes, maximum_frequency_nodes, maximum_bins,
            maximum_xlsr_windows,
        )
        if min(dimensions) <= 0:
            raise ValueError("all model dimensions must be positive")
        if task_interaction_depth < 0:
            raise ValueError("task interaction depth cannot be negative")
        if width % heads:
            raise ValueError("width must be divisible by heads")
        if xlsr_dimension is not None and xlsr_dimension != XLSR_WINDOW_DIMENSION:
            raise ValueError(
                f"ordered XLS-R embeddings must be {XLSR_WINDOW_DIMENSION}-D"
            )
        limits = torch.as_tensor(tuple(residual_limits), dtype=torch.float32)
        if limits.shape != (self.TASKS,) or not torch.isfinite(limits).all():
            raise ValueError("residual_limits must contain three finite values")
        if not torch.all(limits > 0):
            raise ValueError("residual limits must be positive")

        self.eat_layers = int(eat_layers)
        self.eat_dimension = int(eat_dimension)
        self.spear_layers = int(spear_layers)
        self.spear_stats = int(spear_stats)
        self.spear_dimension = int(spear_dimension)
        self.xlsr_dimension = xlsr_dimension
        self.width = int(width)
        self.heads = int(heads)
        self.maximum_views = int(maximum_views)
        self.maximum_time_nodes = int(maximum_time_nodes)
        self.maximum_frequency_nodes = int(maximum_frequency_nodes)
        self.maximum_bins = int(maximum_bins)
        self.maximum_xlsr_windows = int(maximum_xlsr_windows)
        self.task_interaction_depth = int(task_interaction_depth)
        self.register_buffer("residual_limits", limits)

        self.eat_norm = nn.LayerNorm(eat_dimension)
        self.spear_norm = nn.LayerNorm(3 * spear_dimension)
        self.eat_projection = nn.Linear(eat_dimension, width)
        self.spear_projection = nn.Linear(3 * spear_dimension, width)
        self.eat_layer_logits = nn.Parameter(torch.zeros(self.TASKS, eat_layers))
        self.spear_layer_logits = nn.Parameter(torch.zeros(self.TASKS, spear_layers))
        self.task_embedding = nn.Parameter(torch.empty(self.TASKS, width))
        self.stream_embedding = nn.Parameter(torch.empty(3, width))
        self.view_embedding = nn.Parameter(torch.empty(maximum_views, width))
        self.eat_stat_embedding = nn.Parameter(torch.empty(4, width))
        self.time_position = nn.Parameter(torch.empty(maximum_time_nodes, width))
        self.frequency_position = nn.Parameter(
            torch.empty(maximum_frequency_nodes, width)
        )
        self.spear_stat_embedding = nn.Parameter(torch.empty(spear_stats, width))
        self.bin_embedding = nn.Parameter(torch.empty(maximum_bins, width))
        self.xlsr_position = nn.Parameter(torch.empty(maximum_xlsr_windows, width))
        parameters = (
            self.task_embedding, self.stream_embedding, self.view_embedding,
            self.eat_stat_embedding, self.time_position,
            self.frequency_position, self.spear_stat_embedding,
            self.bin_embedding, self.xlsr_position,
        )
        for parameter in parameters:
            nn.init.normal_(parameter, std=0.02)

        self.eat_blocks = nn.ModuleList([
            SwiGLUBlock(width, dropout=dropout) for _ in range(depth)
        ])
        self.spear_blocks = nn.ModuleList([
            SwiGLUBlock(width, dropout=dropout) for _ in range(depth)
        ])
        self.eat_queries = nn.Parameter(torch.empty(self.TASKS, heads, width))
        self.spear_queries = nn.Parameter(torch.empty(self.TASKS, heads, width))
        self.eat_value = nn.Linear(width, width)
        self.spear_value = nn.Linear(width, width)
        nn.init.normal_(self.eat_queries, std=0.02)
        nn.init.normal_(self.spear_queries, std=0.02)

        if xlsr_dimension is not None:
            self.xlsr_norm: nn.Module | None = nn.LayerNorm(xlsr_dimension)
            self.xlsr_projection: nn.Module | None = nn.Linear(xlsr_dimension, width)
            self.xlsr_blocks = nn.ModuleList([
                SwiGLUBlock(width, dropout=dropout) for _ in range(depth)
            ])
            self.xlsr_queries = nn.Parameter(torch.empty(self.TASKS, heads, width))
            self.xlsr_value: nn.Module | None = nn.Linear(width, width)
            nn.init.normal_(self.xlsr_queries, std=0.02)
        else:
            self.xlsr_norm = None
            self.xlsr_projection = None
            self.xlsr_blocks = nn.ModuleList()
            self.register_parameter("xlsr_queries", None)
            self.xlsr_value = None

        pooled_width = 6 * width  # attentive mean/std from each of three streams.
        if self.task_interaction_depth:
            self.task_interaction_in: nn.Module | None = nn.Linear(
                pooled_width, width,
            )
            self.task_interaction_blocks = nn.ModuleList([
                TaskInteractionBlock(width, heads, dropout)
                for _ in range(self.task_interaction_depth)
            ])
            self.task_interaction_out: nn.Module | None = nn.Linear(
                width, pooled_width,
            )
            nn.init.zeros_(self.task_interaction_out.weight)
            nn.init.zeros_(self.task_interaction_out.bias)
        else:
            self.task_interaction_in = None
            self.task_interaction_blocks = nn.ModuleList()
            self.task_interaction_out = None
        self.residual_heads = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(pooled_width),
                nn.Linear(pooled_width, width),
                nn.SiLU(),
                nn.Dropout(dropout),
                nn.Linear(width, 1),
            )
            for _ in range(self.TASKS)
        ])
        # Safety invariant: before fitting, the branch is exactly the anchor.
        for head in self.residual_heads:
            final = head[-1]
            assert isinstance(final, nn.Linear)
            nn.init.zeros_(final.weight)
            nn.init.zeros_(final.bias)

    @staticmethod
    def signed_differences(values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return_values = torch.zeros_like(values)
        return_values[:, :, 1:] = values[:, :, 1:] - values[:, :, :-1]
        second = torch.zeros_like(values)
        second[:, :, 2:] = return_values[:, :, 2:] - return_values[:, :, 1:-1]
        return return_values, second

    def _eat_tokens(
        self,
        temporal: torch.Tensor,
        spectral: torch.Tensor,
        view_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if temporal.ndim != 6 or spectral.ndim != 6:
            raise ValueError("EAT axes must be [batch,view,layer,stat,node,dim]")
        batch, views, layers, statistics, time_nodes, dimension = temporal.shape
        expected = (batch, views, layers, statistics)
        if spectral.shape[:4] != expected:
            raise ValueError("EAT temporal and spectral leading axes differ")
        if layers != self.eat_layers or statistics != 2:
            raise ValueError("EAT layer/statistic dimensions differ from config")
        if dimension != self.eat_dimension or spectral.shape[-1] != dimension:
            raise ValueError("EAT feature dimension differs from config")
        frequency_nodes = spectral.shape[-2]
        if min(time_nodes, frequency_nodes) <= 0:
            raise ValueError("EAT node axes cannot be empty")
        if (
            views > self.maximum_views
            or time_nodes > self.maximum_time_nodes
            or frequency_nodes > self.maximum_frequency_nodes
        ):
            raise ValueError("EAT axes exceed configured maxima")
        _require_bool_mask(view_mask, (batch, views), "eat_mask")
        _require_prefix_mask(view_mask, "eat_mask")
        if not torch.all(view_mask.any(dim=1)):
            raise ValueError("every item needs at least one valid EAT view")
        _require_finite(temporal, "EAT temporal features")
        _require_finite(spectral, "EAT spectral features")

        weights = self.eat_layer_logits.softmax(dim=-1)
        temporal = self.eat_projection(self.eat_norm(temporal.float()))
        spectral = self.eat_projection(self.eat_norm(spectral.float()))
        temporal = torch.einsum("bvlstd,ql->bvqstd", temporal, weights)
        spectral = torch.einsum("bvlstd,ql->bvqstd", spectral, weights)
        temporal = (
            temporal
            + self.task_embedding[None, None, :, None, None]
            + self.stream_embedding[0]
            + self.view_embedding[None, :views, None, None, None]
            + self.eat_stat_embedding[None, None, None, :2, None]
            + self.time_position[None, None, None, None, :time_nodes]
        ).permute(0, 2, 1, 3, 4, 5).reshape(
            batch, self.TASKS, views * 2 * time_nodes, self.width
        )
        spectral = (
            spectral
            + self.task_embedding[None, None, :, None, None]
            + self.stream_embedding[0]
            + self.view_embedding[None, :views, None, None, None]
            + self.eat_stat_embedding[None, None, None, 2:, None]
            + self.frequency_position[None, None, None, None, :frequency_nodes]
        ).permute(0, 2, 1, 3, 4, 5).reshape(
            batch, self.TASKS, views * 2 * frequency_nodes, self.width
        )
        time_mask = view_mask[:, :, None, None].expand(
            -1, -1, 2, time_nodes
        ).reshape(batch, -1)
        frequency_mask = view_mask[:, :, None, None].expand(
            -1, -1, 2, frequency_nodes
        ).reshape(batch, -1)
        values = torch.cat((temporal, spectral), dim=2)
        mask = torch.cat((time_mask, frequency_mask), dim=1)
        for block in self.eat_blocks:
            values = block(values)
        return values, mask

    def _spear_tokens(
        self,
        spear: torch.Tensor,
        spear_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if spear.ndim != 6:
            raise ValueError("SPEAR must be [batch,view,bin,layer,stat,dim]")
        batch, views, bins, layers, statistics, dimension = spear.shape
        if layers != self.spear_layers or statistics != self.spear_stats:
            raise ValueError("SPEAR layer/statistic dimensions differ from config")
        if dimension != self.spear_dimension:
            raise ValueError("SPEAR feature dimension differs from config")
        if views > self.maximum_views or bins > self.maximum_bins:
            raise ValueError("SPEAR axes exceed configured maxima")
        _require_bool_mask(spear_mask, (batch, views, bins), "spear_mask")
        _require_prefix_mask(spear_mask, "spear_mask")
        if not torch.all(spear_mask.any(dim=(1, 2))):
            raise ValueError("every item needs at least one valid SPEAR bin")
        _require_finite(spear, "SPEAR features")

        spear = spear.float()
        first, second = self.signed_differences(spear)
        spear = torch.cat((spear, first, second), dim=-1)
        spear = self.spear_projection(self.spear_norm(spear))
        weights = self.spear_layer_logits.softmax(dim=-1)
        spear = torch.einsum("bvmlsd,ql->bvqmsd", spear, weights)
        spear = (
            spear
            + self.task_embedding[None, None, :, None, None]
            + self.stream_embedding[1]
            + self.view_embedding[None, :views, None, None, None]
            + self.bin_embedding[None, None, None, :bins, None]
            + self.spear_stat_embedding[None, None, None, None, :statistics]
        ).permute(0, 2, 1, 3, 4, 5).reshape(
            batch, self.TASKS, views * bins * statistics, self.width
        )
        mask = spear_mask[:, :, :, None].expand(
            -1, -1, -1, statistics
        ).reshape(batch, -1)
        for block in self.spear_blocks:
            spear = block(spear)
        return spear, mask

    def _xlsr_tokens(
        self,
        embeddings: torch.Tensor | None,
        mask: torch.Tensor | None,
        batch: int,
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        if self.xlsr_dimension is None:
            if embeddings is not None or mask is not None:
                raise ValueError("XLS-R inputs were supplied to an XLS-R-disabled head")
            return None
        if embeddings is None or mask is None:
            raise ValueError("configured XLS-R stream requires embeddings and mask")
        if embeddings.ndim != 3:
            raise ValueError("XLS-R embeddings must be [batch,view,1920]")
        if embeddings.shape[0] != batch:
            raise ValueError("XLS-R and anchor batch axes differ")
        windows, dimension = embeddings.shape[1:]
        if dimension != XLSR_WINDOW_DIMENSION:
            raise ValueError(f"XLS-R embedding dimension must be {XLSR_WINDOW_DIMENSION}")
        if windows <= 0 or windows > self.maximum_xlsr_windows:
            raise ValueError("XLS-R window count exceeds configured maximum")
        _require_bool_mask(mask, (batch, windows), "xlsr_mask")
        _require_prefix_mask(mask, "xlsr_mask")
        if not torch.all(mask.any(dim=1)):
            raise ValueError("every item needs at least one valid XLS-R window")
        _require_finite(embeddings, "XLS-R embeddings")
        assert self.xlsr_norm is not None and self.xlsr_projection is not None
        assert self.xlsr_queries is not None and self.xlsr_value is not None
        values = self.xlsr_projection(self.xlsr_norm(embeddings.float()))
        values = (
            values[:, None]
            + self.task_embedding[None, :, None]
            + self.stream_embedding[2]
            + self.xlsr_position[None, None, :windows]
        )
        for block in self.xlsr_blocks:
            values = block(values)
        return values, mask

    def _pool(
        self,
        tokens: torch.Tensor,
        mask: torch.Tensor,
        queries: torch.Tensor,
        value_layer: nn.Module,
    ) -> torch.Tensor:
        batch, tasks, count, width = tokens.shape
        if mask.shape != (batch, count) or not torch.all(mask.any(dim=1)):
            raise ValueError("token mask is empty or incompatible")
        scores = torch.einsum("bqnd,qhd->bqhn", tokens, queries) / math.sqrt(width)
        scores = scores.masked_fill(~mask[:, None, None], torch.finfo(scores.dtype).min)
        attention = scores.softmax(dim=-1)
        head_width = width // self.heads
        values = value_layer(tokens).reshape(
            batch, tasks, count, self.heads, head_width
        ).permute(0, 1, 3, 2, 4)
        mean = torch.einsum("bqhn,bqhnk->bqhk", attention, values)
        second = torch.einsum("bqhn,bqhnk->bqhk", attention, values.square())
        std = (second - mean.square()).clamp_min(1e-5).sqrt()
        return torch.cat((mean.flatten(2), std.flatten(2)), dim=-1)

    def residual_features(
        self,
        temporal: torch.Tensor,
        spectral: torch.Tensor,
        eat_mask: torch.Tensor,
        spear: torch.Tensor,
        spear_mask: torch.Tensor,
        xlsr_embeddings: torch.Tensor | None = None,
        xlsr_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        eat, eat_token_mask = self._eat_tokens(temporal, spectral, eat_mask)
        spear_tokens, spear_token_mask = self._spear_tokens(spear, spear_mask)
        pooled = [
            self._pool(eat, eat_token_mask, self.eat_queries, self.eat_value),
            self._pool(
                spear_tokens, spear_token_mask,
                self.spear_queries, self.spear_value,
            ),
        ]
        xlsr = self._xlsr_tokens(xlsr_embeddings, xlsr_mask, len(temporal))
        if xlsr is None:
            pooled.append(eat.new_zeros((len(eat), self.TASKS, 2 * self.width)))
        else:
            assert self.xlsr_queries is not None and self.xlsr_value is not None
            pooled.append(self._pool(
                xlsr[0], xlsr[1], self.xlsr_queries, self.xlsr_value
            ))
        features = torch.cat(pooled, dim=-1)
        if self.task_interaction_in is not None:
            assert self.task_interaction_out is not None
            context = self.task_interaction_in(features)
            for block in self.task_interaction_blocks:
                context = block(context)
            # The adapter must not turn channel-consistency losses into a
            # latent-norm shortcut. A unit-bounded update preserves the scale
            # of the independently pooled stream evidence.
            features = features + torch.tanh(self.task_interaction_out(context))
        return features

    def _raw_residuals_from_features(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim != 3 or features.shape[1:] != (
            self.TASKS, 6 * self.width
        ):
            raise ValueError("residual features must have shape [batch,3,6*width]")
        return torch.cat([
            head(features[:, task]).unsqueeze(-1)
            for task, head in enumerate(self.residual_heads)
        ], dim=-1).squeeze(-2)

    def raw_residuals(
        self,
        temporal: torch.Tensor,
        spectral: torch.Tensor,
        eat_mask: torch.Tensor,
        spear: torch.Tensor,
        spear_mask: torch.Tensor,
        xlsr_embeddings: torch.Tensor | None = None,
        xlsr_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        features = self.residual_features(
            temporal, spectral, eat_mask, spear, spear_mask,
            xlsr_embeddings, xlsr_mask,
        )
        return self._raw_residuals_from_features(features)

    def bounded_residuals(self, raw_residuals: torch.Tensor) -> torch.Tensor:
        if raw_residuals.ndim != 2 or raw_residuals.shape[1] != self.TASKS:
            raise ValueError("raw residuals must have shape [batch,3]")
        _require_finite(raw_residuals, "raw residuals")
        return torch.tanh(raw_residuals) * self.residual_limits

    def apply_residuals(
        self,
        anchor_authenticity_logits: torch.Tensor,
        raw_residuals: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply residuals and return final and latent direct authenticity logits.

        File fusion is written in anchor-relative form to make a zero residual
        bit-exactly reproduce the supplied File anchor.  Algebraically it is a
        fixed ``0.7 * direct_file + 0.3 * noisy_or(voice,music)`` compositor;
        the compatible direct-file baseline is inferred from the three anchor
        logits before adding the learned direct-file residual.
        """
        if anchor_authenticity_logits.ndim != 2 or anchor_authenticity_logits.shape[1] != 3:
            raise ValueError("anchor authenticity logits must have shape [batch,3]")
        if raw_residuals.shape != anchor_authenticity_logits.shape:
            raise ValueError("anchor and residual logits must have identical shapes")
        _require_finite(anchor_authenticity_logits, "anchor authenticity logits")
        residual = self.bounded_residuals(raw_residuals)
        voice = anchor_authenticity_logits[:, 0] + residual[:, 0]
        music = anchor_authenticity_logits[:, 1] + residual[:, 1]
        anchor_or = noisy_or_logit(
            anchor_authenticity_logits[:, 0], anchor_authenticity_logits[:, 1]
        )
        component_or = noisy_or_logit(voice, music)
        direct_baseline = (
            anchor_authenticity_logits[:, 2]
            - self.FILE_COMPONENT_WEIGHT * anchor_or
        ) / self.FILE_DIRECT_WEIGHT
        direct_file = direct_baseline + residual[:, 2]
        # This is the numerically stable anchor-relative expansion of
        # 0.7 * direct_file + 0.3 * component_or.  In particular, multiplying
        # a reconstructed baseline cannot introduce identity-rounding error.
        file_logit = (
            anchor_authenticity_logits[:, 2]
            + self.FILE_DIRECT_WEIGHT * residual[:, 2]
            + self.FILE_COMPONENT_WEIGHT * (component_or - anchor_or)
        )
        final = torch.stack((voice, music, file_logit), dim=-1)
        direct = torch.stack((voice, music, direct_file), dim=-1)
        return final, direct

    def forward_with_latent(
        self,
        temporal: torch.Tensor,
        spectral: torch.Tensor,
        eat_mask: torch.Tensor,
        spear: torch.Tensor,
        spear_mask: torch.Tensor,
        anchor_authenticity_logits: torch.Tensor,
        anchor_presence_logits: torch.Tensor,
        xlsr_embeddings: torch.Tensor | None = None,
        xlsr_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        batch = len(temporal)
        if anchor_presence_logits.shape != (batch, 2):
            raise ValueError("anchor presence logits must have shape [batch,2]")
        if anchor_authenticity_logits.shape != (batch, 3):
            raise ValueError("anchor authenticity logits must have shape [batch,3]")
        _require_finite(anchor_presence_logits, "anchor presence logits")
        latent = self.residual_features(
            temporal, spectral, eat_mask, spear, spear_mask,
            xlsr_embeddings, xlsr_mask,
        )
        raw = self._raw_residuals_from_features(latent)
        authenticity, _ = self.apply_residuals(anchor_authenticity_logits, raw)
        # Do not clone or recalibrate: the anchor owns both presence outputs.
        return (
            authenticity, anchor_presence_logits,
            self.bounded_residuals(raw), latent,
        )

    def forward(
        self,
        temporal: torch.Tensor,
        spectral: torch.Tensor,
        eat_mask: torch.Tensor,
        spear: torch.Tensor,
        spear_mask: torch.Tensor,
        anchor_authenticity_logits: torch.Tensor,
        anchor_presence_logits: torch.Tensor,
        xlsr_embeddings: torch.Tensor | None = None,
        xlsr_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        output = self.forward_with_latent(
            temporal, spectral, eat_mask, spear, spear_mask,
            anchor_authenticity_logits, anchor_presence_logits,
            xlsr_embeddings, xlsr_mask,
        )
        return output[:3]


def anchor_residual_loss(
    authenticity_logits: torch.Tensor,
    fake_targets: torch.Tensor,
    presence_targets: torch.Tensor,
    sample_weight: torch.Tensor,
    residuals: torch.Tensor,
    task_weights: Sequence[float] = (0.2, 0.3, 0.5),
    residual_weight: float = 0.02,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Masked authenticity BCE with a conservative residual penalty."""
    if authenticity_logits.ndim != 2 or authenticity_logits.shape[1] != 3:
        raise ValueError("authenticity logits must have shape [batch,3]")
    if fake_targets.shape != authenticity_logits.shape:
        raise ValueError("fake targets must match authenticity logits")
    batch = len(authenticity_logits)
    if presence_targets.shape != (batch, 2):
        raise ValueError("presence targets must have shape [batch,2]")
    if residuals.shape != authenticity_logits.shape:
        raise ValueError("residuals must match authenticity logits")
    if sample_weight.shape != (batch,):
        raise ValueError("sample weights must have shape [batch]")
    weights = torch.as_tensor(tuple(task_weights), dtype=authenticity_logits.dtype,
                              device=authenticity_logits.device)
    if weights.shape != (3,) or not torch.isfinite(weights).all():
        raise ValueError("task_weights must contain three finite values")
    if torch.any(weights < 0) or weights.sum() <= 0:
        raise ValueError("task weights must be nonnegative with positive sum")
    if residual_weight < 0:
        raise ValueError("residual_weight cannot be negative")
    for values, name in (
        (authenticity_logits, "authenticity logits"),
        (fake_targets, "fake targets"),
        (presence_targets, "presence targets"),
        (sample_weight, "sample weights"),
        (residuals, "residuals"),
    ):
        _require_finite(values, name)
    if torch.any((fake_targets < 0) | (fake_targets > 1)):
        raise ValueError("fake targets must lie in [0,1]")
    if torch.any((presence_targets < 0) | (presence_targets > 1)):
        raise ValueError("presence targets must lie in [0,1]")
    if torch.any(sample_weight < 0) or sample_weight.sum() <= 0:
        raise ValueError("sample weights must be nonnegative with positive sum")

    masks = (
        presence_targets[:, 0] > 0.5,
        presence_targets[:, 1] > 0.5,
        torch.ones(batch, dtype=torch.bool, device=authenticity_logits.device),
    )
    losses = []
    for task, mask in enumerate(masks):
        if not mask.any():
            losses.append(authenticity_logits[:, task].sum() * 0)
            continue
        raw = F.binary_cross_entropy_with_logits(
            authenticity_logits[mask, task], fake_targets[mask, task],
            reduction="none",
        )
        selected_weight = sample_weight[mask]
        losses.append(
            (raw * selected_weight).sum() / selected_weight.sum().clamp_min(1e-8)
        )
    authenticity = (torch.stack(losses) * weights).sum() / weights.sum()
    per_item_residual = residuals.square().mean(dim=1)
    penalty = (
        per_item_residual * sample_weight
    ).sum() / sample_weight.sum().clamp_min(1e-8)
    total = authenticity + residual_weight * penalty
    return total, {
        "voice": losses[0].detach(),
        "music": losses[1].detach(),
        "file": losses[2].detach(),
        "residual": penalty.detach(),
    }


def _index_groups(
    groups: torch.Tensor | None,
    width: int,
    batch: int,
    device: torch.device,
    name: str,
) -> torch.Tensor:
    if groups is None:
        return torch.empty((0, width), dtype=torch.long, device=device)
    if groups.ndim != 2 or groups.shape[1] != width:
        raise ValueError(f"{name} must have shape [groups,{width}]")
    if groups.dtype not in (
        torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8,
    ):
        raise TypeError(f"{name} must contain integer indices")
    groups = groups.to(device=device, dtype=torch.long)
    if groups.numel() and (groups.min() < 0 or groups.max() >= batch):
        raise ValueError(f"{name} contains an out-of-range index")
    if groups.numel():
        ordered = groups.sort(dim=1).values
        if torch.any(ordered[:, 1:] == ordered[:, :-1]):
            raise ValueError(f"{name} cannot repeat a row within one group")
    return groups


def quartet_ranking_losses(
    authenticity_logits: torch.Tensor,
    fake_targets: torch.Tensor,
    presence_targets: torch.Tensor,
    quartets: torch.Tensor | None,
    margin: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return Voice/Music/File rank losses for canonical RR/RF/FR/FF rows."""
    if margin < 0:
        raise ValueError("ranking margin cannot be negative")
    groups = _index_groups(
        quartets, 4, len(authenticity_logits), authenticity_logits.device,
        "quartets",
    )
    zero = authenticity_logits.sum() * 0
    if not len(groups):
        return zero, zero, zero
    expected = fake_targets.new_tensor(
        ((0, 0), (0, 1), (1, 0), (1, 1))
    )
    if not torch.all(presence_targets[groups] > 0.5):
        raise ValueError("quartet ranking requires both components to be present")
    if not torch.equal(fake_targets[groups, :2], expected[None].expand(len(groups), -1, -1)):
        raise ValueError("quartets must be ordered RR, RF, FR, FF")

    rr, rf, fr, ff = groups.unbind(dim=1)

    def rank(positive: torch.Tensor, negative: torch.Tensor) -> torch.Tensor:
        return F.softplus(margin - (positive - negative)).mean()

    voice = 0.5 * (
        rank(authenticity_logits[fr, 0], authenticity_logits[rr, 0])
        + rank(authenticity_logits[ff, 0], authenticity_logits[rf, 0])
    )
    music = 0.5 * (
        rank(authenticity_logits[rf, 1], authenticity_logits[rr, 1])
        + rank(authenticity_logits[ff, 1], authenticity_logits[fr, 1])
    )
    file = (
        rank(authenticity_logits[rf, 2], authenticity_logits[rr, 2])
        + rank(authenticity_logits[fr, 2], authenticity_logits[rr, 2])
        + rank(authenticity_logits[ff, 2], authenticity_logits[rr, 2])
    ) / 3
    return voice, music, file


def quartet_component_invariance_losses(
    authenticity_logits: torch.Tensor,
    latent: torch.Tensor | None,
    fake_targets: torch.Tensor,
    presence_targets: torch.Tensor,
    quartets: torch.Tensor | None,
    task_weights: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Keep each component score invariant to the other component's origin.

    For an RR/RF/FR/FF counterfactual quartet the Voice evidence should agree
    between RR/RF and FR/FF, while Music evidence should agree between RR/FR
    and RF/FF.  This directly targets cross-component leakage in the two hard
    mixed cases without forcing the File task (whose label genuinely changes)
    to be invariant.
    """
    groups = _index_groups(
        quartets, 4, len(authenticity_logits), authenticity_logits.device,
        "quartets",
    )
    zero = authenticity_logits.sum() * 0
    if not len(groups):
        return zero, zero
    expected = fake_targets.new_tensor(
        ((0, 0), (0, 1), (1, 0), (1, 1))
    )
    if not torch.all(presence_targets[groups] > 0.5):
        raise ValueError("quartet invariance requires both components present")
    if not torch.equal(
        fake_targets[groups, :2], expected[None].expand(len(groups), -1, -1)
    ):
        raise ValueError("quartets must be ordered RR, RF, FR, FF")
    if task_weights.shape != (3,) or torch.any(task_weights[:2] < 0):
        raise ValueError("task_weights must contain nonnegative Voice/Music weights")
    component_mass = task_weights[:2].sum()
    if component_mass <= 0:
        return zero, zero
    rr, rf, fr, ff = groups.unbind(dim=1)

    def paired(values: torch.Tensor, first: int, pairs) -> torch.Tensor:
        return 0.5 * sum(
            F.smooth_l1_loss(values[left, first], values[right, first])
            for left, right in pairs
        )

    voice_pairs = ((rr, rf), (fr, ff))
    music_pairs = ((rr, fr), (rf, ff))
    voice_logit = paired(authenticity_logits, 0, voice_pairs)
    music_logit = paired(authenticity_logits, 1, music_pairs)
    logit_loss = (
        task_weights[0] * voice_logit + task_weights[1] * music_logit
    ) / component_mass

    if latent is None:
        raise ValueError("latent values are required for quartet invariance")
    if latent.ndim != 3 or latent.shape[:2] != (len(authenticity_logits), 3):
        raise ValueError("latent values must have shape [batch,3,dimension]")
    _require_finite(latent, "latent values")
    voice_latent = paired(latent, 0, voice_pairs)
    music_latent = paired(latent, 1, music_pairs)
    latent_loss = (
        task_weights[0] * voice_latent + task_weights[1] * music_latent
    ) / component_mass
    return logit_loss, latent_loss


def batch_auc_ranking_loss(
    authenticity_logits: torch.Tensor,
    fake_targets: torch.Tensor,
    presence_targets: torch.Tensor,
    sample_weight: torch.Tensor,
    task_weights: torch.Tensor,
    margin: float = 0.0,
) -> torch.Tensor:
    """Pairwise logistic ranking objective aligned with the three EER tasks."""
    if margin < 0:
        raise ValueError("AUC ranking margin cannot be negative")
    batch = len(authenticity_logits)
    if authenticity_logits.shape != (batch, 3):
        raise ValueError("authenticity logits must have shape [batch,3]")
    if fake_targets.shape != (batch, 3) or presence_targets.shape != (batch, 2):
        raise ValueError("AUC ranking targets have incompatible shapes")
    if sample_weight.shape != (batch,) or task_weights.shape != (3,):
        raise ValueError("AUC ranking weights have incompatible shapes")
    masks = (
        presence_targets[:, 0] > 0.5,
        presence_targets[:, 1] > 0.5,
        torch.ones(batch, dtype=torch.bool, device=authenticity_logits.device),
    )
    losses, active_weights = [], []
    for task, present in enumerate(masks):
        positive = present & (fake_targets[:, task] > 0.5)
        negative = present & ~positive
        if not positive.any() or not negative.any():
            continue
        differences = (
            authenticity_logits[positive, task, None]
            - authenticity_logits[negative, task][None, :]
        )
        pair_weight = (
            sample_weight[positive, None] * sample_weight[negative][None, :]
        )
        loss = F.softplus(margin - differences)
        losses.append((loss * pair_weight).sum() / pair_weight.sum().clamp_min(1e-8))
        active_weights.append(task_weights[task])
    if not losses:
        return authenticity_logits.sum() * 0
    active = torch.stack(active_weights)
    return (torch.stack(losses) * active).sum() / active.sum().clamp_min(1e-8)


def channel_consistency_losses(
    authenticity_logits: torch.Tensor,
    latent: torch.Tensor | None,
    fake_targets: torch.Tensor,
    presence_targets: torch.Tensor,
    pairs: torch.Tensor | None,
    task_weights: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Penalize clean/codec drift for metadata-confirmed identical mixtures."""
    pair_index = _index_groups(
        pairs, 2, len(authenticity_logits), authenticity_logits.device,
        "channel pairs",
    )
    zero = authenticity_logits.sum() * 0
    if not len(pair_index):
        return zero, zero
    clean, codec = pair_index.unbind(dim=1)
    if not torch.equal(presence_targets[clean], presence_targets[codec]):
        raise ValueError("channel pair presence targets differ")
    if not torch.equal(fake_targets[clean], fake_targets[codec]):
        raise ValueError("channel pair authenticity targets differ")
    per_task = F.smooth_l1_loss(
        authenticity_logits[codec], authenticity_logits[clean].detach(),
        reduction="none",
    ).mean(dim=0)
    logit_loss = (per_task * task_weights).sum() / task_weights.sum()
    if latent is None:
        raise ValueError("latent values are required when channel pairs are present")
    if latent.ndim != 3 or latent.shape[:2] != (len(authenticity_logits), 3):
        raise ValueError("latent values must have shape [batch,3,dimension]")
    _require_finite(latent, "latent values")
    latent_per_task = F.smooth_l1_loss(
        latent[codec], latent[clean].detach(), reduction="none"
    ).mean(dim=(0, 2))
    latent_loss = (latent_per_task * task_weights).sum() / task_weights.sum()
    return logit_loss, latent_loss


def initial_model_consistency_loss(
    logits: torch.Tensor,
    reference_logits: torch.Tensor,
    presence: torch.Tensor,
    task_weights: Sequence[float],
) -> torch.Tensor:
    """Train-only retention of a frozen warm start, with absent tasks masked."""
    if logits.ndim != 2 or logits.shape[1] != 3 or reference_logits.shape != logits.shape:
        raise ValueError("student/reference logits must have matching [batch,3] shapes")
    if presence.shape != (len(logits), 2):
        raise ValueError("presence must have shape [batch,2]")
    weights = logits.new_tensor(tuple(task_weights))
    if weights.shape != (3,) or not torch.isfinite(weights).all() or (weights < 0).any() or weights.sum() <= 0:
        raise ValueError("task weights must be finite, nonnegative, and have positive mass")
    _require_finite(logits, "student logits")
    _require_finite(reference_logits, "reference logits")
    mask = torch.cat((presence > .5, torch.ones_like(presence[:, :1], dtype=torch.bool)), dim=1)
    active = mask.to(logits.dtype) * weights
    errors = F.smooth_l1_loss(logits, reference_logits.detach(), reduction="none")
    return (errors * active).sum() / active.sum().clamp_min(1e-8)


def group_dro_loss(
    authenticity_logits: torch.Tensor,
    fake_targets: torch.Tensor,
    presence_targets: torch.Tensor,
    sample_weight: torch.Tensor,
    environment_ids: torch.Tensor | None,
    task_weights: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    """Soft worst-group authenticity loss over available environment IDs."""
    if temperature < 0:
        raise ValueError("group-DRO temperature cannot be negative")
    zero = authenticity_logits.sum() * 0
    if environment_ids is None:
        return zero
    if environment_ids.shape != (len(authenticity_logits),):
        raise ValueError("environment IDs must have shape [batch]")
    if environment_ids.dtype not in (
        torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8,
    ):
        raise TypeError("environment IDs must be integers")
    environment_ids = environment_ids.to(authenticity_logits.device)
    group_losses = []
    component_masks = (
        presence_targets[:, 0] > 0.5,
        presence_targets[:, 1] > 0.5,
        torch.ones(
            len(authenticity_logits), dtype=torch.bool,
            device=authenticity_logits.device,
        ),
    )
    for environment in torch.unique(environment_ids[environment_ids >= 0]):
        selected_group = environment_ids == environment
        task_losses, active_weights = [], []
        for task, component_mask in enumerate(component_masks):
            # Specialist training may disable tasks. An environment with no
            # active target (e.g. speech-only for Music) must not yield 0/0.
            if task_weights[task] <= 0:
                continue
            selected = selected_group & component_mask
            if not selected.any() or sample_weight[selected].sum() <= 0:
                continue
            raw = F.binary_cross_entropy_with_logits(
                authenticity_logits[selected, task], fake_targets[selected, task],
                reduction="none",
            )
            weight = sample_weight[selected]
            task_losses.append((raw * weight).sum() / weight.sum())
            active_weights.append(task_weights[task])
        if task_losses:
            active = torch.stack(active_weights)
            group_losses.append(
                (torch.stack(task_losses) * active).sum() / active.sum()
            )
    if not group_losses:
        return zero
    values = torch.stack(group_losses)
    if temperature == 0:
        return values.max()
    scale = authenticity_logits.new_tensor(float(temperature))
    return scale * (
        torch.logsumexp(values / scale, dim=0)
        - math.log(len(values))
    )


def robust_anchor_residual_loss(
    authenticity_logits: torch.Tensor,
    fake_targets: torch.Tensor,
    presence_targets: torch.Tensor,
    sample_weight: torch.Tensor,
    residuals: torch.Tensor,
    latent: torch.Tensor | None = None,
    quartets: torch.Tensor | None = None,
    channel_pairs: torch.Tensor | None = None,
    environment_ids: torch.Tensor | None = None,
    task_weights: Sequence[float] = (0.2, 0.3, 0.5),
    residual_weight: float = 0.02,
    ranking_weight: float = 0.10,
    ranking_margin: float = 0.0,
    quartet_invariance_weight: float = 0.0,
    quartet_latent_invariance_weight: float = 0.0,
    auc_ranking_weight: float = 0.0,
    auc_ranking_margin: float = 0.0,
    logit_consistency_weight: float = 0.05,
    latent_consistency_weight: float = 0.01,
    group_dro_weight: float = 0.20,
    group_dro_temperature: float = 0.10,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Full robustness objective for the offline anchor-residual experiment."""
    auxiliary_weights = (
        ranking_weight, quartet_invariance_weight,
        quartet_latent_invariance_weight, auc_ranking_weight,
        auc_ranking_margin, logit_consistency_weight,
        latent_consistency_weight, group_dro_weight,
    )
    if any(weight < 0 for weight in auxiliary_weights):
        raise ValueError("auxiliary loss weights cannot be negative")
    base, terms = anchor_residual_loss(
        authenticity_logits, fake_targets, presence_targets, sample_weight,
        residuals, task_weights=task_weights, residual_weight=residual_weight,
    )
    weights = torch.as_tensor(
        tuple(task_weights), dtype=authenticity_logits.dtype,
        device=authenticity_logits.device,
    )
    voice_rank, music_rank, file_rank = quartet_ranking_losses(
        authenticity_logits, fake_targets, presence_targets,
        quartets, margin=ranking_margin,
    )
    ranking = (
        weights[0] * voice_rank
        + weights[1] * music_rank
        + weights[2] * file_rank
    ) / weights.sum()
    quartet_invariance, quartet_latent_invariance = (
        quartet_component_invariance_losses(
            authenticity_logits, latent, fake_targets, presence_targets,
            quartets, weights,
        )
    )
    auc_ranking = batch_auc_ranking_loss(
        authenticity_logits, fake_targets, presence_targets, sample_weight,
        weights, margin=auc_ranking_margin,
    )
    logit_consistency, latent_consistency = channel_consistency_losses(
        authenticity_logits, latent, fake_targets, presence_targets,
        channel_pairs, weights,
    )
    group_dro = group_dro_loss(
        authenticity_logits, fake_targets, presence_targets, sample_weight,
        environment_ids, weights, group_dro_temperature,
    )
    total = (
        base
        + ranking_weight * ranking
        + quartet_invariance_weight * quartet_invariance
        + quartet_latent_invariance_weight * quartet_latent_invariance
        + auc_ranking_weight * auc_ranking
        + logit_consistency_weight * logit_consistency
        + latent_consistency_weight * latent_consistency
        + group_dro_weight * group_dro
    )
    terms.update({
        "rank_voice": voice_rank.detach(),
        "rank_music": music_rank.detach(),
        "rank_file": file_rank.detach(),
        "ranking": ranking.detach(),
        "quartet_invariance": quartet_invariance.detach(),
        "quartet_latent_invariance": quartet_latent_invariance.detach(),
        "auc_ranking": auc_ranking.detach(),
        "logit_consistency": logit_consistency.detach(),
        "latent_consistency": latent_consistency.detach(),
        "group_dro": group_dro.detach(),
    })
    return total, terms
