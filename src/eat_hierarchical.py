"""Hierarchical EAT statistics for separation-free audio forensics.

The bundled EAT checkpoint normally exposes only its final hidden layer.  For
deepfake detection this is unnecessarily restrictive: intermediate layers
retain low-level spectral texture while later layers become more semantic.
This module mirrors the checkpoint's encoder forward pass and summarizes every
transformer layer without retaining the full activation stack in memory.
"""

from __future__ import annotations

import numpy as np
import torch
from torch.nn import functional as F

try:  # package import in tests; flat import in the offline submission
    from .dual_domain_stats import sequence_statistics
except ImportError:  # pragma: no cover - exercised by script.py
    from dual_domain_stats import sequence_statistics


STATISTIC_COUNT = 5


def gaussian_projection(
    input_width: int = 768,
    output_width: int = 128,
    seed: int = 20260904,
) -> torch.Tensor:
    """Return a deterministic label-free Johnson--Lindenstrauss projection."""
    if input_width <= 0 or output_width <= 0:
        raise ValueError("projection widths must be positive")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    return (
        torch.randn(input_width, output_width, generator=generator)
        / np.sqrt(input_width)
    )


@torch.inference_mode()
def hierarchical_statistics(
    model,
    features: torch.Tensor,
    projection: torch.Tensor | None = None,
) -> torch.Tensor:
    """Return ``[batch, layers, 5, width]`` EAT statistics.

    The five summaries are mean, standard deviation, mean absolute temporal
    delta, Teager--Kaiser energy, and the layer's CLS token.  Patch-token
    summaries and CLS are kept separate so a learned back-end can choose
    acoustic texture or global content without another encoder pass.
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

    batch = features.shape[0]
    values = core.local_encoder(features)
    positional = getattr(core, "fixed_positional_encoder", None)
    if positional is not None:
        values = values + positional(values, None)[:, :values.size(1), :]
    values = torch.cat((core.extra_tokens.expand(batch, -1, -1), values), dim=1)
    values = core.pos_drop(core.pre_norm(values))

    matrix = None
    if projection is not None:
        matrix = projection.to(device=values.device, dtype=values.dtype)
        if matrix.ndim != 2 or matrix.shape[0] != values.shape[-1]:
            raise ValueError("projection has incompatible shape")

    layers = []
    for block in core.blocks:
        values, _ = block(values)
        patch_statistics = sequence_statistics(values[:, 1:])
        summaries = torch.cat((patch_statistics, values[:, :1].float()), dim=-2)
        if matrix is not None:
            summaries = F.layer_norm(
                summaries, (summaries.shape[-1],)
            ) @ matrix.float()
        layers.append(summaries)
    if not layers:
        raise ValueError("EAT core has no transformer blocks")
    result = torch.stack(layers, dim=1)
    if result.shape[-2] != STATISTIC_COUNT:
        raise RuntimeError("unexpected hierarchical statistic count")
    return result
