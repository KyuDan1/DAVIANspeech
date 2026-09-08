"""Inference and conservative residual fusion for dedicated Music heads."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import torch

try:
    from .channel_invariant_music_head import ChannelInvariantMusicHead
except ImportError:  # pragma: no cover - flat offline package
    from channel_invariant_music_head import ChannelInvariantMusicHead


def _logit(probability: np.ndarray) -> np.ndarray:
    probability = np.clip(np.asarray(probability, dtype=np.float64), 1e-6, 1 - 1e-6)
    return np.log(probability) - np.log1p(-probability)


def residual_music_fusion(
    anchor_probability: np.ndarray,
    expert_probability: np.ndarray,
    weight: float,
) -> np.ndarray:
    """Fuse an expert in log-odds space without changing non-Music outputs."""
    if not 0 <= weight <= 1:
        raise ValueError("weight must be between zero and one")
    anchor = _logit(anchor_probability)
    expert = _logit(expert_probability)
    fused = (1.0 - weight) * anchor + weight * expert
    return (1.0 / (1.0 + np.exp(-np.clip(fused, -40, 40)))).astype(np.float32)


def load_music_heads(
    checkpoint_paths: Iterable[str | Path], device: torch.device,
) -> list[tuple[ChannelInvariantMusicHead, dict[str, torch.Tensor]]]:
    """Load independently selected heads and their own training normalization."""
    loaded = []
    for path in checkpoint_paths:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        if checkpoint.get("model_type") != "channel_invariant_music":
            raise ValueError(f"not a dedicated Music checkpoint: {path}")
        model = ChannelInvariantMusicHead(**checkpoint["config"])
        model.load_state_dict(checkpoint["model"])
        model.to(device).eval()
        normalization = {
            key: torch.as_tensor(value, device=device, dtype=torch.float32)[None, None]
            for key, value in checkpoint["normalization"].items()
        }
        loaded.append((model, normalization))
    if not loaded:
        raise ValueError("at least one checkpoint is required")
    return loaded


@torch.inference_mode()
def predict_music_from_statistics(
    heads: list[tuple[ChannelInvariantMusicHead, dict[str, torch.Tensor]]],
    eat: np.ndarray,
    spear: np.ndarray,
    eat_mask: np.ndarray,
    spear_mask: np.ndarray,
    *,
    device: torch.device,
    batch_size: int = 96,
) -> np.ndarray:
    """Predict Music probability; multi-seed heads are averaged in logit space."""
    member_logits = []
    for model, norm in heads:
        chunks = []
        for start in range(0, len(eat), batch_size):
            stop = start + batch_size
            eat_batch = torch.as_tensor(eat[start:stop], device=device, dtype=torch.float32)
            spear_batch = torch.as_tensor(spear[start:stop], device=device, dtype=torch.float32)
            eat_batch = ((eat_batch - norm["eat_mean"]) / norm["eat_std"]).clamp_(-8, 8)
            spear_batch = ((spear_batch - norm["spear_mean"]) / norm["spear_std"]).clamp_(-8, 8)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=device.type == "cuda",
            ):
                logits, _ = model(
                    eat_batch,
                    spear_batch,
                    torch.as_tensor(eat_mask[start:stop], device=device),
                    torch.as_tensor(spear_mask[start:stop], device=device),
                )
            chunks.append(logits.float().cpu().numpy())
        member_logits.append(np.concatenate(chunks))
    logits = np.mean(np.stack(member_logits), axis=0)
    return (1.0 / (1.0 + np.exp(-np.clip(logits, -40, 40)))).astype(np.float32)
