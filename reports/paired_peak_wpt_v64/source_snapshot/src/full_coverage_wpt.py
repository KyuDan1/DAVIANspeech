"""Full-file MIL windows and content-matched real codec augmentation.

No separation, file-to-file statistics, or per-window fake label invention.
"""
from __future__ import annotations

import math
from pathlib import Path
import shutil
import sys
import numpy as np
import torch

try:
    from .telephone_channel import apply_channel
except ImportError:
    from telephone_channel import apply_channel


def resolve_ffmpeg() -> Path:
    candidate = shutil.which('ffmpeg')
    path = Path(candidate) if candidate else Path(sys.executable).parent / 'ffmpeg'
    if not path.is_file():
        raise FileNotFoundError('ffmpeg is missing from PATH and the Python environment')
    return path


def coverage_windows(audio: np.ndarray, window: int = 64600) -> tuple[np.ndarray, np.ndarray]:
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim != 1 or not len(audio) or not np.isfinite(audio).all() or window <= 0:
        raise ValueError('expected nonempty finite mono audio and a positive window')
    if len(audio) < window:
        return np.tile(audio, math.ceil(window / len(audio)))[:window][None], np.array([0])
    count = math.ceil(len(audio) / window)
    starts = np.rint(np.linspace(0, len(audio) - window, count)).astype(np.int64)
    return np.stack([audio[start:start + window] for start in starts]), starts


def codec_pair(windows: np.ndarray, variant: str, ffmpeg: Path, key: int) -> np.ndarray:
    """Apply a real channel to precisely the SAME already-selected segments."""
    output = []
    for index, window in enumerate(windows):
        changed = apply_channel(window, variant, ffmpeg=ffmpeg, key=key + index)
        # Codec frame padding may add a tail; no independent random crop.
        changed = np.pad(changed, (0, max(0, len(window) - len(changed))))[:len(window)]
        if not np.isfinite(changed).all():
            raise ValueError('codec returned non-finite audio')
        output.append(changed)
    return np.stack(output).astype(np.float32)


def masked_lme(logits: torch.Tensor, mask: torch.Tensor, temperature: float = 2.) -> torch.Tensor:
    if logits.ndim != 3 or mask.shape != logits.shape[:2] or mask.dtype != torch.bool:
        raise ValueError('logits [batch,views,tasks] and bool view mask required')
    if temperature <= 0 or not mask.any(dim=1).all():
        raise ValueError('positive temperature and nonempty bags required')
    masked = logits.float().masked_fill(~mask[..., None], -torch.inf)
    return (torch.logsumexp(temperature * masked, dim=1)
            - mask.sum(dim=1).float().log()[:, None]) / temperature


def forward_bags(model, windows: torch.Tensor, mask: torch.Tensor,
                 temperature: float = 2., chunk_size: int = 16, *,
                 peak_normalize: bool = False) -> torch.Tensor:
    """Padding never enters the encoder/backend; pooling stays file-local."""
    if windows.ndim != 3 or windows.shape[:2] != mask.shape or chunk_size <= 0:
        raise ValueError('invalid windows/mask/chunk size')
    if not isinstance(peak_normalize, bool):
        raise ValueError('peak_normalize must be a boolean')
    valid = windows[mask]
    outputs = []
    for start in range(0, len(valid), chunk_size):
        chunk = valid[start:start + chunk_size]
        emphasized = chunk.clone()
        emphasized[:, 1:] = chunk[:, 1:] - .97 * chunk[:, :-1]
        if peak_normalize:
            emphasized = emphasized / (emphasized.abs().amax(dim=-1, keepdim=True) + 1e-8)
        outputs.append(model.forward_windows(emphasized[:, None]).squeeze(1))
    flat = torch.cat(outputs)
    padded = flat.new_zeros((*mask.shape, flat.shape[-1]))
    padded[mask] = flat
    return masked_lme(padded, mask, temperature)
