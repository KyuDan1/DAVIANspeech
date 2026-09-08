"""Offline X-Codec mini token extraction for music-generation forensics."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch


SAMPLE_RATE = 16_000
CROP_SAMPLES = 160_000
SELECTED_CODEBOOKS = (0, 1, 10, 11)


def center_crop_or_pad(audio: np.ndarray, samples: int = CROP_SAMPLES) -> np.ndarray:
    """Return one deterministic, fixed-length crop without modifying the signal."""
    audio = np.asarray(audio, dtype=np.float32)
    if len(audio) >= samples:
        start = (len(audio) - samples) // 2
        return np.ascontiguousarray(audio[start:start + samples])
    left = (samples - len(audio)) // 2
    return np.pad(audio, (left, samples - len(audio) - left))


def load_xcodec(model_root: Path, device: torch.device, half: bool = True):
    """Load the stripped public X-Codec mini checkpoint entirely offline."""
    model_root = Path(model_root).resolve()
    for directory in (model_root, model_root / "descriptaudiocodec"):
        if str(directory) not in sys.path:
            sys.path.insert(0, str(directory))

    from models.soundstream_hubert_new import SoundStream

    model = SoundStream(
        n_filters=32,
        D=256,
        target_bandwidths=[0.5, 1, 1.5, 2, 4, 6],
        ratios=[8, 5, 4, 2],
        sample_rate=SAMPLE_RATE,
        bins=1024,
    )
    state_path = model_root / "final_ckpt" / "codec_model_state.pt"
    state = torch.load(state_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    model.eval().requires_grad_(False).to(device)
    if half and device.type == "cuda":
        model.half()
    return model


@torch.inference_mode()
def encode_selected(model, audios: list[np.ndarray], device: torch.device) -> np.ndarray:
    """Encode q0/q1 and q10/q11, the CoMoE paper's mini-codebook streams."""
    batch = np.stack([center_crop_or_pad(audio) for audio in audios])[:, None, :]
    dtype = next(model.parameters()).dtype
    waveform = torch.from_numpy(batch).to(device=device, dtype=dtype)
    codes = model.encode(waveform, target_bw=6)
    codes = codes[list(SELECTED_CODEBOOKS)].permute(1, 0, 2)
    return codes.cpu().numpy().astype(np.int16)
