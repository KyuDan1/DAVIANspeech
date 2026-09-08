"""File-independent inference for the low-band Music research model."""
from pathlib import Path

import numpy as np
import torch

from .full_coverage_wpt import coverage_windows
from .lowband_music_v67 import LowbandMusic


def load_model(checkpoint_path: Path, device: torch.device, *, allow_smoke=False):
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    if checkpoint.get('model_type') != 'lowband_music_v67':
        raise ValueError('not a lowband Music checkpoint')
    if checkpoint.get('smoke', True) and not allow_smoke:
        raise ValueError('smoke checkpoint is not eligible for inference deployment')
    phase = checkpoint['phase']
    if phase is not checkpoint['config']['variants'][checkpoint['variant']]['phase']:
        raise ValueError('phase/config mismatch')
    model = LowbandMusic(phase=phase)
    model.load_state_dict(checkpoint['state'], strict=True)
    return model.to(device).eval(), checkpoint['config']


@torch.inference_mode()
def predict_one_audio(model, audio: np.ndarray, config: dict, device: torch.device):
    windows, _ = coverage_windows(audio, config['window'])
    windows = torch.from_numpy(windows)[None].to(device)
    mask = torch.ones(windows.shape[:2], dtype=torch.bool, device=device)
    with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == 'cuda'):
        output = model(windows, mask, config['temperature'], config['window_chunk_size'])
    value = float(output.float().sigmoid().item())
    if not np.isfinite(value):
        raise ValueError('non-finite Music probability')
    return value
