"""Offline, strictly file-local inference for the v60 File specialist."""
from __future__ import annotations

import gc
from pathlib import Path
import time

import numpy as np
import torch

try:
    from .full_coverage_wpt import coverage_windows, forward_bags
    from .pipeline import load_audio
    from .wpt_spectra import WPTSpectraMultitask
    from .wpt_spectra_inference import _load_spectra
except ImportError:
    from full_coverage_wpt import coverage_windows, forward_bags
    from pipeline import load_audio
    from wpt_spectra import WPTSpectraMultitask
    from wpt_spectra_inference import _load_spectra


def load_full_coverage_model(model_dir, checkpoint_path, device, *, allow_smoke=False):
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    if checkpoint.get('model_type') != 'full_coverage_wpt_file_v60':
        raise ValueError('not a full-coverage WPT File checkpoint')
    if checkpoint.get('smoke', True) and not allow_smoke:
        raise ValueError('smoke checkpoints are not eligible for real inference')
    config = checkpoint['config']
    model = WPTSpectraMultitask(_load_spectra(Path(model_dir), device),
                               temperature=config['temperature']).to(device)
    model.load_trainable_state_dict(checkpoint['state'])
    return model.eval(), config


@torch.inference_mode()
def predict_one_audio(model, audio, config, device):
    windows, starts = coverage_windows(audio, config['window'])
    values = torch.from_numpy(windows[None]).to(device)
    mask = torch.ones(values.shape[:2], dtype=torch.bool, device=device)
    with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == 'cuda'):
        logits = forward_bags(model, values, mask, config['temperature'], config['window_chunk_size'])
    score = float(logits[0, 2].sigmoid())
    if not np.isfinite(score) or not 0 <= score <= 1:
        raise ValueError('invalid File probability')
    return score, len(starts)


@torch.inference_mode()
def predict_files(audio_paths, model_dir, checkpoint_path, device='cuda', *, allow_smoke=False):
    target = torch.device(device)
    model, config = load_full_coverage_model(model_dir, checkpoint_path, target, allow_smoke=allow_smoke)
    scores, timings = [], []
    try:
        for path in audio_paths:
            started = time.perf_counter()
            audio = load_audio(Path(path))
            score, windows = predict_one_audio(model, audio, config, target)
            scores.append(score)
            timings.append(dict(path=str(path), duration_seconds=len(audio) / 16000,
                                windows=windows, seconds=time.perf_counter() - started))
    finally:
        del model
        gc.collect()
        if target.type == 'cuda':
            torch.cuda.empty_cache()
    return np.asarray(scores, dtype=np.float64), timings
