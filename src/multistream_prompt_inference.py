"""Offline two-view multistream File expert for the v41 fixed MoE.

The model sees only the original mixture.  Its File logit is blended inside
the existing WPT branch, so Voice, Music, and both presence predictions remain
bit-identical to v41.
"""

from __future__ import annotations

import csv
import gc
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

try:  # package imports in tests; flat imports in the submission archive
    from .multistream_prompt_spectra import MultiStreamSpectraMultitask
    from .pipeline import find_audio_files, load_audio, order_by_submission
    from .wpt_spectra_inference import (
        _load_spectra, _logit, _preemphasis, _sigmoid,
        apply_fixed_task_moe_fusion, fixed_windows, predict_wpt_tasks,
    )
except ImportError:  # pragma: no cover - exercised by script.py
    from multistream_prompt_spectra import MultiStreamSpectraMultitask
    from pipeline import find_audio_files, load_audio, order_by_submission
    from wpt_spectra_inference import (
        _load_spectra, _logit, _preemphasis, _sigmoid,
        apply_fixed_task_moe_fusion, fixed_windows, predict_wpt_tasks,
    )


def load_multistream_model(
    model_dir: Path, checkpoint_path: Path, device: torch.device,
) -> tuple[MultiStreamSpectraMultitask, dict]:
    checkpoint = torch.load(
        checkpoint_path, map_location="cpu", weights_only=False,
    )
    if checkpoint.get("model_type") != "multistream_prompt_spectra_multitask":
        raise ValueError(f"not a multistream checkpoint: {checkpoint_path}")
    config = checkpoint["config"]
    model = MultiStreamSpectraMultitask(
        _load_spectra(model_dir, device),
        base_tokens=int(config["base_tokens"]),
        frequency_tokens=int(config["frequency_tokens"]),
        texture_tokens=int(config["texture_tokens"]),
        prompt_dropout=float(config["prompt_dropout"]),
        temperature=float(config["temperature"]),
    ).to(device)
    model.load_trainable_state_dict(checkpoint["state"])
    return model.eval(), config


@torch.inference_mode()
def predict_multistream_tasks(
    audio_paths: list[Path], model_dir: Path, checkpoint_path: Path,
    device: str = "cuda", batch_size: int = 12, views: int = 2,
) -> np.ndarray:
    """Return original-mixture Voice/Music/File probabilities in path order."""
    if batch_size <= 0 or views <= 0:
        raise ValueError("batch size and view count must be positive")
    target = torch.device(device)
    if target.type == "cuda":
        torch.cuda.empty_cache()
    model, config = load_multistream_model(model_dir, checkpoint_path, target)
    window = int(config["window"])
    outputs = []
    for offset in tqdm(
        range(0, len(audio_paths), batch_size), desc="multistream original-audio",
    ):
        paths = audio_paths[offset:offset + batch_size]
        windows = np.stack([
            fixed_windows(load_audio(path), window, views) for path in paths
        ])
        tensor = _preemphasis(torch.from_numpy(windows).to(target))
        with torch.autocast(
            device_type=target.type, dtype=torch.bfloat16,
            enabled=target.type == "cuda",
        ):
            logits = model(tensor)
        outputs.append(logits.float().cpu().numpy())
    del model
    gc.collect()
    if target.type == "cuda":
        torch.cuda.empty_cache()
    logits = np.concatenate(outputs) if outputs else np.empty((0, 3))
    return _sigmoid(np.clip(logits, -30, 30)).astype(np.float32)


def blend_file_experts(
    wpt_probabilities: np.ndarray,
    multistream_probabilities: np.ndarray,
    multistream_weight: float = .20,
) -> np.ndarray:
    """Blend only File logits; preserve both component columns bit-exactly."""
    if not 0 <= multistream_weight <= 1:
        raise ValueError("multistream weight must lie in [0, 1]")
    wpt = np.asarray(wpt_probabilities, dtype=np.float64)
    multistream = np.asarray(multistream_probabilities, dtype=np.float64)
    if wpt.shape != multistream.shape or wpt.ndim != 2 or wpt.shape[1] != 3:
        raise ValueError("both expert arrays must have shape [files, 3]")
    result = wpt.copy()
    result[:, 2] = _sigmoid(
        (1 - multistream_weight) * _logit(wpt[:, 2])
        + multistream_weight * _logit(multistream[:, 2])
    )
    return result


def apply_multistream_file_moe_fusion(
    test_dir: Path,
    submission_path: Path,
    model_dir: Path,
    wpt_checkpoint_path: Path,
    multistream_checkpoint_path: Path,
    unified_expert_path: Path,
    device: str = "cuda",
    wpt_batch_size: int = 6,
    wpt_file_views: int = 5,
    wpt_file_temperature: float = 2.0,
    multistream_batch_size: int = 12,
    multistream_views: int = 2,
    multistream_weight: float = .20,
    voice_outer_weight: float = .10,
    file_outer_weight: float = .75,
) -> None:
    """Apply the validated v41 fusion with a File-only multistream vote."""
    submission_path = Path(submission_path)
    with submission_path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    audio_paths = order_by_submission(find_audio_files(Path(test_dir)), rows)
    wpt = predict_wpt_tasks(
        audio_paths, model_dir, wpt_checkpoint_path,
        device=device, file_batch_size=wpt_batch_size,
        file_views=wpt_file_views, file_temperature=wpt_file_temperature,
    )
    multistream = predict_multistream_tasks(
        audio_paths, model_dir, multistream_checkpoint_path,
        device=device, batch_size=multistream_batch_size,
        views=multistream_views,
    )
    expert = blend_file_experts(wpt, multistream, multistream_weight)
    apply_fixed_task_moe_fusion(
        submission_path, unified_expert_path,
        np.asarray([row["ID"] for row in rows]), expert,
        voice_outer_weight=voice_outer_weight,
        file_outer_weight=file_outer_weight,
    )
