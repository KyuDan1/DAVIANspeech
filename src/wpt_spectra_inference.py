"""Offline original-audio WPT/Spectra expert and fixed task-wise MoE.

The expert never receives a separated stem.  It observes three deterministic
views of the original waveform, which preserves generator and transmission
artifacts.  Its Voice and File decisions are combined with the independent
unified EAT/SPEAR expert using development-selected, fixed logit weights.
"""

from __future__ import annotations

import csv
import gc
import importlib.util
import math
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file
from tqdm import tqdm

try:  # package import in tests; flat imports inside the submission archive
    from .pipeline import find_audio_files, load_audio, order_by_submission
    from .wpt_spectra import WPTSpectraMultitask
except ImportError:  # pragma: no cover - exercised by script.py
    from pipeline import find_audio_files, load_audio, order_by_submission
    from wpt_spectra import WPTSpectraMultitask


def _logit(values) -> np.ndarray:
    values = np.clip(np.asarray(values, dtype=np.float64), 1e-5, 1 - 1e-5)
    return np.log(values) - np.log1p(-values)


def _sigmoid(values) -> np.ndarray:
    return np.exp(-np.logaddexp(0, -np.asarray(values, dtype=np.float64)))


def fixed_windows(audio: np.ndarray, length: int, count: int) -> np.ndarray:
    """Return deterministic, evenly spaced views without zero-padding."""
    audio = np.asarray(audio, dtype=np.float32)
    if count <= 0 or length <= 0:
        raise ValueError("window length and count must be positive")
    if not len(audio):
        audio = np.zeros(1, dtype=np.float32)
    if len(audio) < length:
        repeated = np.tile(audio, math.ceil(length / len(audio)))[:length]
        return np.repeat(repeated[None], count, axis=0)
    maximum = len(audio) - length
    starts = (
        np.asarray([maximum // 2], dtype=np.int64)
        if count == 1
        else np.rint(np.linspace(0, maximum, count)).astype(np.int64)
    )
    return np.stack([audio[start:start + length] for start in starts])


def _preemphasis(waveforms: torch.Tensor, coefficient: float = .97) -> torch.Tensor:
    result = waveforms.clone()
    result[..., 1:] = waveforms[..., 1:] - coefficient * waveforms[..., :-1]
    return result


def _load_spectra(model_dir: Path, device: torch.device) -> torch.nn.Module:
    spec = importlib.util.spec_from_file_location(
        "wpt_vendored_spectra", Path(model_dir) / "model.py"
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot import Spectra-AASIST from {model_dir}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    model = module.SpectraAASIST()
    missing, unexpected = model.load_state_dict(
        load_file(Path(model_dir) / "model.safetensors"), strict=False
    )
    if missing or unexpected:
        raise RuntimeError(
            f"Spectra checkpoint mismatch: {missing[:5]} {unexpected[:5]}"
        )
    return model.to(device)


def load_wpt_model(
    model_dir: Path, checkpoint_path: Path, device: torch.device,
) -> tuple[WPTSpectraMultitask, dict]:
    checkpoint = torch.load(
        checkpoint_path, map_location="cpu", weights_only=False
    )
    if checkpoint.get("model_type") != "wpt_spectra_multitask":
        raise ValueError(f"not a WPT/Spectra checkpoint: {checkpoint_path}")
    config = checkpoint["config"]
    model = WPTSpectraMultitask(
        _load_spectra(model_dir, device),
        prompt_tokens=int(config["prompt_tokens"]),
        wavelet_tokens=int(config["wavelet_tokens"]),
        temperature=float(config["temperature"]),
    ).to(device)
    model.load_trainable_state_dict(checkpoint["state"])
    return model.eval(), config


@torch.inference_mode()
def predict_wpt_tasks(
    audio_paths: list[Path], model_dir: Path, checkpoint_path: Path,
    device: str = "cuda", file_batch_size: int = 4,
) -> np.ndarray:
    """Predict Voice/Music/File probabilities in input-path order."""
    if file_batch_size <= 0:
        raise ValueError("file_batch_size must be positive")
    target = torch.device(device)
    if target.type == "cuda":
        torch.cuda.empty_cache()
    model, config = load_wpt_model(model_dir, checkpoint_path, target)
    views = int(config["views"])
    window = int(config["window"])
    outputs = []
    for offset in tqdm(
        range(0, len(audio_paths), file_batch_size), desc="WPT original-audio"
    ):
        paths = audio_paths[offset:offset + file_batch_size]
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


def apply_fixed_task_moe_fusion(
    submission_path: Path,
    unified_expert_path: Path,
    wpt_ids: np.ndarray,
    wpt_probabilities: np.ndarray,
    voice_outer_weight: float = .10,
    file_outer_weight: float = .60,
    voice_wpt_weight: float = .90,
    file_wpt_weight: float = .80,
) -> None:
    """Fuse fixed expert votes into Voice/File while leaving Music/CPS exact."""
    weights = (
        voice_outer_weight, file_outer_weight,
        voice_wpt_weight, file_wpt_weight,
    )
    if any(not 0 <= value <= 1 for value in weights):
        raise ValueError("all MoE weights must lie in [0, 1]")
    wpt_ids = np.asarray(wpt_ids).astype(str)
    wpt_probabilities = np.asarray(wpt_probabilities, dtype=np.float64)
    if wpt_probabilities.shape != (len(wpt_ids), 3):
        raise ValueError("WPT probabilities must have shape [files, 3]")
    if len(set(wpt_ids)) != len(wpt_ids):
        raise ValueError("WPT predictions contain duplicate IDs")
    with np.load(unified_expert_path, allow_pickle=False) as archive:
        unified_ids = archive["ids"].astype(str)
        unified_probabilities = archive["probabilities"].astype(np.float64)
    if unified_probabilities.shape != (len(unified_ids), 3):
        raise ValueError("unified probabilities must have shape [files, 3]")
    if len(set(unified_ids)) != len(unified_ids):
        raise ValueError("unified predictions contain duplicate IDs")
    if set(unified_ids) != set(wpt_ids):
        raise ValueError("unified and WPT prediction IDs differ")
    unified_by_id = dict(zip(unified_ids, unified_probabilities))
    wpt_by_id = dict(zip(wpt_ids, wpt_probabilities))

    submission_path = Path(submission_path)
    with submission_path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        columns = list(reader.fieldnames or [])
        rows = list(reader)
    if {row["ID"] for row in rows} != set(wpt_ids):
        raise ValueError("submission and expert prediction IDs differ")
    for row in rows:
        item = row["ID"]
        unified = unified_by_id[item]
        wpt = wpt_by_id[item]
        voice_expert = _sigmoid(
            (1 - voice_wpt_weight) * _logit(unified[0])
            + voice_wpt_weight * _logit(wpt[0])
        )
        file_expert = _sigmoid(
            (1 - file_wpt_weight) * _logit(unified[2])
            + file_wpt_weight * _logit(wpt[2])
        )
        row["VOICE_FAKE_PROB"] = round(float(_sigmoid(
            (1 - voice_outer_weight) * _logit(float(row["VOICE_FAKE_PROB"]))
            + voice_outer_weight * _logit(voice_expert)
        )), 10)
        row["FILE_FAKE_PROB"] = round(float(_sigmoid(
            (1 - file_outer_weight) * _logit(float(row["FILE_FAKE_PROB"]))
            + file_outer_weight * _logit(file_expert)
        )), 10)
    temporary = submission_path.with_suffix(".tmp.csv")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(submission_path)


def apply_wpt_fixed_moe_fusion(
    test_dir: Path,
    submission_path: Path,
    model_dir: Path,
    checkpoint_path: Path,
    unified_expert_path: Path,
    device: str = "cuda",
    file_batch_size: int = 4,
    voice_outer_weight: float = .10,
    file_outer_weight: float = .60,
) -> None:
    """Score original audio, then apply the validated fixed task-wise MoE."""
    submission_path = Path(submission_path)
    with submission_path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    audio_paths = order_by_submission(find_audio_files(Path(test_dir)), rows)
    probabilities = predict_wpt_tasks(
        audio_paths, model_dir, checkpoint_path,
        device=device, file_batch_size=file_batch_size,
    )
    apply_fixed_task_moe_fusion(
        submission_path, unified_expert_path,
        np.asarray([row["ID"] for row in rows]), probabilities,
        voice_outer_weight=voice_outer_weight,
        file_outer_weight=file_outer_weight,
    )
