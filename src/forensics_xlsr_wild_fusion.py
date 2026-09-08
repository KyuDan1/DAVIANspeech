"""Frozen original-mixture Forensics XLS-R Wild Voice residual.

The external detector is deliberately treated as a weak diversity expert.  It
scores three deterministic five-second views of each original file, averages
its fake-oriented logits, and changes only ``VOICE_FAKE_PROB``.  A caller may
therefore run component-consistent File fusion afterwards without this module
ever touching Music or either presence output.
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

try:  # package import in tests; flat import inside the submission package
    from .pipeline import find_audio_files, load_audio, order_by_submission
except ImportError:  # pragma: no cover - exercised by submission script.py
    from pipeline import find_audio_files, load_audio, order_by_submission


REQUIRED_COLUMNS = {
    "ID", "FILE_FAKE_PROB", "VOICE_FAKE_PROB", "MUSIC_FAKE_PROB",
    "VOICE_PRESENT_PROB", "MUSIC_PRESENT_PROB",
}
ALLOWED_UNEXPECTED_KEYS = {
    "projection.0.bias", "projection.0.weight",
    "projection.2.bias", "projection.2.weight",
}


def _logit(values) -> np.ndarray:
    values = np.clip(np.asarray(values, dtype=np.float64), 1e-5, 1 - 1e-5)
    return np.log(values) - np.log1p(-values)


def _sigmoid(values) -> np.ndarray:
    return np.exp(-np.logaddexp(0.0, -np.asarray(values, dtype=np.float64)))


def fixed_windows(
    audio: np.ndarray, length: int = 80_000, count: int = 3,
) -> np.ndarray:
    """Return peak-normalized, evenly spaced deterministic waveform views."""
    if length <= 0 or count <= 0:
        raise ValueError("window length and count must be positive")
    audio = np.asarray(audio, dtype=np.float32).reshape(-1)
    if not len(audio):
        audio = np.zeros(1, dtype=np.float32)
    if not np.isfinite(audio).all():
        raise ValueError("Forensics input audio contains non-finite values")
    peak = float(np.max(np.abs(audio)))
    if peak > 0:
        audio = audio / (peak + 1e-8)
    if len(audio) < length:
        tiled = np.tile(audio, math.ceil(length / len(audio)))[:length]
        return np.repeat(tiled[None], count, axis=0)
    starts = (
        np.asarray([(len(audio) - length) // 2], dtype=np.int64)
        if count == 1 else
        np.rint(np.linspace(0, len(audio) - length, count)).astype(np.int64)
    )
    return np.stack([audio[start:start + length] for start in starts])


def mix_voice_logit(
    anchor_probability: np.ndarray,
    forensics_fake_logit: np.ndarray,
    *,
    voice_weight: float = 0.075,
) -> np.ndarray:
    """Blend a calibrated anchor probability with a raw fake-oriented logit."""
    if not 0.0 <= voice_weight <= 0.10:
        raise ValueError("frozen guard limits Forensics Voice weight to 10%")
    anchor = np.asarray(anchor_probability, dtype=np.float64)
    expert = np.asarray(forensics_fake_logit, dtype=np.float64)
    if anchor.shape != expert.shape:
        raise ValueError("anchor probabilities and Forensics logits must align")
    if not np.isfinite(anchor).all() or not np.isfinite(expert).all():
        raise ValueError("Voice fusion inputs must be finite")
    if not ((anchor >= 0.0) & (anchor <= 1.0)).all():
        raise ValueError("anchor Voice values must be probabilities")
    return _sigmoid(
        (1.0 - voice_weight) * _logit(anchor) + voice_weight * expert
    )


def load_forensics_model(
    model_dir: Path, device: torch.device,
) -> torch.nn.Module:
    """Construct the vendored architecture and load the pinned safe checkpoint."""
    model_dir = Path(model_dir)
    spec = importlib.util.spec_from_file_location(
        "vendored_forensics_xlsr_wild", model_dir / "model.py",
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import Forensics model from {model_dir}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    model = module.DeepfakeDetector()
    state = load_file(model_dir / "checkpoint_epoch_4.safetensors")
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or set(unexpected) != ALLOWED_UNEXPECTED_KEYS:
        raise RuntimeError(
            f"Forensics checkpoint mismatch: missing={missing[:5]}, "
            f"unexpected={unexpected[:5]}"
        )
    del state
    return model.eval().to(device)


@torch.inference_mode()
def predict_forensics_fake_logits(
    audio_files: list[Path],
    model_dir: Path,
    *,
    device: str = "cuda",
    windows: int = 3,
    window_samples: int = 80_000,
    file_batch_size: int = 8,
) -> tuple[np.ndarray, np.ndarray]:
    """Return IDs and mean fake logits for fixed original-mixture views."""
    if windows <= 0 or window_samples <= 0 or file_batch_size <= 0:
        raise ValueError("Forensics window and batch arguments must be positive")
    if not audio_files:
        raise ValueError("Forensics Voice fusion received no audio files")
    ids = np.asarray([path.stem for path in audio_files])
    if len(set(ids)) != len(ids):
        raise ValueError("Forensics audio IDs must be unique")
    target = torch.device(device)
    model = load_forensics_model(model_dir, target)
    outputs: list[np.ndarray] = []
    try:
        for offset in tqdm(
            range(0, len(audio_files), file_batch_size),
            desc="Forensics Voice",
        ):
            paths = audio_files[offset:offset + file_batch_size]
            waveforms = np.concatenate([
                fixed_windows(load_audio(path), window_samples, windows)
                for path in paths
            ])
            values = torch.from_numpy(waveforms).to(target)
            with torch.autocast(
                device_type=target.type,
                dtype=torch.bfloat16,
                enabled=target.type == "cuda",
            ):
                # The released classifier logit is oriented toward P(real).
                fake = -model(values).float().reshape(len(paths), windows)
            outputs.append(fake.mean(dim=1).cpu().numpy())
    finally:
        del model
        gc.collect()
        if target.type == "cuda":
            torch.cuda.empty_cache()
    scores = np.concatenate(outputs).astype(np.float64)
    if not np.isfinite(scores).all():
        raise RuntimeError("Forensics model returned non-finite logits")
    return ids, scores


def apply_voice_logit_residual(
    submission_path: Path,
    expert_ids: np.ndarray,
    expert_fake_logits: np.ndarray,
    *,
    voice_weight: float = 0.075,
) -> None:
    """Atomically change Voice only after strict per-file ID alignment."""
    expert_ids = np.asarray(expert_ids).astype(str)
    expert_fake_logits = np.asarray(expert_fake_logits, dtype=np.float64)
    if expert_fake_logits.shape != (len(expert_ids),):
        raise ValueError("Forensics logits must have shape [files]")
    if len(set(expert_ids)) != len(expert_ids):
        raise ValueError("Forensics expert IDs contain duplicates")

    submission_path = Path(submission_path)
    with submission_path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        columns = list(reader.fieldnames or [])
        rows = list(reader)
    if missing := REQUIRED_COLUMNS.difference(columns):
        raise ValueError(f"submission misses columns: {sorted(missing)}")
    row_ids = [row["ID"] for row in rows]
    if len(set(row_ids)) != len(row_ids) or set(row_ids) != set(expert_ids):
        raise ValueError("submission and Forensics IDs differ")
    position = {item: index for index, item in enumerate(expert_ids)}
    anchor = np.asarray(
        [float(row["VOICE_FAKE_PROB"]) for row in rows], dtype=np.float64,
    )
    order = np.asarray([position[item] for item in row_ids], dtype=np.int64)
    updated = mix_voice_logit(
        anchor, expert_fake_logits[order], voice_weight=voice_weight,
    )
    for row, probability in zip(rows, updated):
        row["VOICE_FAKE_PROB"] = round(float(probability), 10)

    temporary = submission_path.with_suffix(".tmp.csv")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(submission_path)


def apply_forensics_voice_fusion(
    test_dir: Path,
    submission_path: Path,
    model_dir: Path,
    *,
    device: str = "cuda",
    voice_weight: float = 0.075,
    windows: int = 3,
    window_samples: int = 80_000,
    file_batch_size: int = 8,
) -> None:
    """Score original mixtures and apply the frozen Voice-only residual."""
    audio_files = find_audio_files(Path(test_dir))
    with Path(submission_path).open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    audio_files = order_by_submission(audio_files, rows)
    ids, scores = predict_forensics_fake_logits(
        audio_files, model_dir, device=device, windows=windows,
        window_samples=window_samples, file_batch_size=file_batch_size,
    )
    apply_voice_logit_residual(
        submission_path, ids, scores, voice_weight=voice_weight,
    )
