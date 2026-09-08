"""Offline inference and protected-output fusion for EAT-large–AASIST v56."""

from __future__ import annotations

import csv
import hashlib
import importlib.util
import math
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

try:
    from .eat_timm_compat import install_timm_compat
    from .eat_large_aasist import EatLargeAASISTExpert
except ImportError:  # pragma: no cover - flat imports in a submission package
    from eat_timm_compat import install_timm_compat
    from eat_large_aasist import EatLargeAASISTExpert


SAMPLE_RATE = 16_000
SAMPLES = 163_840  # 10.24 s
FRAMES = 1_024
NORM_MEAN = -4.268
NORM_STD = 4.569
PROTECTED_COLUMNS = (
    "VOICE_FAKE_PROB", "VOICE_PRESENT_PROB", "MUSIC_PRESENT_PROB",
)


def _load_local_model(model_dir: Path, device: torch.device):
    """Load bundled EAT custom code without importing torchaudio or timm."""
    install_timm_compat()
    package = "davianspeech_eat_v56"
    package_spec = importlib.util.spec_from_file_location(
        package, model_dir / "__init__.py",
        submodule_search_locations=[str(model_dir)],
    )
    module = importlib.util.module_from_spec(package_spec)
    sys.modules[package] = module
    model_spec = importlib.util.spec_from_file_location(
        f"{package}.modeling_eat", model_dir / "modeling_eat.py"
    )
    modeling = importlib.util.module_from_spec(model_spec)
    sys.modules[model_spec.name] = modeling
    model_spec.loader.exec_module(modeling)
    return modeling.EATModel.from_pretrained(model_dir).eval().to(device)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def crop_or_pad(audio: np.ndarray, start: int, samples: int = SAMPLES) -> np.ndarray:
    audio = np.asarray(audio, dtype=np.float32)
    if len(audio) >= start + samples:
        return np.ascontiguousarray(audio[start : start + samples])
    result = np.zeros(samples, dtype=np.float32)
    available = audio[start : min(len(audio), start + samples)]
    result[: len(available)] = available
    return result


def view_starts(length: int, samples: int = SAMPLES) -> list[int]:
    if length <= samples:
        return [0]
    duration = length / SAMPLE_RATE
    count = 2 if duration <= 30 else 3
    return sorted(set(
        np.rint(np.linspace(0, length - samples, count)).astype(int).tolist()
    ))


@lru_cache(maxsize=1)
def _kaldi_window_and_mel() -> tuple[torch.Tensor, torch.Tensor]:
    """Pure-torch constants matching torchaudio.compliance.kaldi defaults."""
    window_size, padded_size, mel_bins = 400, 512, 128
    window = torch.hann_window(window_size, periodic=False)
    low_freq, high_freq = 20.0, SAMPLE_RATE / 2
    mel_low = 1127.0 * math.log(1.0 + low_freq / 700.0)
    mel_high = 1127.0 * math.log(1.0 + high_freq / 700.0)
    delta = (mel_high - mel_low) / (mel_bins + 1)
    index = torch.arange(mel_bins).unsqueeze(1)
    left = mel_low + index * delta
    center = mel_low + (index + 1) * delta
    right = mel_low + (index + 2) * delta
    frequencies = (SAMPLE_RATE / padded_size) * torch.arange(padded_size // 2)
    mel = 1127.0 * torch.log1p(frequencies / 700.0)[None]
    up = (mel - left) / (center - left)
    down = (right - mel) / (right - center)
    banks = torch.minimum(up, down).clamp_min(0)
    banks = F.pad(banks, (0, 1))
    return window, banks


def fbank(audio: np.ndarray) -> torch.Tensor:
    """Kaldi-compatible 128-bin fbank implemented without torchaudio.

    This mirrors the established EAT settings: 25 ms Hann frames, 10 ms
    shift, frame DC removal, 0.97 pre-emphasis, 512-point power spectrum,
    Kaldi mel scale and snip-edges.
    """
    waveform = torch.from_numpy(np.asarray(audio, dtype=np.float32)).float()
    if len(waveform) < 400:
        waveform = F.pad(waveform, (0, 400 - len(waveform)))
    frames = waveform.unfold(0, 400, 160).clone()
    frames -= frames.mean(dim=1, keepdim=True)
    previous = F.pad(frames[:, :-1], (1, 0), mode="replicate")
    frames = frames - 0.97 * previous
    window, banks = _kaldi_window_and_mel()
    frames *= window
    frames = F.pad(frames, (0, 112))
    spectrum = torch.fft.rfft(frames).abs().square()
    mel = (spectrum @ banks.T).clamp_min(torch.finfo(spectrum.dtype).eps).log()
    if mel.shape[0] < FRAMES:
        mel = F.pad(mel, (0, 0, 0, FRAMES - mel.shape[0]))
    else:
        mel = mel[:FRAMES]
    return (mel - NORM_MEAN) / (NORM_STD * 2)


def audio_views(audio: np.ndarray, maximum_views: int = 3) -> tuple[torch.Tensor, torch.Tensor]:
    starts = view_starts(len(audio))[:maximum_views]
    values = torch.zeros(maximum_views, FRAMES, 128)
    mask = torch.zeros(maximum_views, dtype=torch.bool)
    for index, start in enumerate(starts):
        values[index] = fbank(crop_or_pad(audio, start))
        mask[index] = True
    return values, mask


def load_expert(
    model_dir: Path, checkpoint_path: Path, device: torch.device,
) -> EatLargeAASISTExpert:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint.get("model_type") != "eat_large_aasist_v56":
        raise ValueError("not an EAT-large–AASIST v56 checkpoint")
    actual = sha256(Path(model_dir) / "model.safetensors")
    if actual != checkpoint["base_eat_sha256"]:
        raise ValueError("EAT-large base hash differs from checkpoint")
    eat = _load_local_model(Path(model_dir), device)
    model = EatLargeAASISTExpert(eat, **checkpoint["config"]).to(device)
    incompatible = model.load_state_dict(checkpoint["model"], strict=False)
    if incompatible.unexpected_keys or any(
        not name.startswith("eat.") for name in incompatible.missing_keys
    ):
        raise ValueError(f"incompatible v56 checkpoint: {incompatible}")
    return model.eval()


class EATLargeAASISTScorer:
    """Batched original-mixture scorer used by the deployment pipeline."""

    def __init__(
        self, model_dir: Path, checkpoint_path: Path, device: str = "cuda",
        file_batch_size: int = 2, maximum_views: int = 3,
    ) -> None:
        self.device = torch.device(device)
        self.model = load_expert(model_dir, checkpoint_path, self.device)
        self.file_batch_size = int(file_batch_size)
        self.maximum_views = int(maximum_views)
        self.pending: list[tuple[str, np.ndarray]] = []
        self.ids: list[str] = []
        self.probabilities: list[np.ndarray] = []

    def add(self, item_id: str, original_audio: np.ndarray) -> None:
        self.pending.append((str(item_id), np.asarray(original_audio, dtype=np.float32)))
        if len(self.pending) >= self.file_batch_size:
            self._flush()

    @torch.inference_mode()
    def _flush(self) -> None:
        if not self.pending:
            return
        blocks = [audio_views(audio, self.maximum_views) for _, audio in self.pending]
        features = torch.stack([item[0] for item in blocks])[:, :, None].to(self.device)
        mask = torch.stack([item[1] for item in blocks]).to(self.device)
        with torch.autocast(
            device_type=self.device.type, dtype=torch.bfloat16,
            enabled=self.device.type == "cuda",
        ):
            logits = self.model(features, mask)
            probability = self.model.backend.probabilities(logits)
        self.ids.extend(item[0] for item in self.pending)
        self.probabilities.extend(probability.float().cpu().numpy())
        self.pending.clear()

    def save(self, output: Path) -> None:
        self._flush()
        output = Path(output)
        output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            output, ids=np.asarray(self.ids),
            probabilities=np.asarray(self.probabilities, dtype=np.float32),
        )


def _logit(value: float | np.ndarray) -> np.ndarray:
    clipped = np.clip(np.asarray(value, dtype=np.float64), 1e-6, 1 - 1e-6)
    return np.log(clipped) - np.log1p(-clipped)


def _sigmoid(value: float | np.ndarray) -> np.ndarray:
    value = np.asarray(value, dtype=np.float64)
    return np.exp(-np.logaddexp(0.0, -value))


def apply_eat_large_aasist_fusion(
    submission_path: Path, statistics_path: Path,
    file_weight: float = 0.20, music_weight: float = 0.30,
) -> None:
    """Fuse File/Music logits while keeping Voice and CPS text byte-identical."""
    if not 0 <= file_weight <= 1 or not 0 <= music_weight <= 1:
        raise ValueError("fusion weights must lie in [0, 1]")
    archive = np.load(statistics_path, allow_pickle=False)
    ids = archive["ids"].astype(str)
    probability = archive["probabilities"]
    if probability.shape != (len(ids), 3) or len(set(ids)) != len(ids):
        raise ValueError("invalid v56 statistics archive")
    expert = dict(zip(ids, probability))

    submission_path = Path(submission_path)
    with submission_path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        columns, rows = list(reader.fieldnames or []), list(reader)
    before = [tuple(row[column] for column in PROTECTED_COLUMNS) for row in rows]
    if {row["ID"] for row in rows} != set(expert):
        raise ValueError("submission and v56 statistics IDs differ")
    for row in rows:
        current = expert[row["ID"]]
        row["FILE_FAKE_PROB"] = round(float(_sigmoid(
            (1 - file_weight) * _logit(float(row["FILE_FAKE_PROB"]))
            + file_weight * _logit(float(current[2]))
        )), 10)
        row["MUSIC_FAKE_PROB"] = round(float(_sigmoid(
            (1 - music_weight) * _logit(float(row["MUSIC_FAKE_PROB"]))
            + music_weight * _logit(float(current[1]))
        )), 10)
    after = [tuple(row[column] for column in PROTECTED_COLUMNS) for row in rows]
    if before != after:
        raise RuntimeError("v56 modified a protected Voice/CPS output")
    temporary = submission_path.with_suffix(".tmp.csv")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(submission_path)
