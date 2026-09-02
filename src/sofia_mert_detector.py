"""Torchaudio-free inference for the official SOFIA G1-MERT expert.

The released SOFIA G1 head is reproduced with its frozen MERT-v1-95M encoder.
Only original mixture audio is used; no source separation is performed.
"""

from __future__ import annotations

import csv
import math
from pathlib import Path

import librosa
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from transformers import AutoModel, Wav2Vec2FeatureExtractor


def _sinc_resample(waveform: torch.Tensor, source_rate: int, target_rate: int) -> torch.Tensor:
    """Pure-PyTorch equivalent of torchaudio's default functional resampler."""
    if source_rate == target_rate:
        return waveform
    gcd = math.gcd(int(source_rate), int(target_rate))
    source = int(source_rate) // gcd
    target = int(target_rate) // gcd
    lowpass_width, rolloff = 6, 0.99
    base = min(source, target) * rolloff
    width = math.ceil(lowpass_width * source / base)
    idx = torch.arange(
        -width, width + source, dtype=waveform.dtype,
        device=waveform.device,
    )[None, None] / source
    time = (
        torch.arange(0, -target, -1, dtype=waveform.dtype,
                     device=waveform.device)[:, None, None] / target + idx
    )
    time = (time * base).clamp_(-lowpass_width, lowpass_width)
    window = torch.cos(time * math.pi / lowpass_width / 2).square()
    time = time * math.pi
    kernel = torch.where(time == 0, torch.ones_like(time), time.sin() / time)
    kernel = kernel * window * (base / source)
    shape = waveform.shape
    packed = waveform.reshape(-1, shape[-1])
    padded = F.pad(packed, (width, width + source))
    result = F.conv1d(padded[:, None], kernel, stride=source)
    result = result.transpose(1, 2).reshape(len(packed), -1)
    length = math.ceil(target * shape[-1] / source)
    return result[..., :length].reshape(shape[:-1] + (length,))


class _SofiaG1Head(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.projector = nn.Sequential(
            nn.Linear(768, 256), nn.LayerNorm(256), nn.GELU(), nn.Dropout(0.1)
        )
        self.fusion = nn.Sequential(
            nn.Linear(256, 512), nn.LayerNorm(512), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(512, 256),
        )
        self.classifier = nn.Linear(256, 2)

    def load_sofia_state(self, state: dict) -> None:
        fusion = state["fusion"]
        mapping = {
            "projector.0.weight": "projectors.mert.net.0.weight",
            "projector.0.bias": "projectors.mert.net.0.bias",
            "projector.1.weight": "projectors.mert.net.1.weight",
            "projector.1.bias": "projectors.mert.net.1.bias",
            "fusion.0.weight": "fusion.net.0.weight",
            "fusion.0.bias": "fusion.net.0.bias",
            "fusion.1.weight": "fusion.net.1.weight",
            "fusion.1.bias": "fusion.net.1.bias",
            "fusion.4.weight": "fusion.net.4.weight",
            "fusion.4.bias": "fusion.net.4.bias",
        }
        own = self.state_dict()
        for target, source in mapping.items():
            own[target] = fusion[source]
        own["classifier.weight"] = state["head"]["fc.weight"]
        own["classifier.bias"] = state["head"]["fc.bias"]
        self.load_state_dict(own)

    def forward(self, embedding: torch.Tensor) -> torch.Tensor:
        value = self.projector(F.normalize(embedding, dim=-1, eps=1e-6))
        value = self.fusion(value)
        value = F.normalize(value, dim=-1, eps=1e-6)
        return self.classifier(value)


class SofiaMertDetector:
    SAMPLE_RATE = 24_000
    # The released SOFIA collator uses ``60 * 16000`` samples as a global cap
    # even for its 24 kHz branches: exactly 40 seconds at the MERT rate.
    TARGET_SAMPLES = 960_000

    def __init__(
        self, model_dir: Path, head_path: Path, device: str = "cuda",
        pad_to_release_length: bool = False,
    ) -> None:
        self.device = torch.device(device)
        self.pad_to_release_length = bool(pad_to_release_length)
        self.processor = Wav2Vec2FeatureExtractor.from_pretrained(
            model_dir, local_files_only=True, trust_remote_code=True
        )
        self.encoder = AutoModel.from_pretrained(
            model_dir, local_files_only=True, trust_remote_code=True
        ).to(self.device).eval()
        for parameter in self.encoder.parameters():
            parameter.requires_grad_(False)
        state = torch.load(head_path, map_location="cpu", weights_only=False)
        self.head = _SofiaG1Head().to(self.device).eval()
        self.head.load_sofia_state(state)

    @torch.inference_mode()
    def _embed_24k(self, audio_24k: np.ndarray) -> torch.Tensor:
        audio = np.asarray(audio_24k, dtype=np.float32)
        # Match the released SOFIA loader's per-file peak normalization before
        # the MERT feature extractor applies zero-mean/unit-variance scaling.
        audio = audio / max(float(np.max(np.abs(audio))), 1e-9)
        if self.pad_to_release_length:
            if audio.size > self.TARGET_SAMPLES:
                start = (audio.size - self.TARGET_SAMPLES) // 2
                audio = audio[start:start + self.TARGET_SAMPLES]
            elif audio.size < self.TARGET_SAMPLES:
                missing = self.TARGET_SAMPLES - audio.size
                before = missing // 2
                audio = np.pad(audio, (before, missing - before))
        inputs = self.processor(
            audio, sampling_rate=self.SAMPLE_RATE, return_tensors="pt"
        ).to(self.device)
        output = self.encoder(**inputs, output_hidden_states=True)
        if output.hidden_states is None:
            raise RuntimeError("MERT did not return hidden states")
        # SOFIA G1-MERT release uses layer_mean: temporal mean for every layer,
        # followed by an unweighted mean over all hidden layers.
        return torch.stack(output.hidden_states).mean(dim=2).mean(dim=0)

    def _score_24k(self, audio_24k: np.ndarray) -> float:
        embedding = self._embed_24k(audio_24k)
        return float(self.head(embedding).softmax(dim=-1)[0, 1].float().cpu())

    def _prepare_path_24k(self, path: Path) -> np.ndarray:
        # Reproduce SOFIA's released two-stage torchaudio path without importing
        # libtorchaudio.so (which has failed on the competition worker before).
        audio, source_rate = librosa.load(
            path, sr=None, mono=False, dtype=np.float32
        )
        if audio.ndim == 1:
            audio = audio[None]
        waveform = torch.from_numpy(audio)
        waveform = _sinc_resample(waveform, int(source_rate), 44_100)
        waveform = waveform / waveform.abs().max().clamp_min(1e-9)
        waveform = _sinc_resample(waveform, 44_100, self.SAMPLE_RATE)
        waveform = waveform.mean(dim=0)
        return waveform.numpy()

    def embed_path(self, path: Path) -> np.ndarray:
        return self._embed_24k(self._prepare_path_24k(path)).float().cpu().numpy()[0]

    def score_path(self, path: Path) -> float:
        return self._score_24k(self._prepare_path_24k(path))

    def score(self, audio_16k: np.ndarray) -> float:
        """Compatibility helper; prefer :meth:`score_path` when a file exists."""
        audio = librosa.resample(
            np.asarray(audio_16k, dtype=np.float32), orig_sr=16_000,
            target_sr=self.SAMPLE_RATE, res_type="soxr_hq",
        )
        return self._score_24k(audio)


def _logit(value: float) -> float:
    clipped = float(np.clip(value, 1e-5, 1 - 1e-5))
    return float(np.log(clipped) - np.log1p(-clipped))


def _blend(anchor: float, expert: float, weight: float) -> float:
    value = (1 - weight) * _logit(anchor) + weight * _logit(expert)
    return float(1 / (1 + np.exp(-value)))


def apply_sofia_mert_fusion(
    test_dir: Path,
    submission_path: Path,
    model_dir: Path,
    head_path: Path,
    device: str = "cuda",
    file_weight: float = 0.10,
    music_weight: float = 0.05,
) -> None:
    """Add low-weight MERT evidence to File/Music without changing Voice/CPS."""
    detector = SofiaMertDetector(model_dir, head_path, device=device)
    with submission_path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        columns = list(reader.fieldnames or [])
        rows = list(reader)
    paths = {
        path.stem: path for path in test_dir.iterdir()
        if path.is_file()
    }
    for row in rows:
        path = paths.get(row["ID"])
        if path is None:
            raise FileNotFoundError(f"No audio for {row['ID']}")
        score = detector.score_path(path)
        row["FILE_FAKE_PROB"] = round(_blend(
            float(row["FILE_FAKE_PROB"]), score, file_weight
        ), 10)
        row["MUSIC_FAKE_PROB"] = round(_blend(
            float(row["MUSIC_FAKE_PROB"]), score, music_weight
        ), 10)
    temporary = submission_path.with_suffix(".tmp.csv")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(submission_path)
