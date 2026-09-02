"""General-audio EAT encoder with a locally fitted music-forensics head."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import librosa
import numpy as np
import torch
import torch.nn.functional as F
import torchaudio

try:  # package import in tests; flat import in the offline submission
    from .eat_timm_compat import install_timm_compat
except ImportError:  # pragma: no cover - exercised by script.py
    from eat_timm_compat import install_timm_compat


def _load_local_model(model_dir: Path, device: torch.device):
    """Load bundled custom code without depending on HF's module cache."""
    # Do not pip-install timm in the grader: its torch/torchvision dependency
    # can replace the preinstalled CUDA build and make libtorchaudio.so fail.
    install_timm_compat()
    package = "davianspeech_eat"
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


class EatMusicDetector:
    SAMPLE_RATE = 16_000
    SAMPLES = 6 * SAMPLE_RATE
    FRAMES = 614
    NORM_MEAN = -4.268
    NORM_STD = 4.569

    def __init__(self, model_dir, head_path, device="cuda", extra_head_path=None):
        self.device = torch.device(device)
        self.model = _load_local_model(Path(model_dir), self.device)
        head = np.load(head_path)
        self.weight = torch.from_numpy(head["weight"]).to(self.device)
        self.bias = torch.as_tensor(head["bias"], device=self.device)
        self.extra_weight = self.extra_bias = None
        if extra_head_path is not None:
            extra = np.load(extra_head_path)
            self.extra_weight = torch.from_numpy(extra["weight"]).to(self.device)
            self.extra_bias = torch.as_tensor(extra["bias"], device=self.device)

    @classmethod
    def _crop(cls, audio: np.ndarray) -> np.ndarray:
        if len(audio) >= cls.SAMPLES:
            start = (len(audio) - cls.SAMPLES) // 2
            return audio[start:start + cls.SAMPLES]
        left = (cls.SAMPLES - len(audio)) // 2
        return np.pad(audio, (left, cls.SAMPLES - len(audio) - left))

    @classmethod
    def _fbank(cls, audio: np.ndarray) -> torch.Tensor:
        waveform = torch.from_numpy(cls._crop(audio)).float()
        waveform = waveform - waveform.mean()
        mel = torchaudio.compliance.kaldi.fbank(
            waveform.unsqueeze(0), htk_compat=True,
            sample_frequency=cls.SAMPLE_RATE, use_energy=False,
            window_type="hanning", num_mel_bins=128, dither=0.0,
            frame_shift=10,
        )
        if mel.shape[0] < cls.FRAMES:
            mel = F.pad(mel, (0, 0, 0, cls.FRAMES - mel.shape[0]))
        else:
            mel = mel[:cls.FRAMES]
        return (mel - cls.NORM_MEAN) / (cls.NORM_STD * 2)

    @torch.inference_mode()
    def fake_probabilities(self, audio: np.ndarray) -> tuple[float, ...]:
        mel = self._fbank(audio)[None, None].to(self.device)
        embedding = self.model.extract_features(mel)[:, 0]
        logit = embedding.float() @ self.weight + self.bias
        probabilities = [float(torch.sigmoid(logit)[0])]
        if self.extra_weight is not None:
            extra_logit = embedding.float() @ self.extra_weight + self.extra_bias
            probabilities.append(float(torch.sigmoid(extra_logit)[0]))
        return tuple(probabilities)

    def fake_probability(self, audio: np.ndarray) -> float:
        return self.fake_probabilities(audio)[0]
