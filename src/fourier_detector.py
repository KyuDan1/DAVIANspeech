"""High-frequency Fourier fakeprint detector for AI-generated music.

Feature extraction follows Deezer's ISMIR 2025 reference implementation
(CC BY-NC 4.0), adapted to the competition's fixed 16 kHz sample rate.
"""

from __future__ import annotations

from pathlib import Path

import librosa
import numpy as np
from scipy import interpolate


class FourierMusicDetector:
    SR = 16_000
    N_FFT = 1 << 14
    FMIN = 5_000
    FMAX = 7_990

    def __init__(self, head_path: Path):
        head = np.load(head_path)
        self.weight = head["weight"]
        self.bias = float(head["bias"])

    @staticmethod
    def _lower_hull(values: np.ndarray, area=10):
        indices, hull = [], []
        for offset in range(len(values) - area + 1):
            index = offset + int(np.argmin(values[offset:offset + area]))
            if not indices or index != indices[-1]:
                indices.append(index); hull.append(values[index])
        if indices[0] != 0:
            indices.insert(0, 0); hull.insert(0, values[0])
        if indices[-1] != len(values) - 1:
            indices.append(len(values) - 1); hull.append(values[-1])
        return np.asarray(indices), np.asarray(hull)

    @classmethod
    def embedding(cls, audio: np.ndarray):
        spectrum = np.abs(librosa.stft(audio, n_fft=cls.N_FFT)) ** 2
        curve = np.mean(
            10 * np.log10(np.clip(spectrum, 1e-10, 1e6)), axis=1
        )
        frequencies = np.linspace(0, cls.SR / 2, cls.N_FFT // 2 + 1)
        values = curve[(frequencies > cls.FMIN) & (frequencies < cls.FMAX)]
        indices, hull = cls._lower_hull(values)
        baseline = interpolate.interp1d(
            indices, hull, kind="quadratic"
        )(np.arange(len(values)))
        residual = np.clip(values - np.clip(baseline, -45, None), 0, 5)
        return residual / (1e-6 + residual.max())

    def fake_probability(self, audio: np.ndarray) -> float:
        logit = float(self.embedding(audio) @ self.weight + self.bias)
        return float(1 / (1 + np.exp(-np.clip(logit, -60, 60))))
