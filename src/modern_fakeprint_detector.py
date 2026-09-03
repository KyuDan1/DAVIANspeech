"""Lightweight Suno/Udio fakeprint detector on original 16 kHz mixtures.

The feature path reproduces the public ``lofcz/ai-music-detector`` release
without importing torchaudio.  This matters on the DACON image, where replacing
the preinstalled torch/torchaudio pair has previously broken ``libtorchaudio``.
"""

from __future__ import annotations

from pathlib import Path
import csv

import librosa
import numpy as np
from scipy.ndimage import minimum_filter1d


class ModernFakeprintDetector:
    SAMPLE_RATE = 16_000
    N_FFT = 8_192
    HOP_LENGTH = 4_096
    FREQ_MIN = 1_000
    FREQ_MAX = 8_000
    HULL_AREA = 10
    MIN_DB = -45.0
    MAX_DB = 5.0

    def __init__(self, model_path: Path):
        model_path = Path(model_path)
        self.session = None
        self.weight = None
        self.bias = None
        if model_path.suffix == ".npz":
            checkpoint = np.load(model_path)
            self.weight = np.asarray(checkpoint["weights"], dtype=np.float64).reshape(-1)
            self.bias = float(np.asarray(checkpoint["bias"]).reshape(-1)[0])
            self.feature_count = int(self.weight.size)
        else:
            import onnxruntime as ort

            self.session = ort.InferenceSession(
                str(model_path), providers=["CPUExecutionProvider"]
            )
            model_input = self.session.get_inputs()[0]
            self.input_name = model_input.name
            self.output_name = self.session.get_outputs()[0].name
            self.feature_count = int(model_input.shape[1])

    @classmethod
    def embedding(cls, audio: np.ndarray) -> np.ndarray:
        audio = np.asarray(audio, dtype=np.float32)
        if audio.ndim != 1 or not audio.size or not np.isfinite(audio).all():
            raise ValueError("audio must be a finite non-empty mono waveform")
        spectrum = np.abs(librosa.stft(
            audio, n_fft=cls.N_FFT, hop_length=cls.HOP_LENGTH,
            win_length=cls.N_FFT, window="hann", center=True,
            pad_mode="reflect",
        )) ** 2
        mean_spectrum = (10.0 * np.log10(
            np.clip(spectrum, 1e-10, 1e6)
        )).mean(axis=1)
        frequencies = np.linspace(
            0.0, cls.SAMPLE_RATE / 2, cls.N_FFT // 2 + 1
        )
        selected = mean_spectrum[
            (frequencies >= cls.FREQ_MIN) & (frequencies <= cls.FREQ_MAX)
        ]
        hull = minimum_filter1d(
            selected, size=cls.HULL_AREA, mode="nearest"
        )
        hull = np.clip(hull, cls.MIN_DB, None)
        residual = np.clip(selected - hull, 0.0, cls.MAX_DB)
        return (residual / (residual.max() + 1e-6)).astype(np.float32)

    def fake_margin(self, audio: np.ndarray) -> float:
        feature = self.embedding(audio)
        if feature.size != self.feature_count:
            raise ValueError(
                f"expected {self.feature_count} features, got {feature.size}"
            )
        if self.weight is not None:
            margin = float(feature.astype(np.float64) @ self.weight + self.bias)
        else:
            output = self.session.run(
                [self.output_name], {self.input_name: feature[None]}
            )[0]
            probability = float(np.asarray(output).reshape(-1)[0])
            clipped = np.clip(probability, 1e-15, 1 - 1e-15)
            margin = float(np.log(clipped) - np.log1p(-clipped))
        if not np.isfinite(margin):
            raise ValueError("fakeprint model returned a non-finite margin")
        return margin

    def fake_probability(self, audio: np.ndarray) -> float:
        margin = self.fake_margin(audio)
        probability = float(
            np.exp(-np.logaddexp(0.0, -np.clip(margin, -700.0, 700.0)))
        )
        if not np.isfinite(probability):
            raise ValueError("fakeprint model returned a non-finite score")
        return float(np.clip(probability, 0.0, 1.0))


def _logit(value: float) -> float:
    clipped = float(np.clip(value, 1e-5, 1 - 1e-5))
    return float(np.log(clipped) - np.log1p(-clipped))


def _blend(anchor: float, expert: float, weight: float) -> float:
    value = (1 - weight) * _logit(anchor) + weight * _logit(expert)
    return float(np.exp(-np.logaddexp(0.0, -value)))


def apply_modern_fakeprint_fusion(
    test_dir: Path,
    submission_path: Path,
    weights_path: Path,
    file_weight: float = 0.025,
    music_weight: float = 0.025,
) -> None:
    """Add low-weight original-mixture evidence to File and Music scores."""
    detector = ModernFakeprintDetector(weights_path)
    with submission_path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        columns = list(reader.fieldnames or [])
        rows = list(reader)
    paths = {
        path.stem: path for path in Path(test_dir).iterdir()
        if path.is_file()
    }
    for row in rows:
        path = paths.get(row["ID"])
        if path is None:
            raise FileNotFoundError(f"No audio for {row['ID']}")
        audio, _ = librosa.load(
            path, sr=detector.SAMPLE_RATE, mono=True, dtype=np.float32
        )
        score = detector.fake_probability(audio)
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
