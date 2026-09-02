"""Offline XLSR-SLS ONNX inference for complementary speech spoof evidence."""

from __future__ import annotations

from pathlib import Path

import numpy as np


SAMPLE_RATE = 16_000
WINDOW_SAMPLES = 64_600


def segment_starts(num_samples: int, window: int = WINDOW_SAMPLES) -> list[int]:
    """Return deterministic non-overlapping windows with one tail-aligned view."""
    if num_samples <= 0:
        raise ValueError("audio must contain at least one sample")
    if window <= 0:
        raise ValueError("window must be positive")
    if num_samples <= window:
        return [0]
    last = num_samples - window
    starts = list(range(0, last + 1, window))
    if starts[-1] != last:
        starts.append(last)
    return starts


def fixed_window(audio: np.ndarray, start: int, window: int = WINDOW_SAMPLES) -> np.ndarray:
    """Crop a fixed view, tile-repeating short audio as used by XLSR-SLS."""
    audio = np.asarray(audio, dtype=np.float32)
    if audio.size == 0:
        raise ValueError("audio must contain at least one sample")
    if audio.size < window:
        repeats = window // audio.size + 1
        return np.tile(audio, repeats)[:window].astype(np.float32, copy=False)
    return audio[start:start + window].astype(np.float32, copy=False)


def logmeanexp_probability(scores: np.ndarray, temperature: float = 5.0) -> float:
    """Length-stable soft maximum matching the verified XLS-R anchor pooling."""
    values = np.asarray(scores, dtype=np.float64)
    if values.size == 0:
        raise ValueError("at least one score is required")
    scaled = temperature * np.clip(values, 0.0, 1.0)
    peak = scaled.max()
    return float((peak + np.log(np.mean(np.exp(scaled - peak)))) / temperature)


class XlsrSLSDetector:
    """Score 4.04-second raw-waveform windows with the released ONNX graph."""

    def __init__(self, model_path: Path, device: str = "cuda", batch_size: int = 8):
        import onnxruntime as ort

        self.batch_size = batch_size
        available = set(ort.get_available_providers())
        providers = ["CPUExecutionProvider"]
        if device.startswith("cuda") and "CUDAExecutionProvider" in available:
            providers.insert(0, "CUDAExecutionProvider")
        self.session = ort.InferenceSession(str(model_path), providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

    @staticmethod
    def _softmax(logits: np.ndarray) -> np.ndarray:
        shifted = logits.astype(np.float64) - logits.max(axis=-1, keepdims=True)
        values = np.exp(shifted)
        return values / values.sum(axis=-1, keepdims=True)

    def window_probabilities(self, audio: np.ndarray) -> np.ndarray:
        """Return P(fake) for every deterministic view of one file."""
        starts = segment_starts(len(audio))
        views = np.stack([fixed_window(audio, start) for start in starts])
        probabilities = []
        for offset in range(0, len(views), self.batch_size):
            logits = self.session.run(
                [self.output_name],
                {self.input_name: views[offset:offset + self.batch_size]},
            )[0]
            # The released class order is [spoof, bona fide].
            probabilities.extend(self._softmax(np.asarray(logits))[:, 0].tolist())
        result = np.asarray(probabilities, dtype=np.float64)
        if not np.isfinite(result).all():
            raise ValueError("XLSR-SLS returned a non-finite score")
        return result
