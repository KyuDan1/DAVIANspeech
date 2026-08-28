"""Inference wrapper for the CC BY-NC 4.0 ArtifactNet v9.4 ONNX build."""

from __future__ import annotations

from pathlib import Path

import librosa
import numpy as np


class ArtifactNetMusicDetector:
    SAMPLE_RATE = 44_100
    WINDOW = 4 * SAMPLE_RATE

    def __init__(self, model_dir, providers=None):
        import onnxruntime as ort

        model_path = Path(model_dir) / "artifactnet_v94_full.onnx"
        self.session = ort.InferenceSession(
            str(model_path), providers=providers or [
                "CUDAExecutionProvider", "CPUExecutionProvider"
            ]
        )
        self.input_name = self.session.get_inputs()[0].name

    @staticmethod
    def _windows(audio: np.ndarray) -> np.ndarray:
        window = ArtifactNetMusicDetector.WINDOW
        if len(audio) < window:
            audio = np.pad(audio, (0, window - len(audio)))
        starts = list(range(0, max(1, len(audio) - window + 1), window))
        tail = len(audio) - window
        if starts[-1] != tail:
            starts.append(tail)
        return np.stack([audio[start:start + window] for start in starts]).astype(np.float32)

    def fake_probability(self, audio: np.ndarray, source_sr=16_000) -> float:
        if source_sr != self.SAMPLE_RATE:
            audio = librosa.resample(
                audio, orig_sr=source_sr, target_sr=self.SAMPLE_RATE, res_type="soxr_hq"
            )
        # The released graph exposes a dynamic batch axis but contains a
        # reshape that is only valid for batch size one. Run chunks singly.
        probabilities = [
            float(np.asarray(self.session.run(
                None, {self.input_name: window[None]}
            )[0]).reshape(-1)[0])
            for window in self._windows(audio)
        ]
        probabilities = [value if np.isfinite(value) else 0.5 for value in probabilities]
        return float(np.median(probabilities))
