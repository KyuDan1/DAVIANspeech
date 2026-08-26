"""PANNs Cnn14 voice / music presence estimation.

Unchanged from the competition baseline -- it already scores ~0.989 on the
public leaderboard, so this stage is deliberately left alone while the
separation and spoof-detection stages are swapped out.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import librosa
import numpy as np
import torch

AUDIO_SR = 16_000
PANNS_SR = 32_000
SEGMENT_SAMPLES = 64_600


def segment_starts(num_samples: int, segment: int = SEGMENT_SAMPLES) -> list[int]:
    """Non-overlapping windows, with the last one snapped to the tail."""
    if num_samples <= segment:
        return [0]
    last = num_samples - segment
    starts = list(range(0, last + 1, segment))
    if starts[-1] != last:
        starts.append(last)
    return starts


def extract_segment(audio: np.ndarray, start: int, segment: int = SEGMENT_SAMPLES):
    if audio.size < segment:
        repeats = segment // audio.size + 1
        return np.tile(audio, repeats)[:segment].astype(np.float32)
    return audio[start:start + segment].astype(np.float32, copy=False)


class PannsPresence:
    """Predicts P(voice present) and P(music present) for a waveform."""

    def __init__(self, model_dir, device="cuda"):
        model_dir = Path(model_dir)

        # panns_inference reads its label CSV from a fixed location.
        labels_csv = model_dir / "class_labels_indices.csv"
        if labels_csv.is_file():
            destination = Path.home() / "panns_data" / "class_labels_indices.csv"
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(labels_csv, destination)

        from panns_inference import AudioTagging, labels

        checkpoint = model_dir / "Cnn14_mAP=0.431.pth"
        self.model = AudioTagging(checkpoint_path=str(checkpoint), device=device)

        groups = json.loads((model_dir / "component_labels.json").read_text("utf-8"))
        index_of = {label: i for i, label in enumerate(labels)}
        self.voice_indices = [index_of[name] for name in groups["voice"]]
        self.music_indices = [index_of[name] for name in groups["music"]]

    def _segments_32k(self, audio: np.ndarray) -> np.ndarray:
        chunks = []
        for start in segment_starts(audio.size):
            chunk = extract_segment(audio, start)
            chunks.append(
                librosa.resample(
                    chunk, orig_sr=AUDIO_SR, target_sr=PANNS_SR, res_type="soxr_hq"
                ).astype(np.float32)
            )
        return np.stack(chunks)

    def predict(self, audio: np.ndarray) -> tuple[float, float]:
        clipwise, _ = self.model.inference(self._segments_32k(audio))
        voice = float(clipwise[:, self.voice_indices].max())
        music = float(clipwise[:, self.music_indices].max())
        return voice, music
