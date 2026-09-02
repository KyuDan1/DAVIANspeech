"""Train-free AudioSet presence expert based on the general-audio EAT model."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

try:  # package import in tests; flat import in the offline submission
    from .eat_detector import EatMusicDetector, _load_local_model
except ImportError:  # pragma: no cover - exercised by script.py
    from eat_detector import EatMusicDetector, _load_local_model


class EatPresence:
    """Return voice/music AudioSet evidence from up to three temporal views."""

    SAMPLES = EatMusicDetector.SAMPLES

    def __init__(self, model_dir: Path, labels_dir: Path, device: str = "cuda"):
        self.device = torch.device(device)
        self.model = _load_local_model(Path(model_dir), self.device)
        labels_dir = Path(labels_dir)
        labels = pd.read_csv(labels_dir / "class_labels_indices.csv").display_name.tolist()
        index = {label: offset for offset, label in enumerate(labels)}
        groups = json.loads(
            (labels_dir / "component_labels.json").read_text(encoding="utf-8")
        )
        self.voice_indices = torch.tensor(
            [index[label] for label in groups["voice"]], device=self.device
        )
        self.music_indices = torch.tensor(
            [index[label] for label in groups["music"]], device=self.device
        )

    @classmethod
    def temporal_views(cls, audio: np.ndarray) -> list[np.ndarray]:
        if len(audio) <= cls.SAMPLES:
            return [audio]
        last = len(audio) - cls.SAMPLES
        starts = sorted({0, last // 2, last})
        return [audio[start:start + cls.SAMPLES] for start in starts]

    @torch.inference_mode()
    def predict(self, audio: np.ndarray) -> tuple[float, float]:
        features = torch.stack([
            EatMusicDetector._fbank(view) for view in self.temporal_views(audio)
        ])[:, None].to(self.device)
        probabilities = torch.sigmoid(self.model(features))
        voice = probabilities.index_select(1, self.voice_indices).max()
        music = probabilities.index_select(1, self.music_indices).max()
        return float(voice), float(music)


def fuse_presence(
    panns_voice: float,
    panns_music: float,
    eat_voice: float,
    eat_music: float,
    voice_weight: float = 0.30,
    music_weight: float = 0.90,
) -> tuple[float, float]:
    """Conservative score fusion; thresholds are deliberately handled elsewhere."""
    voice = (1 - voice_weight) * panns_voice + voice_weight * eat_voice
    music = (1 - music_weight) * panns_music + music_weight * eat_music
    return float(np.clip(voice, 0, 1)), float(np.clip(music, 0, 1))
