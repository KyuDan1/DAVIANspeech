"""Train-free AudioSet presence expert based on the general-audio EAT model."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

try:  # package import in tests; flat import in the offline submission
    from .eat_detector import EatMusicDetector, _load_local_model
    from .dual_domain_stats import sequence_statistics
except ImportError:  # pragma: no cover - exercised by script.py
    from eat_detector import EatMusicDetector, _load_local_model
    from dual_domain_stats import sequence_statistics


class EatPresence:
    """Return voice/music AudioSet evidence from up to three temporal views."""

    SAMPLES = EatMusicDetector.SAMPLES

    def __init__(
        self, model_dir: Path, labels_dir: Path, device: str = "cuda",
        presence_head_path: Path | None = None,
    ):
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
        self.probe_mean = self.probe_std = None
        self.probe_music_weight = self.probe_music_bias = None
        if presence_head_path is not None:
            checkpoint = np.load(presence_head_path)
            self.probe_mean = torch.from_numpy(checkpoint["mean"]).to(self.device)
            self.probe_std = torch.from_numpy(checkpoint["std"]).to(self.device)
            self.probe_music_weight = torch.from_numpy(
                checkpoint["music_coefficient"].reshape(-1)
            ).to(self.device)
            self.probe_music_bias = torch.as_tensor(
                checkpoint["music_intercept"].item(), device=self.device
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
        voice, music, _ = self.predict_with_probe(audio)
        return voice, music

    @torch.inference_mode()
    def predict_with_probe(self, audio: np.ndarray) -> tuple[float, float, float | None]:
        """Return AudioSet evidence and optional frozen latent music probe."""
        features = torch.stack([
            EatMusicDetector._fbank(view) for view in self.temporal_views(audio)
        ])[:, None].to(self.device)
        tokens = self.model.extract_features(features)
        embedding = self.model.model.fc_norm(tokens[:, 0])
        probabilities = torch.sigmoid(self.model.model.head(embedding))
        voice = probabilities.index_select(1, self.voice_indices).max()
        music = probabilities.index_select(1, self.music_indices).max()
        probe = None
        if self.probe_music_weight is not None:
            # Match the leakage-guarded training cache: per-view token
            # statistics were stored as float16 before view aggregation.
            stats = sequence_statistics(tokens[:, 1:]).half().float()
            features = torch.cat((stats.mean(dim=0), stats.max(dim=0).values)).flatten()
            features = ((features - self.probe_mean) / self.probe_std).clamp_(-8, 8)
            logit = features @ self.probe_music_weight + self.probe_music_bias
            # Temperature avoids float saturation/ties while preserving rank.
            probe = float(torch.sigmoid(logit / 5.0))
        return float(voice), float(music), probe


def fuse_presence(
    panns_voice: float,
    panns_music: float,
    eat_voice: float,
    eat_music: float,
    voice_weight: float = 0.35,
    music_weight: float = 0.90,
) -> tuple[float, float]:
    """Conservative score fusion; thresholds are deliberately handled elsewhere."""
    voice = (1 - voice_weight) * panns_voice + voice_weight * eat_voice
    music = (1 - music_weight) * panns_music + music_weight * eat_music
    return float(np.clip(voice, 0, 1)), float(np.clip(music, 0, 1))


def fuse_music_probe(base_music: float, probe_music: float,
                     probe_weight: float = 0.40) -> float:
    """Blend content evidence with a source-disjoint latent presence probe."""
    result = (1 - probe_weight) * base_music + probe_weight * probe_music
    return float(np.clip(result, 0, 1))
