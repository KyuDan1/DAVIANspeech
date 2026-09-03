"""Train-free AudioSet presence expert based on the general-audio EAT model."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F

try:  # package import in tests; flat import in the offline submission
    from .eat_detector import EatMusicDetector, _load_local_model
    from .dual_domain_stats import (
        crop_or_pad, pad_views, segment_starts, sequence_statistics,
        temporal_starts,
    )
    from .eat_hierarchical import hierarchical_statistics
except ImportError:  # pragma: no cover - exercised by script.py
    from eat_detector import EatMusicDetector, _load_local_model
    from dual_domain_stats import (
        crop_or_pad, pad_views, segment_starts, sequence_statistics,
        temporal_starts,
    )
    from eat_hierarchical import hierarchical_statistics


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
        self.last_statistics: tuple[np.ndarray, np.ndarray] | None = None
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
    def predict_audio_set(self, audio: np.ndarray) -> tuple[float, float]:
        """Preserve the established per-file AudioSet presence computation."""
        features = torch.stack([
            EatMusicDetector._fbank(view) for view in self.temporal_views(audio)
        ])[:, None].to(self.device)
        probabilities = torch.sigmoid(self.model(features))
        voice = probabilities.index_select(1, self.voice_indices).max()
        music = probabilities.index_select(1, self.music_indices).max()
        return float(voice), float(music)

    def probe_from_statistics(self, matrix: np.ndarray, mask: np.ndarray) -> float | None:
        if self.probe_music_weight is None:
            return None
        values = matrix.astype(np.float32)
        valid = mask[:, None, None]
        view_mean = (values * valid).sum(axis=0) / max(int(mask.sum()), 1)
        view_max = np.where(valid, values, -np.inf).max(axis=0)
        features = torch.from_numpy(
            np.concatenate((view_mean, view_max), axis=0).reshape(-1)
        ).to(self.device)
        features = ((features - self.probe_mean) / self.probe_std).clamp_(-8, 8)
        logit = features @ self.probe_music_weight + self.probe_music_bias
        return float(torch.sigmoid(logit / 5.0))

    @torch.inference_mode()
    def latent_statistics_batch(
        self,
        audios: list[np.ndarray],
        hierarchical_projection: torch.Tensor | None = None,
        segmental_max_views: int | None = None,
    ) -> list[tuple]:
        """Batch endpoint and optional long-range views in one encoder pass."""
        if segmental_max_views is not None and segmental_max_views <= 0:
            raise ValueError("segmental_max_views must be positive")
        if segmental_max_views is not None and hierarchical_projection is None:
            raise ValueError("segmental statistics require a projection")
        grouped_views = []
        view_plans = []
        for audio in audios:
            endpoint_starts = temporal_starts(len(audio), self.SAMPLES, 3)
            long_starts = (
                segment_starts(len(audio), self.SAMPLES, segmental_max_views)
                if segmental_max_views is not None else endpoint_starts
            )
            starts = sorted(set(endpoint_starts) | set(long_starts))
            by_start = {start: index for index, start in enumerate(starts)}
            view_plans.append((
                [by_start[start] for start in endpoint_starts],
                [by_start[start] for start in long_starts],
                len(starts),
            ))
            grouped_views.append([
                crop_or_pad(audio, start, self.SAMPLES) for start in starts
            ])
        features = torch.stack([
            EatMusicDetector._fbank(view)
            for views in grouped_views for view in views
        ])[:, None].to(self.device)
        hierarchical = None
        if hierarchical_projection is None:
            tokens = self.model.extract_features(features)[:, 1:]
            statistics = sequence_statistics(tokens)
        else:
            all_statistics = hierarchical_statistics(self.model, features)
            statistics = all_statistics[:, -1, :4]
            projection = hierarchical_projection.to(
                device=features.device, dtype=torch.float32
            )
            if projection.ndim != 2 or projection.shape[0] != all_statistics.shape[-1]:
                raise ValueError("hierarchical EAT projection has incompatible shape")
            hierarchical = F.layer_norm(
                all_statistics, (all_statistics.shape[-1],)
            ) @ projection
        result, offset = [], 0
        for endpoint_indices, long_indices, count in view_plans:
            matrix, mask = pad_views(
                [statistics[offset + index] for index in endpoint_indices],
                3, (4, 768),
            )
            item = (matrix, mask, self.probe_from_statistics(matrix, mask))
            if hierarchical is not None:
                hierarchical_matrix, hierarchical_mask = pad_views(
                    [hierarchical[offset + index] for index in endpoint_indices],
                    3, tuple(hierarchical.shape[1:]),
                )
                if not np.array_equal(mask, hierarchical_mask):
                    raise RuntimeError("EAT hierarchical view mask mismatch")
                item = (*item, hierarchical_matrix)
                if segmental_max_views is not None:
                    segmental_matrix, segmental_mask = pad_views(
                        [hierarchical[offset + index] for index in long_indices],
                        segmental_max_views, tuple(hierarchical.shape[1:]),
                    )
                    item = (*item, segmental_matrix, segmental_mask)
            result.append(item)
            offset += count
        return result

    @torch.inference_mode()
    def predict(self, audio: np.ndarray) -> tuple[float, float]:
        return self.predict_audio_set(audio)

    @torch.inference_mode()
    def predict_with_probe(self, audio: np.ndarray) -> tuple[float, float, float | None]:
        """Return AudioSet evidence and optional frozen latent music probe."""
        voice, music = self.predict_audio_set(audio)
        self.last_statistics = None
        probe = None
        if self.probe_music_weight is not None:
            matrix, mask, probe = self.latent_statistics_batch([audio])[0]
            self.last_statistics = (matrix, mask)
        return voice, music, probe


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
