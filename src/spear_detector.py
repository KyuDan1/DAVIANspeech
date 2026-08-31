"""Offline SPEAR v2 expert with a lightweight linear spoof head."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import torch

from presence import extract_segment, segment_starts


def _load_local_model(model_dir: Path, device: torch.device):
    # Loading the custom code as a real package also avoids relying on the
    # Hugging Face module cache, which is not available on the offline grader.
    package = "davianspeech_spear"
    package_spec = importlib.util.spec_from_file_location(
        package, model_dir / "__init__.py",
        submodule_search_locations=[str(model_dir)],
    )
    module = importlib.util.module_from_spec(package_spec)
    sys.modules[package] = module
    model_spec = importlib.util.spec_from_file_location(
        f"{package}.modeling_spear", model_dir / "modeling_spear.py"
    )
    modeling = importlib.util.module_from_spec(model_spec)
    sys.modules[model_spec.name] = modeling
    model_spec.loader.exec_module(modeling)
    return modeling.SpearModel.from_pretrained(model_dir).to(device).eval()


class SpearMusicDetector:
    def __init__(self, model_dir: Path, head_path: Path, device="cuda",
                 window=160_000, max_windows=3, extra_head_paths=()):
        self.device = torch.device(device)
        self.model = _load_local_model(Path(model_dir), self.device)
        head = np.load(head_path)
        self.weight = torch.from_numpy(head["weight"]).to(self.device)
        self.bias = torch.as_tensor(head["bias"], device=self.device)
        self.extra_heads = []
        for path in extra_head_paths:
            extra = np.load(path)
            self.extra_heads.append((
                torch.from_numpy(extra["weight"]).to(self.device),
                torch.as_tensor(extra["bias"], device=self.device),
            ))
        self.window = window
        self.max_windows = max_windows

    def embedding(self, audio: np.ndarray) -> torch.Tensor:
        """Return the file-level 13-layer SPEAR embedding."""
        starts = segment_starts(len(audio), self.window)
        if self.max_windows and len(starts) > self.max_windows:
            indices = np.linspace(0, len(starts) - 1, self.max_windows, dtype=int)
            starts = [starts[index] for index in np.unique(indices)]
        embeddings = []
        for start in starts:
            waveform = torch.from_numpy(
                extract_segment(audio, start, self.window)
            ).unsqueeze(0).to(self.device)
            lengths = torch.tensor([waveform.shape[1]], device=self.device)
            with torch.inference_mode():
                output = self.model(waveform, lengths)
                embeddings.append(torch.cat([
                    hidden.float().mean(dim=1)
                    for hidden in output["hidden_states"]
                ], dim=-1))
        return torch.cat(embeddings).mean(dim=0)

    def fake_probabilities(self, audio: np.ndarray) -> tuple[float, ...]:
        file_embedding = self.embedding(audio)
        heads = [(self.weight, self.bias), *self.extra_heads]
        return tuple(
            float(torch.sigmoid(file_embedding @ weight + bias))
            for weight, bias in heads
        )

    def fake_probability(self, audio: np.ndarray) -> float:
        return self.fake_probabilities(audio)[0]


class SpearCrossComponentDetector(SpearMusicDetector):
    """Joint RR/RF/FR/FF probe plus a generator-robust music expert."""

    def __init__(self, model_dir: Path, music_head_path: Path,
                 joint_head_path: Path, device="cuda", window=160_000,
                 max_windows=3):
        super().__init__(
            model_dir, music_head_path, device=device, window=window,
            max_windows=max_windows,
        )
        joint = np.load(joint_head_path)
        self.joint_mean = torch.from_numpy(joint["mean"]).to(self.device).squeeze(0)
        self.joint_std = torch.from_numpy(joint["std"]).to(self.device).squeeze(0)
        self.joint_weight = torch.from_numpy(joint["joint_weight"]).to(self.device)
        self.joint_bias = torch.from_numpy(joint["joint_bias"]).to(self.device)
        self.layers = self.joint_mean.shape[0]
        self.dimension = self.joint_mean.shape[1]

    def component_probabilities(self, audio: np.ndarray,
                                layer: int = 2) -> dict[str, float]:
        if not 0 <= layer < self.layers:
            raise ValueError(f"layer must be between 0 and {self.layers - 1}")
        embedding = self.embedding(audio)
        music_probability = torch.sigmoid(embedding @ self.weight + self.bias)
        hidden = embedding.reshape(self.layers, self.dimension)
        normalized = (hidden - self.joint_mean) / self.joint_std
        logits = normalized[layer] @ self.joint_weight[layer] + self.joint_bias[layer]
        probabilities = torch.softmax(logits, dim=-1)
        return {
            "file": float(1.0 - probabilities[0]),
            "voice": float(probabilities[2] + probabilities[3]),
            "music": float(probabilities[1] + probabilities[3]),
            "music_expert": float(music_probability),
        }


def fuse_cross_component_scores(
    file_probability: float, music_probability: float,
    joint_file_probability: float, music_expert_probability: float,
    weight: float = 0.10,
) -> tuple[float, float]:
    """Conservatively add joint/file and music evidence to an anchor model."""
    if not 0.0 <= weight <= 1.0:
        raise ValueError("weight must lie in [0, 1]")
    file_fused = (1.0 - weight) * file_probability + weight * joint_file_probability
    music_fused = (
        (1.0 - weight) * music_probability + weight * music_expert_probability
    )
    return float(np.clip(file_fused, 0.0, 1.0)), float(
        np.clip(music_fused, 0.0, 1.0)
    )
