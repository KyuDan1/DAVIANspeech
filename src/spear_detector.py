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
                 window=160_000, max_windows=3):
        self.device = torch.device(device)
        self.model = _load_local_model(Path(model_dir), self.device)
        head = np.load(head_path)
        self.weight = torch.from_numpy(head["weight"]).to(self.device)
        self.bias = torch.as_tensor(head["bias"], device=self.device)
        self.window = window
        self.max_windows = max_windows

    def fake_probability(self, audio: np.ndarray) -> float:
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
        file_embedding = torch.cat(embeddings).mean(dim=0)
        return float(torch.sigmoid(file_embedding @ self.weight + self.bias))
