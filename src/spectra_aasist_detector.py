"""Offline Spectra-AASIST scoring and conservative v18 fusion.

The model sees the HTDemucs vocal stem already produced by the legacy path.
It therefore adds a genuinely different Wav2Vec2-300M+AASIST decision head
without running source separation twice or modifying the music prediction.
"""

from __future__ import annotations

import csv
import importlib.util
import math
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file


def _logit(value: np.ndarray) -> np.ndarray:
    value = np.clip(np.asarray(value, dtype=np.float64), 1e-5, 1 - 1e-5)
    return np.log(value) - np.log1p(-value)


def _sigmoid(value: np.ndarray) -> np.ndarray:
    return np.exp(-np.logaddexp(0.0, -np.asarray(value, dtype=np.float64)))


def _fuse_margin(anchor: float, margin: float, weight: float) -> float:
    mixed = (1 - weight) * _logit(np.asarray([anchor])) + weight * margin
    return float(_sigmoid(mixed)[0])


def _fixed_windows(audio: np.ndarray, length: int, count: int) -> np.ndarray:
    audio = np.asarray(audio, dtype=np.float32)
    if not len(audio):
        audio = np.zeros(1, dtype=np.float32)
    if len(audio) < length:
        audio = np.tile(audio, math.ceil(length / len(audio)))[:length]
        return np.repeat(audio[None], count, axis=0)
    if count == 1:
        starts = np.asarray([(len(audio) - length) // 2])
    else:
        starts = np.rint(np.linspace(0, len(audio) - length, count)).astype(int)
    return np.stack([audio[start:start + length] for start in starts])


def _preemphasis(waveforms: torch.Tensor, coefficient: float = 0.97) -> torch.Tensor:
    output = waveforms.clone()
    output[:, 1:] = waveforms[:, 1:] - coefficient * waveforms[:, :-1]
    return output


def _load_model(model_dir: Path, device: torch.device) -> torch.nn.Module:
    spec = importlib.util.spec_from_file_location(
        "vendored_spectra_aasist", model_dir / "model.py"
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load Spectra-AASIST from {model_dir}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    model = module.SpectraAASIST()
    missing, unexpected = model.load_state_dict(
        load_file(model_dir / "model.safetensors"), strict=False
    )
    if missing or unexpected:
        raise RuntimeError(
            f"Spectra checkpoint mismatch: missing={missing[:5]}, "
            f"unexpected={unexpected[:5]}"
        )
    return model.eval().to(device)


class SpectraStemScorer:
    """Batch deterministic vocal-stem windows during the main inference loop."""

    def __init__(
        self, model_dir: Path, device: str = "cuda", windows: int = 3,
        file_batch_size: int = 4, window_samples: int = 64_600,
        silence_rms: float = 1e-5,
    ) -> None:
        self.device = torch.device(device)
        self.model = _load_model(Path(model_dir), self.device)
        self.windows = int(windows)
        self.file_batch_size = int(file_batch_size)
        self.window_samples = int(window_samples)
        self.silence_rms = float(silence_rms)
        self.pending: list[tuple[str, np.ndarray, bool]] = []
        self.ids: list[str] = []
        self.margins: list[float] = []
        self.valid: list[bool] = []

    def add(self, item_id: str, voice_audio: np.ndarray) -> None:
        rms = float(np.sqrt(np.mean(
            np.square(voice_audio, dtype=np.float64)
        ))) if voice_audio.size else 0.0
        valid = rms >= self.silence_rms
        self.pending.append((
            str(item_id),
            _fixed_windows(voice_audio, self.window_samples, self.windows),
            valid,
        ))
        if len(self.pending) >= self.file_batch_size:
            self._flush()

    @torch.inference_mode()
    def _flush(self) -> None:
        if not self.pending:
            return
        waveforms = torch.from_numpy(np.concatenate([
            windows for _, windows, _ in self.pending
        ])).to(self.device)
        waveforms = _preemphasis(waveforms)
        with torch.autocast(
            device_type=self.device.type, dtype=torch.bfloat16,
            enabled=self.device.type == "cuda",
        ):
            logits = self.model(waveforms).float().reshape(
                len(self.pending), self.windows, 2
            )
        margins = (logits[:, :, 0] - logits[:, :, 1]).mean(dim=1).cpu().numpy()
        for (item_id, _, valid), margin in zip(self.pending, margins):
            self.ids.append(item_id)
            self.margins.append(float(margin))
            self.valid.append(valid)
        self.pending.clear()

    def save(self, output_path: Path) -> None:
        self._flush()
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            output_path,
            ids=np.asarray(self.ids),
            fake_margin=np.asarray(self.margins, dtype=np.float32),
            valid=np.asarray(self.valid, dtype=np.bool_),
        )


def apply_spectra_voice_fusion(
    submission_path: Path,
    statistics_path: Path,
    voice_weight: float = 0.10,
    file_weight: float = 0.05,
    voice_presence_gate: float = 0.50,
) -> None:
    """Fuse the stem expert after v18 without touching Music probabilities.

    Voice is evaluated only on voice-present rows, so every valid vocal stem
    receives the small ensemble vote.  File receives it only when the existing
    v18 scores say Voice is present and at least as suspicious as Music.  This
    avoids suppressing the real-voice/fake-music cell.
    """
    values = np.load(statistics_path, allow_pickle=False)
    score_by_id = {
        str(item): (float(score), bool(valid))
        for item, score, valid in zip(
            values["ids"], values["fake_margin"], values["valid"]
        )
    }
    with Path(submission_path).open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        columns = list(reader.fieldnames or [])
        rows = list(reader)
    if {row["ID"] for row in rows} != set(score_by_id):
        raise ValueError("Submission/Spectra statistic IDs differ")
    for row in rows:
        margin, valid = score_by_id[row["ID"]]
        if not valid:
            continue
        old_voice = float(row["VOICE_FAKE_PROB"])
        old_music = float(row["MUSIC_FAKE_PROB"])
        if (
            float(row["VOICE_PRESENT_PROB"]) >= voice_presence_gate
            and old_voice >= old_music
            and file_weight > 0
        ):
            row["FILE_FAKE_PROB"] = round(_fuse_margin(
                float(row["FILE_FAKE_PROB"]), margin, file_weight
            ), 10)
        if voice_weight > 0:
            row["VOICE_FAKE_PROB"] = round(
                _fuse_margin(old_voice, margin, voice_weight), 10
            )
    temporary = Path(submission_path).with_suffix(".tmp.csv")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(submission_path)
