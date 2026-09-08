"""Exact inference reimplementation of the public Suno/GTZAN hybrid model.

The MIT checkpoint was trained on 10-second, 22.05-kHz log-mel inputs.  This
module intentionally exposes every non-overlapping/tail window instead of
silently truncating a competition file to its first ten seconds, allowing a
file-level MIL readout to be audited separately.
"""
from __future__ import annotations

from pathlib import Path
import math

import librosa
import numpy as np
import torch
from torch import nn


class AudioEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        layers = []
        in_channels = 1
        for out_channels in (32, 64, 128, 256):
            layers.extend((
                nn.Conv2d(in_channels, out_channels, 3, stride=2, padding=1),
                nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True),
                nn.Dropout2d(0.2),
            ))
            in_channels = out_channels
        self.encoder = nn.Sequential(*layers)

    def forward(self, values):
        return self.encoder(values)


class AudioDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        layers = []
        for in_channels, out_channels in zip((256, 128, 64), (128, 64, 32)):
            layers.extend((
                nn.ConvTranspose2d(in_channels, out_channels, 4, stride=2, padding=1),
                nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True),
                nn.Dropout2d(0.2),
            ))
        layers.extend((nn.ConvTranspose2d(32, 1, 4, stride=2, padding=1), nn.Tanh()))
        self.decoder = nn.Sequential(*layers)

    def forward(self, values):
        return self.decoder(values)


class Autoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = AudioEncoder()
        self.decoder = AudioDecoder()

    def forward(self, values):
        encoded = self.encoder(values)
        return self.decoder(encoded), encoded


class PositionalEncoding(nn.Module):
    def __init__(self, width=512, max_length=5000):
        super().__init__()
        values = torch.zeros(max_length, width)
        position = torch.arange(max_length, dtype=torch.float32).unsqueeze(1)
        scale = torch.exp(torch.arange(0, width, 2).float() * (-math.log(10_000.0) / width))
        values[:, 0::2] = torch.sin(position * scale)
        values[:, 1::2] = torch.cos(position * scale)
        self.register_buffer("pe", values.unsqueeze(0))

    def forward(self, values):
        return values + self.pe[:, :values.shape[1]]


class TransformerEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_projection = nn.Linear(256, 512)
        self.pos_encoder = PositionalEncoding()
        layer = nn.TransformerEncoderLayer(
            d_model=512, nhead=8, dim_feedforward=2048, dropout=0.1,
            activation="gelu", batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=6)
        self.dropout = nn.Dropout(0.1)

    def forward(self, values):
        batch, channels, height, width = values.shape
        values = values.permute(0, 2, 3, 1).reshape(batch, height * width, channels)
        return self.transformer(self.dropout(self.pos_encoder(self.input_projection(values))))


class HybridSunoModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.autoencoder = Autoencoder()
        self.transformer = TransformerEncoder()
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.fusion = nn.Linear(512, 768)
        blocks = []
        in_features = 768
        for out_features in (512, 256, 128):
            blocks.extend((nn.Linear(in_features, out_features), nn.ReLU(inplace=True),
                           nn.Dropout(0.3), nn.BatchNorm1d(out_features)))
            in_features = out_features
        blocks.append(nn.Linear(in_features, 2))
        self.classifier = nn.Sequential(*blocks)

    def forward(self, values):
        _, encoded = self.autoencoder(values)
        tokens = self.transformer(encoded).permute(0, 2, 1)
        return self.classifier(self.fusion(self.global_pool(tokens).squeeze(-1)))


class HybridSunoDetector:
    SAMPLE_RATE = 22_050
    WINDOW = 220_500

    def __init__(self, checkpoint: Path, device="cuda"):
        state = torch.load(checkpoint, map_location="cpu", weights_only=False)
        self.model = HybridSunoModel()
        self.model.load_state_dict(state["model_state_dict"], strict=True)
        self.model.to(device).eval().requires_grad_(False)
        self.device = torch.device(device)

    @classmethod
    def waveform_windows(cls, audio: np.ndarray, sample_rate=16_000) -> np.ndarray:
        values = librosa.resample(np.asarray(audio, dtype=np.float32),
                                  orig_sr=sample_rate, target_sr=cls.SAMPLE_RATE)
        if len(values) < cls.WINDOW:
            values = np.pad(values, (0, cls.WINDOW - len(values)))
        starts = list(range(0, max(1, len(values) - cls.WINDOW + 1), cls.WINDOW))
        tail = len(values) - cls.WINDOW
        if starts[-1] != tail:
            starts.append(tail)
        windows = np.stack([values[start:start + cls.WINDOW] for start in starts])
        peaks = np.max(np.abs(windows), axis=1, keepdims=True)
        return windows / np.maximum(peaks, 1e-12)

    @classmethod
    def mel(cls, windows: np.ndarray) -> np.ndarray:
        result = []
        for window in windows:
            power = librosa.feature.melspectrogram(
                y=window, sr=cls.SAMPLE_RATE, n_fft=2048, hop_length=512,
                n_mels=128, fmin=0, fmax=8000,
            )
            result.append(librosa.power_to_db(power, ref=np.max))
        return np.asarray(result, dtype=np.float32)[:, None]

    @torch.inference_mode()
    def window_probabilities(self, audio: np.ndarray, sample_rate=16_000,
                             batch_size=8) -> np.ndarray:
        features = self.mel(self.waveform_windows(audio, sample_rate))
        probabilities = []
        for start in range(0, len(features), batch_size):
            logits = self.model(torch.from_numpy(features[start:start + batch_size]).to(self.device))
            probabilities.append(logits.softmax(-1)[:, 1].cpu().numpy())
        return np.concatenate(probabilities).astype(np.float64)


def lme_probability(probabilities: np.ndarray, temperature=5.0) -> float:
    probabilities = np.clip(np.asarray(probabilities, dtype=np.float64), 1e-6, 1 - 1e-6)
    logits = np.log(probabilities) - np.log1p(-probabilities)
    maximum = logits.max()
    aggregate = maximum + math.log(np.exp((logits - maximum) * temperature).mean()) / temperature
    return float(1 / (1 + math.exp(-aggregate)))
