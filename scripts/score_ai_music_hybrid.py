#!/usr/bin/env python3
"""Audit the MIT Suno/GTZAN autoencoder-transformer music checkpoint."""

from __future__ import annotations

import argparse
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import yaml


AUDIO_EXTENSIONS = {".aac", ".flac", ".m4a", ".mp3", ".ogg", ".opus", ".wav", ".wma"}


class AudioEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        layers = []
        in_channels = config["autoencoder"]["input_channels"]
        for out_channels in config["autoencoder"]["encoder_channels"]:
            layers.extend((
                nn.Conv2d(in_channels, out_channels, 3, 2, 1),
                nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True),
                nn.Dropout2d(config["autoencoder"]["dropout"]),
            ))
            in_channels = out_channels
        self.encoder = nn.Sequential(*layers)

    def forward(self, values):
        return self.encoder(values)


class AudioDecoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        channels = config["autoencoder"]["decoder_channels"]
        layers = []
        for source, target in zip(channels[:-1], channels[1:]):
            layers.extend((
                nn.ConvTranspose2d(source, target, 4, 2, 1),
                nn.BatchNorm2d(target), nn.ReLU(inplace=True),
                nn.Dropout2d(config["autoencoder"]["dropout"]),
            ))
        layers.extend((
            nn.ConvTranspose2d(
                channels[-1], config["autoencoder"]["input_channels"], 4, 2, 1,
            ),
            nn.Tanh(),
        ))
        self.decoder = nn.Sequential(*layers)

    def forward(self, values):
        return self.decoder(values)


class Autoencoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.encoder = AudioEncoder(config)
        self.decoder = AudioDecoder(config)

    def forward(self, values):
        encoded = self.encoder(values)
        return self.decoder(encoded), encoded


class PositionalEncoding(nn.Module):
    def __init__(self, dimension: int, max_length: int = 5_000):
        super().__init__()
        position = torch.arange(max_length, dtype=torch.float32).unsqueeze(1)
        scale = torch.exp(
            torch.arange(0, dimension, 2, dtype=torch.float32)
            * (-np.log(10_000.0) / dimension)
        )
        encoding = torch.zeros(max_length, dimension)
        encoding[:, 0::2] = torch.sin(position * scale)
        encoding[:, 1::2] = torch.cos(position * scale)
        self.register_buffer("pe", encoding.unsqueeze(0))

    def forward(self, values):
        return values + self.pe[:, :values.shape[1]]


class TransformerEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        values = config["transformer"]
        dimension = values["d_model"]
        self.input_projection = nn.Linear(256, dimension)
        self.pos_encoder = PositionalEncoding(dimension)
        layer = nn.TransformerEncoderLayer(
            d_model=dimension, nhead=values["nhead"],
            dim_feedforward=values["dim_feedforward"],
            dropout=values["dropout"], activation="gelu", batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, values["num_layers"])
        self.dropout = nn.Dropout(values["dropout"])

    def forward(self, values):
        batch, channels, height, width = values.shape
        values = values.permute(0, 2, 3, 1).reshape(
            batch, height * width, channels,
        )
        values = self.dropout(self.pos_encoder(self.input_projection(values)))
        return self.transformer(values)


class HybridModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.autoencoder = Autoencoder(config)
        self.transformer = TransformerEncoder(config)
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        dimension = config["transformer"]["d_model"]
        fusion = config["hybrid"]["fusion_dim"]
        self.fusion = nn.Linear(dimension, fusion)
        layers = []
        source = fusion
        for target in config["hybrid"]["classifier_hidden"]:
            layers.extend((
                nn.Linear(source, target), nn.ReLU(inplace=True),
                nn.Dropout(.3), nn.BatchNorm1d(target),
            ))
            source = target
        layers.append(nn.Linear(source, config["hybrid"]["num_classes"]))
        self.classifier = nn.Sequential(*layers)

    def forward(self, values):
        encoded = self.autoencoder.encoder(values)
        sequence = self.transformer(encoded).permute(0, 2, 1)
        pooled = self.global_pool(sequence).squeeze(-1)
        return self.classifier(self.fusion(pooled))


class MelDataset(Dataset):
    def __init__(self, paths: list[Path], config):
        self.paths = paths
        self.config = config

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths[index]
        config = self.config["audio"]
        sample_rate, duration = config["sample_rate"], config["duration"]
        audio, _ = librosa.load(path, sr=sample_rate, duration=duration, mono=True)
        target = sample_rate * duration
        audio = np.pad(audio[:target], (0, max(0, target - len(audio))))
        peak = float(np.max(np.abs(audio)))
        if peak > 0:
            audio = audio / peak
        mel = librosa.feature.melspectrogram(
            y=audio, sr=sample_rate, n_fft=config["n_fft"],
            hop_length=config["hop_length"], n_mels=config["n_mels"],
            fmin=config["fmin"], fmax=config["fmax"],
        )
        mel = librosa.power_to_db(mel, ref=np.max).astype(np.float32)
        return path.stem, torch.from_numpy(mel).unsqueeze(0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audio-dir", type=Path, required=True)
    parser.add_argument("--truth", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()

    with (args.model_dir / "config.yaml").open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    truth = pd.read_csv(args.truth, dtype={"ID": str})
    by_id = {
        path.stem: path for path in args.audio_dir.iterdir()
        if path.is_file() and path.suffix.lower() in AUDIO_EXTENSIONS
    }
    missing = [item_id for item_id in truth.ID if item_id not in by_id]
    if missing:
        raise ValueError(f"missing audio for {missing[:5]}")
    paths = [by_id[item_id] for item_id in truth.ID]

    model = HybridModel(config)
    checkpoint = torch.load(
        args.model_dir / "best_model.pth", map_location="cpu", weights_only=True,
    )
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model = model.to(args.device).eval()
    loader = DataLoader(
        MelDataset(paths, config), batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=args.device.startswith("cuda"),
    )
    ids, scores = [], []
    with torch.inference_mode():
        for batch_ids, mel in loader:
            probability = model(mel.to(args.device, non_blocking=True)).softmax(-1)[:, 1]
            ids.extend(batch_ids)
            scores.extend(probability.float().cpu().tolist())
    if ids != truth.ID.tolist():
        raise RuntimeError("scorer changed truth order")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"ID": ids, "HYBRID_FAKE_PROB": scores}).to_csv(
        args.output, index=False,
    )
    print(f"Wrote {len(ids)} hybrid scores to {args.output}")


if __name__ == "__main__":
    main()
