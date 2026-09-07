#!/usr/bin/env python3
"""Score local audio with the MIT-licensed Orphea MusicGen detector.

This is an audit utility, not a submission dependency.  Preprocessing follows
the public Orphea Space exactly: first 30 seconds at 22.05 kHz, 128-bin librosa
mel spectrogram, per-file 0--255 scaling, 224x224 resize, and ImageNet-like
three-channel normalization to [-1, 1].  Orphea class 0 is AI and class 1 is
human, so the exported score is ``softmax(logits)[0]``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import mobilenet_v3_small


AUDIO_EXTENSIONS = {".aac", ".flac", ".m4a", ".mp3", ".ogg", ".opus", ".wav", ".wma"}
SAMPLE_RATE = 22_050
DURATION = 30


class AudioImages(Dataset):
    def __init__(self, paths: list[Path]):
        self.paths = paths
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.5] * 3, [0.5] * 3),
        ])

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> tuple[str, torch.Tensor]:
        path = self.paths[index]
        waveform, _ = librosa.load(
            path, sr=SAMPLE_RATE, mono=True, duration=DURATION,
        )
        target = SAMPLE_RATE * DURATION
        if len(waveform) < target:
            waveform = np.pad(waveform, (0, target - len(waveform)))
        mel = librosa.feature.melspectrogram(
            y=waveform, sr=SAMPLE_RATE, n_mels=128,
        )
        mel_db = librosa.power_to_db(mel, ref=np.max)
        span = float(mel_db.max() - mel_db.min())
        if span > 0:
            image = (mel_db - mel_db.min()) / span
        else:
            image = np.zeros_like(mel_db)
        image = (image * 255).clip(0, 255).astype(np.uint8)
        image = np.stack([image] * 3, axis=-1)
        return path.stem, self.transform(image)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audio-dir", type=Path, required=True)
    parser.add_argument("--truth", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()

    truth = pd.read_csv(args.truth, dtype={"ID": str})
    by_id = {
        path.stem: path for path in args.audio_dir.iterdir()
        if path.is_file() and path.suffix.lower() in AUDIO_EXTENSIONS
    }
    missing = [item_id for item_id in truth.ID if item_id not in by_id]
    if missing:
        raise ValueError(f"missing audio for {missing[:5]}")
    paths = [by_id[item_id] for item_id in truth.ID]

    model = mobilenet_v3_small(weights=None)
    model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, 2)
    model.load_state_dict(torch.load(
        args.checkpoint, map_location="cpu", weights_only=True,
    ))
    model = model.to(args.device).eval()
    loader = DataLoader(
        AudioImages(paths), batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=args.device.startswith("cuda"),
    )
    ids, probabilities = [], []
    with torch.inference_mode():
        for batch_ids, images in loader:
            scores = model(images.to(args.device, non_blocking=True)).softmax(-1)[:, 0]
            ids.extend(batch_ids)
            probabilities.extend(scores.float().cpu().tolist())
    if ids != truth.ID.tolist():
        raise RuntimeError("scorer changed truth order")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"ID": ids, "ORPHEA_FAKE_PROB": probabilities}).to_csv(
        args.output, index=False,
    )
    print(f"Wrote {len(ids)} Orphea scores to {args.output}")


if __name__ == "__main__":
    main()
