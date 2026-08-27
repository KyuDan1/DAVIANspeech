"""Score a directory with the SONICS music deepfake detector."""

from __future__ import annotations

import argparse
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
import torch
from sonics import HFAudioClassifier


def load_windows(path, samples=80_000):
    audio, _ = librosa.load(path, sr=16_000, mono=True, dtype=np.float32)
    if len(audio) < samples:
        audio = np.pad(audio, (0, samples - len(audio)))
    starts = list(range(0, max(1, len(audio) - samples + 1), samples))
    tail = len(audio) - samples
    if starts[-1] != tail:
        starts.append(tail)
    windows = np.stack([audio[start:start + samples] for start in starts])
    std = np.maximum(windows.std(axis=1, keepdims=True), 1e-6)
    return windows / std


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audio-dir", type=Path, required=True)
    parser.add_argument("--sample-submission", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--pool", choices=["mean", "max"], default="mean")
    args = parser.parse_args()

    sample = pd.read_csv(args.sample_submission, dtype={"ID": str})
    paths = {path.stem: path for path in args.audio_dir.iterdir() if path.is_file()}
    missing = [sample_id for sample_id in sample.ID if sample_id not in paths]
    if missing:
        raise ValueError(f"Missing audio for {missing[:5]}")
    model = HFAudioClassifier.from_pretrained(str(args.model_dir)).to(args.device).eval()

    probabilities = []
    for sample_id in sample.ID:
        windows = load_windows(paths[sample_id])
        window_probabilities = []
        for offset in range(0, len(windows), args.batch_size):
            batch = torch.from_numpy(windows[offset:offset + args.batch_size]).to(args.device)
            with torch.inference_mode():
                window_probabilities.extend(
                    torch.sigmoid(model(batch).flatten()).cpu().tolist()
                )
        probabilities.append(
            float(np.mean(window_probabilities)) if args.pool == "mean"
            else float(np.max(window_probabilities))
        )
    sample["MUSIC_FAKE_PROB"] = probabilities
    sample["FILE_FAKE_PROB"] = probabilities
    sample["VOICE_FAKE_PROB"] = 0.0
    sample["VOICE_PRESENT_PROB"] = 0.0
    sample["MUSIC_PRESENT_PROB"] = 1.0
    args.output.parent.mkdir(parents=True, exist_ok=True)
    sample.to_csv(args.output, index=False, lineterminator="\r\n")
    print(f"Saved {len(sample)} predictions to {args.output}")


if __name__ == "__main__":
    main()
