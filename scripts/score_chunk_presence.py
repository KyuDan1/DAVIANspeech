#!/usr/bin/env python3
"""Dump per-window PANNs voice/music activity without source separation."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import librosa
import numpy as np
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from presence import PannsPresence  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audio-dir", type=Path, required=True)
    parser.add_argument("--panns-dir", type=Path, default=REPO_ROOT / "models/panns")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    detector = PannsPresence(args.panns_dir, device=args.device)
    rows = []
    audio_paths = sorted(
        path for path in args.audio_dir.iterdir()
        if path.suffix.lower() in {".wav", ".flac", ".mp3", ".ogg", ".m4a"}
    )
    for path in tqdm(audio_paths, desc="chunk presence"):
        audio, _ = librosa.load(path, sr=16_000, mono=True, dtype=np.float32)
        clipwise, _ = detector.model.inference(detector._segments_32k(audio))
        for chunk_index, probabilities in enumerate(clipwise):
            rows.append({
                "ID": path.stem,
                "CHUNK_INDEX": chunk_index,
                "VOICE_ACTIVITY": float(probabilities[detector.voice_indices].max()),
                "MUSIC_ACTIVITY": float(probabilities[detector.music_indices].max()),
            })

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} chunks from {len(audio_paths)} files to {args.output}")


if __name__ == "__main__":
    main()
