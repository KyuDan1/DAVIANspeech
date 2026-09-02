#!/usr/bin/env python3
"""Score the public modern Suno/Udio fakeprint detector on labelled banks."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from modern_fakeprint_detector import ModernFakeprintDetector  # noqa: E402


AUDIO_EXTENSIONS = {".aac", ".flac", ".m4a", ".mp3", ".ogg", ".opus", ".wav"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, nargs="+", required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    detector = ModernFakeprintDetector(args.model)
    rows = []
    for dataset in args.dataset:
        truth = pd.read_csv(dataset / "truth.csv", dtype={"ID": str})
        audio_dir = dataset / "audio"
        paths = {
            path.stem: path for path in audio_dir.rglob("*")
            if path.is_file() and path.suffix.lower() in AUDIO_EXTENSIONS
        }
        missing = sorted(set(truth.ID) - set(paths))
        if missing:
            raise FileNotFoundError(f"{dataset}: missing {missing[:5]}")
        for sample_id in tqdm(truth.ID, desc=f"fakeprint {dataset.name}"):
            audio, _ = librosa.load(
                paths[sample_id], sr=detector.SAMPLE_RATE, mono=True,
                dtype="float32",
            )
            margin = detector.fake_margin(audio)
            rows.append({
                "DATASET": dataset.name,
                "ID": sample_id,
                "MODERN_FAKEPRINT_MARGIN": margin,
                "MODERN_FAKEPRINT_PROB": float(
                    np.exp(-np.logaddexp(0.0, -np.clip(margin, -700.0, 700.0)))
                ),
            })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.output, index=False)
    print(f"Saved {len(rows)} rows to {args.output}")


if __name__ == "__main__":
    main()
