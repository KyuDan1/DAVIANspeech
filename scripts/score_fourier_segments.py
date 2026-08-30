"""Score fixed waveform segments with a Fourier head, without separation."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import librosa
import numpy as np
import pandas as pd
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from fourier_detector import FourierMusicDetector  # noqa: E402
from simple_pipeline import AUDIO_EXTENSIONS  # noqa: E402
from presence import extract_segment, segment_starts  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audio-dir", type=Path, required=True)
    parser.add_argument("--head", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--segment-seconds", type=float, default=4.0)
    args = parser.parse_args()
    detector = FourierMusicDetector(args.head)
    size = round(args.segment_seconds * 16_000)
    files = sorted(p for p in args.audio_dir.iterdir() if p.suffix.lower() in AUDIO_EXTENSIONS)
    rows = []
    for path in tqdm(files, desc="Fourier segments"):
        audio, _ = librosa.load(path, sr=16_000, mono=True, dtype=np.float32)
        scores = np.asarray([
            detector.fake_probability(extract_segment(audio, start, size))
            for start in segment_starts(len(audio), size)
        ])
        rows.append({
            "ID": path.stem, "N_SEGMENTS": len(scores),
            "WHOLE": detector.fake_probability(audio),
            "SEG_MIN": scores.min(), "SEG_Q25": np.quantile(scores, .25),
            "SEG_MEDIAN": np.median(scores), "SEG_MEAN": scores.mean(),
            "SEG_Q75": np.quantile(scores, .75), "SEG_MAX": scores.max(),
        })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.output, index=False)


if __name__ == "__main__":
    main()
