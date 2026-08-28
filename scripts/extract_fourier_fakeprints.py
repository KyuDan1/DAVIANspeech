"""Extract Deezer/ISMIR Fourier fakeprints from 16 kHz competition audio.

Adapted from the CC BY-NC 4.0 reference implementation accompanying
"A Fourier Explanation of AI-music Artifacts" (ISMIR 2025).  The original
5--16 kHz band becomes 5--8 kHz at the competition's fixed 16 kHz rate.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import librosa
import numpy as np
from scipy import interpolate
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from pipeline import find_audio_files  # noqa: E402

SR = 16_000
N_FFT = 1 << 14


def lower_hull(values: np.ndarray, area=10):
    indices, hull = [], []
    for offset in range(len(values) - area + 1):
        index = offset + int(np.argmin(values[offset:offset + area]))
        if not indices or index != indices[-1]:
            indices.append(index)
            hull.append(values[index])
    if indices[0] != 0:
        indices.insert(0, 0); hull.insert(0, values[0])
    if indices[-1] != len(values) - 1:
        indices.append(len(values) - 1); hull.append(values[-1])
    return np.asarray(indices), np.asarray(hull)


def fakeprint(path: Path, fmin: int, fmax: int):
    audio, _ = librosa.load(path, sr=SR, mono=True, dtype=np.float32)
    spectrum = np.abs(librosa.stft(audio, n_fft=N_FFT)) ** 2
    curve = np.mean(10 * np.log10(np.clip(spectrum, 1e-10, 1e6)), axis=1)
    frequencies = np.linspace(0, SR / 2, N_FFT // 2 + 1)
    selected = (frequencies > fmin) & (frequencies < fmax)
    values = curve[selected]
    indices, hull = lower_hull(values)
    baseline = interpolate.interp1d(
        indices, hull, kind="quadratic"
    )(np.arange(len(values)))
    residual = np.clip(values - np.clip(baseline, -45, None), 0, 5)
    return residual / (1e-6 + residual.max())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fmin", type=int, default=5_000)
    parser.add_argument("--fmax", type=int, default=7_990)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    args = parser.parse_args()
    files = find_audio_files(args.test_dir)[args.shard_index::args.num_shards]
    vectors = [fakeprint(path, args.fmin, args.fmax)
               for path in tqdm(files, desc="Fourier fakeprints")]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output, ids=np.asarray([p.stem for p in files]),
        embeddings=np.asarray(vectors, dtype=np.float32),
    )


if __name__ == "__main__":
    main()
