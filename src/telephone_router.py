"""Content-robust handcrafted features and a lightweight telephone router."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.fft import dct
from scipy.special import expit


SAMPLE_RATE = 16_000
CUTOFFS = (250, 300, 500, 1_000, 2_000, 3_000, 3_400, 3_650,
           3_900, 4_000, 4_200, 4_500, 5_000, 6_000, 7_000)
# Layout: band(160), delta(64), cepstral(80), scalar(35), then five
# aggregates of the frame cutoff ratios.  The median 4.2 kHz ratio is robust
# to sparse packet-loss discontinuities that contaminate a whole-file FFT.
FRAME_CUTOFF_START = 160 + 64 + 80 + 35
ROBUST_BAND_FEATURE_INDEX = FRAME_CUTOFF_START + 3 * len(CUTOFFS) + CUTOFFS.index(4_200)
GLOBAL_CUTOFF_START = FRAME_CUTOFF_START + 5 * len(CUTOFFS)
GLOBAL_BAND_FEATURE_INDEX = GLOBAL_CUTOFF_START + CUTOFFS.index(4_200)


def _frames(audio: np.ndarray, frame: int = 400, hop: int = 160,
            max_frames: int = 1_200) -> np.ndarray:
    audio = np.asarray(audio, dtype=np.float32).reshape(-1)
    if audio.size < frame:
        audio = np.pad(audio, (0, frame - audio.size))
    count = 1 + (len(audio) - frame) // hop
    starts = np.arange(count) * hop
    if count > max_frames:
        starts = starts[np.linspace(0, count - 1, max_frames, dtype=int)]
    return np.stack([audio[start:start + frame] for start in starts])


def _aggregate(values: np.ndarray) -> np.ndarray:
    return np.concatenate([
        np.mean(values, axis=0),
        np.std(values, axis=0),
        np.quantile(values, 0.10, axis=0),
        np.quantile(values, 0.50, axis=0),
        np.quantile(values, 0.90, axis=0),
    ])


def extract_telephone_features(audio: np.ndarray) -> np.ndarray:
    """Extract spectral-envelope, dynamics, cepstral, and cutoff features."""
    audio = np.nan_to_num(np.asarray(audio, dtype=np.float32).reshape(-1))
    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    if peak:
        audio = audio / peak
    framed = _frames(audio)
    windowed = framed * np.hanning(framed.shape[1])[None, :]
    power = np.abs(np.fft.rfft(windowed, n=512, axis=1)) ** 2
    power += 1e-12
    frequencies = np.fft.rfftfreq(512, 1 / SAMPLE_RATE)
    distribution = power / power.sum(axis=1, keepdims=True)

    # 32 equal-width bands preserve sharp 3.4/4/7 kHz codec cutoffs.
    edges = np.linspace(0, 8_000, 33)
    bands = np.stack([
        distribution[:, (frequencies >= low) & (frequencies < high)].sum(axis=1)
        for low, high in zip(edges[:-1], edges[1:])
    ], axis=1)
    log_bands = np.log10(bands + 1e-12)
    band_features = _aggregate(log_bands)
    delta_features = np.concatenate([
        np.mean(np.abs(np.diff(log_bands, axis=0)), axis=0),
        np.std(np.diff(log_bands, axis=0), axis=0),
    ])

    # Cepstral envelope complements raw bandwidth ratios for codec artifacts.
    cepstra = dct(log_bands, type=2, norm="ortho", axis=1)[:, :16]
    cepstral_features = _aggregate(cepstra)

    centroid = (distribution * frequencies).sum(axis=1) / 8_000
    spread = np.sqrt(
        (distribution * (frequencies[None, :] / 8_000 - centroid[:, None]) ** 2).sum(axis=1)
    )
    flatness = np.exp(np.mean(np.log(power), axis=1)) / np.mean(power, axis=1)
    cumulative = np.cumsum(distribution, axis=1)
    roll85 = frequencies[np.argmax(cumulative >= 0.85, axis=1)] / 8_000
    roll95 = frequencies[np.argmax(cumulative >= 0.95, axis=1)] / 8_000
    zcr = np.mean(np.diff(np.signbit(framed), axis=1), axis=1)
    rms = np.sqrt(np.mean(framed * framed, axis=1) + 1e-12)
    scalar_frames = np.stack([
        centroid, spread, flatness, roll85, roll95, zcr, np.log10(rms + 1e-12)
    ], axis=1)
    scalar_features = _aggregate(scalar_frames)

    frame_cutoff_ratios = np.stack([
        distribution[:, frequencies >= cutoff].sum(axis=1)
        for cutoff in CUTOFFS
    ], axis=1)
    frame_cutoff_features = _aggregate(
        np.log10(frame_cutoff_ratios + 1e-20)
    )

    global_power = np.abs(np.fft.rfft(audio)) ** 2 + 1e-20
    global_frequency = np.fft.rfftfreq(len(audio), 1 / SAMPLE_RATE)
    total = global_power.sum()
    cumulative_energy = np.asarray([
        global_power[global_frequency >= cutoff].sum() / total for cutoff in CUTOFFS
    ])
    cutoff_features = np.concatenate([
        np.log10(cumulative_energy + 1e-20),
        np.diff(np.log10(cumulative_energy + 1e-20)),
    ])
    result = np.concatenate([
        band_features, delta_features, cepstral_features,
        scalar_features, frame_cutoff_features, cutoff_features,
    ]).astype(np.float32)
    if not np.isfinite(result).all():
        raise ValueError("non-finite telephone features")
    return result


class TelephoneRouter:
    """Portable standardized linear router stored as a NumPy archive."""

    def __init__(self, checkpoint: Path):
        state = np.load(checkpoint)
        self.mean = state["mean"].astype(np.float32)
        self.scale = state["scale"].astype(np.float32)
        self.weight = state["weight"].astype(np.float32)
        self.bias = float(state["bias"])
        self.threshold = float(state["threshold"])
        feature_count = len(self.mean)
        if not (len(self.scale) == len(self.weight) == feature_count):
            raise ValueError("inconsistent telephone-router checkpoint dimensions")

    def probability(self, audio: np.ndarray) -> float:
        features = extract_telephone_features(audio)
        logit = ((features - self.mean) / self.scale) @ self.weight + self.bias
        return float(expit(logit))

    def is_narrowband(self, audio: np.ndarray) -> bool:
        """Return whether the narrowband fake-detection expert should run."""
        return self.probability(audio) >= self.threshold
