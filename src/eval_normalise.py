"""Strip the label out of everything that is not the label.

Every eval set built here so far failed the leakage gate: a three-line feature
matched or beat the detector on the very label the set exists to measure.

    korean_v2     flatness  0.1647
    complike_v1   rms       0.1957
    complike_v2   rms       0.2053
    mixtures4     duration  0.2800

None of those features know anything about generation. They score below 0.5
because of how the clips were built -- TTS comes out quieter and shorter than
the recordings it imitates, MusicGen sits lower in the spectrum than a 2002 CD
rip, and generated audio arrives without the room tone a microphone leaves at
the edges. Four controls, one per leak:

    loudness   -23 LUFS with jitter        kills rms, peak, crest
    duration   crop to a shared target     kills duration
    bandwidth  common rate + one MP3 pass  kills flatness, rolloff, centroid, band_*
    silence    trim the edges              kills lead_silence, tail_silence

Applied identically to both classes, so nothing survives that could stand in for
the label. The jitter matters: normalising every clip to exactly -23 LUFS would
erase loudness as a cue but also as a variable, and a set with no level variation
is its own kind of unrepresentative.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import librosa
import numpy as np
import pyloudnorm as pyln
import soundfile as sf

SR = 16_000
TARGET_LUFS = -23.0
LUFS_JITTER = 3.0
MP3_BITRATE = "64k"
_METER = pyln.Meter(SR)


def _ffmpeg() -> str:
    """ffmpeg lives in the conda env's bin, which is not always on PATH."""
    found = shutil.which("ffmpeg")
    if found:
        return found
    beside = Path(sys.executable).parent / "ffmpeg"
    if beside.is_file():
        return str(beside)
    raise RuntimeError(
        "ffmpeg not found; install it into this environment "
        "(conda install -c conda-forge ffmpeg) or put it on PATH"
    )


def trim_silence(audio: np.ndarray, top_db: float = 40.0) -> np.ndarray:
    trimmed, _ = librosa.effects.trim(audio, top_db=top_db)
    return trimmed if trimmed.size >= SR // 2 else audio


def set_loudness(audio: np.ndarray, target: float) -> np.ndarray:
    if audio.size < SR // 2:
        return audio
    measured = _METER.integrated_loudness(audio)
    if not np.isfinite(measured):
        return audio
    scaled = audio * (10.0 ** ((target - measured) / 20.0))
    peak = np.abs(scaled).max()
    # Loudness normalisation can push a limited track past full scale; pulling
    # it back would undo the control, so leave a little headroom instead.
    return scaled / peak * 0.98 if peak > 0.98 else scaled


def fit_duration(audio: np.ndarray, seconds: float, rng) -> np.ndarray:
    want = int(seconds * SR)
    if audio.size < want:
        audio = np.tile(audio, want // max(audio.size, 1) + 1)
    if audio.size > want:
        start = int(rng.integers(0, audio.size - want + 1))
        audio = audio[start:start + want]
    return audio[:want]


def one_mp3_generation(audio: np.ndarray) -> np.ndarray:
    """One shared lossy pass, so codec history cannot separate the classes."""
    with tempfile.TemporaryDirectory() as work:
        raw, encoded = Path(work) / "a.wav", Path(work) / "a.mp3"
        sf.write(raw, audio.astype(np.float32), SR)
        subprocess.run([_ffmpeg(), "-y", "-loglevel", "error", "-i", str(raw),
                        "-ar", str(SR), "-ac", "1", "-b:a", MP3_BITRATE, str(encoded)],
                       check=True)
        decoded, _ = librosa.load(encoded, sr=SR, mono=True, dtype=np.float32)
    return decoded


def normalise(audio: np.ndarray, seconds: float, rng,
              trim: bool = True, mp3: bool = True) -> np.ndarray:
    """The full chain. Order matters -- trim before cropping so the crop lands
    on signal, set loudness after cropping so it measures what ships, and encode
    last so every clip carries exactly one generation."""
    if trim:
        audio = trim_silence(audio)
    audio = fit_duration(audio, seconds, rng)
    audio = set_loudness(audio, TARGET_LUFS + float(rng.uniform(-LUFS_JITTER, LUFS_JITTER)))
    if mp3:
        audio = one_mp3_generation(audio)
        audio = fit_duration(audio, seconds, rng)   # the codec adds padding
    return audio.astype(np.float32)
