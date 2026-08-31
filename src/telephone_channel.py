"""Deterministic telephone/VoIP channel simulations for router experiments.

The transform is applied to the final mixed waveform.  This mirrors a call
path: speech, background music, and noise first enter the microphone and the
resulting mixture then passes through the channel codec.
"""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import butter, resample_poly, sosfilt


SAMPLE_RATE = 16_000
NARROW_BAND = (260, 3_650)


def _peak_limit(audio: np.ndarray) -> np.ndarray:
    audio = np.nan_to_num(np.asarray(audio, dtype=np.float32))
    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    return audio / max(peak / 0.98, 1.0)


def _bandpass(audio: np.ndarray, low: int, high: int,
              sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    sos = butter(6, [low, high], btype="bandpass", fs=sample_rate, output="sos")
    return sosfilt(sos, audio).astype(np.float32)


def _resample(audio: np.ndarray, source_rate: int, target_rate: int) -> np.ndarray:
    divisor = np.gcd(source_rate, target_rate)
    return resample_poly(
        audio, target_rate // divisor, source_rate // divisor
    ).astype(np.float32)


def _ffmpeg_roundtrip(audio: np.ndarray, ffmpeg: Path,
                      encode_args: list[str], suffix: str,
                      encode_rate: int = SAMPLE_RATE,
                      decode_input_args: list[str] | None = None) -> np.ndarray:
    """Encode/decode one waveform without relying on system codec caches."""
    with tempfile.TemporaryDirectory(prefix="davianspeech_phone_") as directory:
        directory = Path(directory)
        source = directory / "source.wav"
        encoded = directory / f"encoded{suffix}"
        decoded = directory / "decoded.wav"
        sf.write(source, audio, SAMPLE_RATE, subtype="PCM_16")
        subprocess.run(
            [str(ffmpeg), "-nostdin", "-hide_banner", "-loglevel", "error", "-y",
             "-i", str(source), "-ac", "1", "-ar", str(encode_rate),
             *encode_args, str(encoded)],
            check=True,
        )
        subprocess.run(
            [str(ffmpeg), "-nostdin", "-hide_banner", "-loglevel", "error", "-y",
             *(decode_input_args or []), "-i", str(encoded),
             "-ac", "1", "-ar", str(SAMPLE_RATE),
             "-c:a", "pcm_s16le", str(decoded)],
            check=True,
        )
        result, rate = sf.read(decoded, dtype="float32", always_2d=False)
    if rate != SAMPLE_RATE:
        result = _resample(result, rate, SAMPLE_RATE)
    return np.asarray(result, dtype=np.float32)


def _frame_erasure(audio: np.ndarray, key: int, probability: float = 0.035,
                   frame_ms: int = 20) -> np.ndarray:
    """Deterministic packet-loss concealment proxy using repeated frames."""
    result = audio.copy()
    frame = SAMPLE_RATE * frame_ms // 1_000
    rng = np.random.default_rng(key)
    for start in range(frame, len(result), frame):
        if rng.random() < probability:
            previous = result[start - frame:start]
            end = min(start + frame, len(result))
            result[start:end] = previous[:end - start] * 0.96
    return result


def _post_noise(audio: np.ndarray, key: int, snr_db: float) -> np.ndarray:
    rng = np.random.default_rng(key)
    noise = rng.standard_normal(len(audio)).astype(np.float32)
    signal_rms = float(np.sqrt(np.mean(audio.astype(np.float64) ** 2) + 1e-12))
    noise /= float(np.sqrt(np.mean(noise.astype(np.float64) ** 2) + 1e-12))
    result = audio + noise * signal_rms / 10 ** (snr_db / 20)
    return _peak_limit(result)


def _numpy_mulaw(audio: np.ndarray) -> np.ndarray:
    mu = 255.0
    compressed = np.sign(audio) * np.log1p(mu * np.abs(audio)) / np.log1p(mu)
    quantized = np.round((compressed + 1) * 127.5) / 127.5 - 1
    return (
        np.sign(quantized) * np.expm1(np.abs(quantized) * np.log1p(mu)) / mu
    ).astype(np.float32)


POSITIVE_VARIANTS = (
    "resample8k",
    "pstn_bandpass",
    "g711_ulaw",
    "g711_alaw",
    "g726_24k",
    "opus_nb_8k",
    "opus_nb_12k",
    "transcode_g711_opus",
    "packetloss_opus_nb",
    "packetloss_opus_nb_heavy",
    "packetloss_g711",
    "random_bandpass",
    "mulaw_numpy",
    "g711_postnoise_30",
    "g711_postnoise_20",
    "opus_nb_postnoise_30",
    "opus_nb_postnoise_20",
    "g711_clipped",
    "fft_narrowband",
    "g726_postnoise_25",
    "opus_nb_clipped",
    "double_g726_g711",
)

NEGATIVE_VARIANTS = (
    "clean",
    "mp3_64k",
    "ogg_48k",
    "opus_audio_48k",
    "aac_48k",
    "lowpass_6k",
    "lowpass_5k",
    "opus_audio_32k",
)

# These may originate from a call, but preserve the 7--8 kHz band and do not
# necessarily need the narrowband fake-detection expert.  They are explicit
# negatives for the routing *decision*, not claims about recording provenance.
WIDEBAND_VARIANTS = (
    "g722_wb",
    "opus_wb_16k",
    "opus_wb_12k",
    "packetloss_g722",
)


def apply_channel(audio: np.ndarray, variant: str, ffmpeg: Path | None = None,
                  key: int = 0) -> np.ndarray:
    """Return a 16 kHz mono waveform for one channel condition."""
    audio = _peak_limit(audio)
    if variant == "clean":
        return audio
    if variant == "resample8k":
        return _resample(_resample(audio, SAMPLE_RATE, 8_000), 8_000, SAMPLE_RATE)
    if variant == "pstn_bandpass":
        filtered = _bandpass(audio, *NARROW_BAND)
        return _resample(_resample(filtered, SAMPLE_RATE, 8_000), 8_000, SAMPLE_RATE)
    if variant == "random_bandpass":
        rng = np.random.default_rng(key)
        low = int(rng.integers(180, 420))
        high = int(rng.integers(3_150, 3_850))
        filtered = _bandpass(audio, low, high)
        return _resample(_resample(filtered, SAMPLE_RATE, 8_000), 8_000, SAMPLE_RATE)
    if variant == "mulaw_numpy":
        filtered = _bandpass(audio, *NARROW_BAND)
        narrow = _resample(filtered, SAMPLE_RATE, 8_000)
        return _resample(_numpy_mulaw(narrow), 8_000, SAMPLE_RATE)
    if variant == "lowpass_6k":
        sos = butter(6, 6_000, btype="lowpass", fs=SAMPLE_RATE, output="sos")
        return sosfilt(sos, audio).astype(np.float32)
    if variant == "lowpass_5k":
        sos = butter(6, 5_000, btype="lowpass", fs=SAMPLE_RATE, output="sos")
        return sosfilt(sos, audio).astype(np.float32)
    if variant == "fft_narrowband":
        rng = np.random.default_rng(key)
        low = float(rng.integers(180, 380))
        high = float(rng.integers(3_200, 3_750))
        spectrum = np.fft.rfft(audio)
        frequencies = np.fft.rfftfreq(len(audio), 1 / SAMPLE_RATE)
        transition = 120.0
        low_gain = np.clip((frequencies - low) / transition, 0, 1)
        high_gain = np.clip((high - frequencies) / transition, 0, 1)
        filtered = np.fft.irfft(spectrum * low_gain * high_gain, n=len(audio))
        return filtered.astype(np.float32)
    if ffmpeg is None:
        raise ValueError(f"{variant} requires an ffmpeg executable")
    ffmpeg = Path(ffmpeg)
    if variant in {"g711_ulaw", "g711_alaw"}:
        codec = "pcm_mulaw" if variant.endswith("ulaw") else "pcm_alaw"
        filtered = _bandpass(audio, *NARROW_BAND)
        return _ffmpeg_roundtrip(
            filtered, ffmpeg, ["-c:a", codec], ".wav", encode_rate=8_000
        )
    if variant == "g722_wb":
        return _ffmpeg_roundtrip(
            audio, ffmpeg, ["-c:a", "g722", "-b:a", "64k"], ".g722"
        )
    if variant == "g726_24k":
        filtered = _bandpass(audio, *NARROW_BAND)
        return _ffmpeg_roundtrip(
            filtered, ffmpeg,
            ["-c:a", "g726", "-b:a", "24k", "-f", "g726"],
            ".g726", encode_rate=8_000,
            decode_input_args=["-f", "g726", "-code_size", "3", "-ar", "8000"],
        )
    if variant in {
        "opus_nb_8k", "opus_nb_12k", "packetloss_opus_nb",
        "packetloss_opus_nb_heavy",
    }:
        bitrate = "12k" if variant == "opus_nb_12k" else "8k"
        filtered = _bandpass(audio, *NARROW_BAND)
        result = _ffmpeg_roundtrip(
            filtered, ffmpeg,
            ["-c:a", "libopus", "-application", "voip", "-b:a", bitrate,
             "-vbr", "off", "-frame_duration", "20"],
            ".opus", encode_rate=8_000,
        )
        if variant == "packetloss_opus_nb":
            return _frame_erasure(result, key, probability=.035)
        if variant == "packetloss_opus_nb_heavy":
            return _frame_erasure(result, key, probability=.08)
        return result
    if variant in {"opus_wb_16k", "opus_wb_12k"}:
        bitrate = "12k" if variant == "opus_wb_12k" else "18k"
        return _ffmpeg_roundtrip(
            audio, ffmpeg,
            ["-c:a", "libopus", "-application", "voip", "-b:a", bitrate,
             "-vbr", "off", "-frame_duration", "20"],
            ".opus", encode_rate=16_000,
        )
    if variant == "transcode_g711_opus":
        first = apply_channel(audio, "g711_ulaw", ffmpeg=ffmpeg, key=key)
        return apply_channel(first, "opus_nb_8k", ffmpeg=ffmpeg, key=key)
    if variant == "packetloss_g711":
        result = apply_channel(audio, "g711_alaw", ffmpeg=ffmpeg, key=key)
        return _frame_erasure(result, key, probability=.05)
    if variant == "packetloss_g722":
        result = apply_channel(audio, "g722_wb", ffmpeg=ffmpeg, key=key)
        return _frame_erasure(result, key, probability=.05)
    if variant in {"g711_postnoise_30", "g711_postnoise_20"}:
        result = apply_channel(audio, "g711_ulaw", ffmpeg=ffmpeg, key=key)
        snr = 30 if variant.endswith("30") else 20
        return _post_noise(result, key, snr)
    if variant in {"opus_nb_postnoise_30", "opus_nb_postnoise_20"}:
        result = apply_channel(audio, "opus_nb_8k", ffmpeg=ffmpeg, key=key)
        snr = 30 if variant.endswith("30") else 20
        return _post_noise(result, key, snr)
    if variant == "g711_clipped":
        result = apply_channel(audio, "g711_alaw", ffmpeg=ffmpeg, key=key)
        return _peak_limit(np.tanh(2.5 * result).astype(np.float32))
    if variant == "g726_postnoise_25":
        result = apply_channel(audio, "g726_24k", ffmpeg=ffmpeg, key=key)
        return _post_noise(result, key, 25)
    if variant == "opus_nb_clipped":
        result = apply_channel(audio, "opus_nb_12k", ffmpeg=ffmpeg, key=key)
        return _peak_limit(np.tanh(2.0 * result).astype(np.float32))
    if variant == "double_g726_g711":
        first = apply_channel(audio, "g726_24k", ffmpeg=ffmpeg, key=key)
        return apply_channel(first, "g711_alaw", ffmpeg=ffmpeg, key=key)
    if variant == "mp3_64k":
        return _ffmpeg_roundtrip(
            audio, ffmpeg, ["-c:a", "libmp3lame", "-b:a", "64k"], ".mp3"
        )
    if variant == "ogg_48k":
        return _ffmpeg_roundtrip(
            audio, ffmpeg, ["-c:a", "libvorbis", "-b:a", "48k"], ".ogg"
        )
    if variant == "opus_audio_48k":
        return _ffmpeg_roundtrip(
            audio, ffmpeg,
            ["-c:a", "libopus", "-application", "audio", "-b:a", "48k",
             "-vbr", "off", "-frame_duration", "20"],
            ".opus", encode_rate=16_000,
        )
    if variant == "opus_audio_32k":
        return _ffmpeg_roundtrip(
            audio, ffmpeg,
            ["-c:a", "libopus", "-application", "audio", "-b:a", "32k",
             "-vbr", "off", "-frame_duration", "20"],
            ".opus", encode_rate=16_000,
        )
    if variant == "aac_48k":
        return _ffmpeg_roundtrip(
            audio, ffmpeg, ["-c:a", "aac", "-b:a", "48k"], ".m4a"
        )
    raise ValueError(f"unknown channel variant: {variant}")
