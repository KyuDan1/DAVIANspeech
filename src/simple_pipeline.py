"""Separation-free competition pipeline selected on source-disjoint eval data.

Voice fake: released NII head plus a domain-balanced mixed-audio linear head.
Music fake: domain-balanced Fourier head with temporal multiple-instance pooling.
File fake: logical OR implemented as max(component scores).
Presence: unchanged PANNs baseline.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import librosa
import numpy as np
import torch
from scipy.special import expit, logit
from tqdm import tqdm

from fourier_detector import FourierMusicDetector
from presence import PannsPresence, extract_segment, segment_starts
from xlsr_antideepfake import XlsrAntiDeepfake

AUDIO_EXTENSIONS = {".aac", ".flac", ".m4a", ".mp3", ".ogg", ".opus", ".wav", ".wma"}
PREDICTION_COLUMNS = [
    "FILE_FAKE_PROB", "VOICE_FAKE_PROB", "MUSIC_FAKE_PROB",
    "VOICE_PRESENT_PROB", "MUSIC_PRESENT_PROB",
]


def load_audio(path: Path) -> np.ndarray:
    audio, _ = librosa.load(path, sr=16_000, mono=True, dtype=np.float32)
    if not audio.size or not np.isfinite(audio).all():
        raise ValueError(f"Invalid audio: {path}")
    return audio


def voice_fake_probability(model, audio, device, window=64_000, batch_size=1):
    windows = np.stack([
        extract_segment(audio, start, window)
        for start in segment_starts(audio.size, window)
    ])
    best, embeddings = 0.0, []
    for offset in range(0, len(windows), batch_size):
        batch = torch.from_numpy(windows[offset:offset + batch_size]).to(device)
        with torch.inference_mode():
            pooled = model.embedding(model.normalize(batch))
            probabilities = torch.softmax(model.proj_fc(pooled).float(), dim=-1)[:, 0]
        best = max(best, float(probabilities.max()))
        embeddings.append(pooled.float().cpu().numpy())
    return best, np.concatenate(embeddings).mean(axis=0)


def blend_logits(first, second, second_weight):
    values = np.clip([first, second], 1e-7, 1 - 1e-7)
    return float(expit((1 - second_weight) * logit(values[0]) + second_weight * logit(values[1])))


def high_band_energy_ratio(audio, cutoff_hz=4_200):
    """File-local telephone-band cue; no statistics from other test files."""
    spectrum = np.abs(np.fft.rfft(audio)) ** 2
    frequencies = np.fft.rfftfreq(len(audio), d=1 / 16_000)
    return float(spectrum[frequencies >= cutoff_hz].sum() / (spectrum.sum() + 1e-20))


def select_route(mixture_present, voice_present, music_present,
                 mixture_threshold=.8, voice_min=.4, music_max=.075,
                 is_phone=False):
    """Choose one expert path using only information from the current file."""
    if mixture_present >= mixture_threshold:
        return "mixed"
    # Telephone filtering changes the absolute PANNs scale.  The relative
    # ordering retained better held-out concurrent/cascaded ADS.
    if is_phone:
        return "voice" if voice_present > music_present else "music"
    if voice_present >= voice_min and music_present <= music_max:
        return "voice"
    return "music"


def run(args):
    audio_by_id = {
        path.stem: path for path in args.test_dir.iterdir()
        if path.is_file() and path.suffix.lower() in AUDIO_EXTENSIONS
    }
    with args.sample_submission.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        columns, sample_rows = reader.fieldnames, list(reader)
    if not columns or any(column not in columns for column in ["ID", *PREDICTION_COLUMNS]):
        raise ValueError("Invalid sample submission columns")
    missing = [row["ID"] for row in sample_rows if row["ID"] not in audio_by_id]
    if missing:
        raise ValueError(f"Missing audio for IDs: {missing[:5]}")

    presence = PannsPresence(args.panns_dir, device=args.device)
    voice_model = XlsrAntiDeepfake.from_checkpoint(
        args.xlsr_dir, device=args.device, dtype=torch.float32
    )
    music_model = FourierMusicDetector(args.fourier_music_head)
    voice_head = np.load(args.xlsr_mixed_voice_head) if args.xlsr_mixed_voice_head else None
    router = None
    if args.spear_dir and args.spear_mixture_head:
        from spear_detector import SpearMusicDetector
        router = SpearMusicDetector(
            args.spear_dir, args.spear_mixture_head, device=args.device
        )
    phone_eat = phone_sonics = None
    device = torch.device(args.device)

    output_rows, diagnostic_rows = [], []
    for sample in tqdm(sample_rows, desc="simple inference"):
        sample_id = str(sample["ID"])
        audio = load_audio(audio_by_id[sample_id])
        band_ratio = high_band_energy_ratio(audio)
        is_phone = band_ratio < args.phone_band_threshold
        voice_present, music_present = presence.predict(audio)
        mixture_present = router.fake_probability(audio) if router is not None else 1.0
        released_voice, voice_embedding = voice_fake_probability(
            voice_model, audio, device, args.window, args.batch_size
        )
        if voice_head is None:
            adapted_voice = released_voice
            mixed_voice = released_voice
        else:
            adapted_voice = float(expit(
                voice_embedding @ voice_head["weight"] + float(voice_head["bias"])
            ))
            mixed_voice = blend_logits(
                released_voice, adapted_voice, float(voice_head["blend_new"])
            )
        if is_phone:
            mixed_voice = blend_logits(
                released_voice, adapted_voice, args.phone_adapted_voice_weight
            )
        whole_music = music_model.fake_probability(audio)
        segment_music = max(
            music_model.fake_probability(extract_segment(audio, start, args.music_segment))
            for start in segment_starts(len(audio), args.music_segment)
        )
        regular_music = blend_logits(whole_music, segment_music, args.music_segment_weight)
        if is_phone and args.eat_dir and args.eat_phone_head and args.sonics_dir:
            if phone_eat is None:
                from eat_detector import EatMusicDetector
                from sonics_detector import SonicsMusicDetector
                phone_eat = EatMusicDetector(
                    args.eat_dir, args.eat_phone_head, device=args.device
                )
                phone_sonics = SonicsMusicDetector.from_checkpoint(
                    args.sonics_dir, device=args.device
                )
            eat_music = phone_eat.fake_probability(audio)
            sonics_music = phone_sonics.fake_probability(audio, device=args.device)
            music_fake = float(expit(
                args.phone_sonics_weight * logit(np.clip(sonics_music, 1e-7, 1 - 1e-7))
                + (1 - args.phone_sonics_weight) * logit(np.clip(eat_music, 1e-7, 1 - 1e-7))
                + args.phone_music_bias
            ))
        else:
            eat_music = sonics_music = np.nan
            music_fake = regular_music

        route = select_route(
            mixture_present, voice_present, music_present,
            args.mixture_threshold, args.single_voice_min,
            args.single_music_max, is_phone,
        )
        voice_only, music_only = route == "voice", route == "music"
        voice_fake = released_voice if voice_only else mixed_voice
        if voice_only:
            file_fake = voice_fake
        elif music_only:
            file_fake = music_fake
        else:
            file_fake = max(voice_fake, music_fake)
        diagnostic_rows.append({
            "ID": sample_id, "voice_present": voice_present,
            "music_present": music_present, "released_voice": released_voice,
            "adapted_voice": adapted_voice, "mixed_voice": mixed_voice,
            "whole_music": whole_music, "segment_music": segment_music,
            "regular_music": regular_music, "eat_music": eat_music,
            "sonics_music": sonics_music, "music_fake": music_fake,
            "mixture_present": mixture_present, "is_phone": int(is_phone),
            "route": route,
            "high_band_ratio": band_ratio,
        })
        output_rows.append({
            "ID": sample_id,
            "FILE_FAKE_PROB": file_fake,
            "VOICE_FAKE_PROB": voice_fake,
            "MUSIC_FAKE_PROB": music_fake,
            "VOICE_PRESENT_PROB": voice_present,
            "MUSIC_PRESENT_PROB": music_present,
        })

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["ID", *PREDICTION_COLUMNS])
        writer.writeheader()
        writer.writerows(output_rows)
    if args.diagnostic_output:
        args.diagnostic_output.parent.mkdir(parents=True, exist_ok=True)
        with args.diagnostic_output.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(diagnostic_rows[0]))
            writer.writeheader(); writer.writerows(diagnostic_rows)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test-dir", type=Path, default=Path("data/test"))
    parser.add_argument("--sample-submission", type=Path, default=Path("data/sample_submission.csv"))
    parser.add_argument("--output", type=Path, default=Path("output/submission.csv"))
    parser.add_argument("--diagnostic-output", type=Path, default=None)
    parser.add_argument("--panns-dir", type=Path, default=Path("models/panns"))
    parser.add_argument("--xlsr-dir", type=Path, default=Path("models/xls-r-2b-anti-deepfake"))
    parser.add_argument("--fourier-music-head", type=Path, default=Path("model_heads/fourier-echoes-music-head.npz"))
    parser.add_argument("--xlsr-mixed-voice-head", type=Path, default=None)
    parser.add_argument("--spear-dir", type=Path, default=None)
    parser.add_argument("--spear-mixture-head", type=Path, default=None)
    parser.add_argument("--eat-dir", type=Path, default=None)
    parser.add_argument("--eat-phone-head", type=Path, default=None)
    parser.add_argument("--sonics-dir", type=Path, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--window", type=int, default=64_000)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--music-segment", type=int, default=64_000)
    parser.add_argument("--music-segment-weight", type=float, default=0.0)
    parser.add_argument("--mixture-threshold", type=float, default=0.8)
    parser.add_argument("--single-voice-min", type=float, default=0.4)
    parser.add_argument("--single-music-max", type=float, default=0.075)
    parser.add_argument("--phone-band-threshold", type=float, default=3e-6)
    parser.add_argument("--phone-adapted-voice-weight", type=float, default=0.7)
    parser.add_argument("--phone-sonics-weight", type=float, default=0.8)
    parser.add_argument("--phone-music-bias", type=float, default=1.5)
    return parser.parse_args(argv)


if __name__ == "__main__":
    run(parse_args())
