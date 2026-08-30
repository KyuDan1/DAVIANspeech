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
    device = torch.device(args.device)

    output_rows = []
    for sample in tqdm(sample_rows, desc="simple inference"):
        sample_id = str(sample["ID"])
        audio = load_audio(audio_by_id[sample_id])
        voice_present, music_present = presence.predict(audio)
        released_voice, voice_embedding = voice_fake_probability(
            voice_model, audio, device, args.window, args.batch_size
        )
        if voice_head is None:
            voice_fake = released_voice
        else:
            adapted_voice = float(expit(
                voice_embedding @ voice_head["weight"] + float(voice_head["bias"])
            ))
            voice_fake = blend_logits(
                released_voice, adapted_voice, float(voice_head["blend_new"])
            )
        whole_music = music_model.fake_probability(audio)
        segment_music = max(
            music_model.fake_probability(extract_segment(audio, start, args.music_segment))
            for start in segment_starts(len(audio), args.music_segment)
        )
        music_fake = blend_logits(whole_music, segment_music, args.music_segment_weight)
        output_rows.append({
            "ID": sample_id,
            "FILE_FAKE_PROB": max(voice_fake, music_fake),
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


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test-dir", type=Path, default=Path("data/test"))
    parser.add_argument("--sample-submission", type=Path, default=Path("data/sample_submission.csv"))
    parser.add_argument("--output", type=Path, default=Path("output/submission.csv"))
    parser.add_argument("--panns-dir", type=Path, default=Path("models/panns"))
    parser.add_argument("--xlsr-dir", type=Path, default=Path("models/xls-r-2b-anti-deepfake"))
    parser.add_argument("--fourier-music-head", type=Path, default=Path("model_heads/fourier-echoes-music-head.npz"))
    parser.add_argument("--xlsr-mixed-voice-head", type=Path, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--window", type=int, default=64_000)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--music-segment", type=int, default=64_000)
    parser.add_argument("--music-segment-weight", type=float, default=0.0)
    return parser.parse_args(argv)


if __name__ == "__main__":
    run(parse_args())
