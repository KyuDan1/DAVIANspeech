"""Extract original-audio EAT CLS embeddings for diagnostic training.

EAT is a general-audio SSL encoder used by the winning AT-ADD Track 2
systems for non-speech domains.  This script deliberately reads the original
mixture and never applies source separation.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import librosa
import numpy as np
import torch
import torch.nn.functional as F
import torchaudio
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from pipeline import find_audio_files  # noqa: E402
from eat_detector import _load_local_model  # noqa: E402

SAMPLE_RATE = 16_000
TARGET_SECONDS = 6
TARGET_FRAMES = 614
NORM_MEAN = -4.268
NORM_STD = 4.569


def center_crop(audio: np.ndarray, samples: int) -> np.ndarray:
    if len(audio) >= samples:
        start = (len(audio) - samples) // 2
        return audio[start:start + samples]
    left = (samples - len(audio)) // 2
    return np.pad(audio, (left, samples - len(audio) - left))


def fbank(path: Path) -> torch.Tensor:
    audio, _ = librosa.load(path, sr=SAMPLE_RATE, mono=True, dtype=np.float32)
    audio = center_crop(audio, TARGET_SECONDS * SAMPLE_RATE)
    waveform = torch.from_numpy(audio)
    waveform = waveform - waveform.mean()
    mel = torchaudio.compliance.kaldi.fbank(
        waveform.unsqueeze(0), htk_compat=True, sample_frequency=SAMPLE_RATE,
        use_energy=False, window_type="hanning", num_mel_bins=128,
        dither=0.0, frame_shift=10,
    )
    if mel.shape[0] < TARGET_FRAMES:
        mel = F.pad(mel, (0, 0, 0, TARGET_FRAMES - mel.shape[0]))
    else:
        mel = mel[:TARGET_FRAMES]
    return (mel - NORM_MEAN) / (NORM_STD * 2)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, default=ROOT / "models/eat-base-as2m")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    args = parser.parse_args()

    files = find_audio_files(args.test_dir)[args.shard_index::args.num_shards]
    model = _load_local_model(args.model_dir, torch.device(args.device))
    ids, embeddings = [], []
    for offset in tqdm(range(0, len(files), args.batch_size), desc="EAT"):
        batch_files = files[offset:offset + args.batch_size]
        batch = torch.stack([fbank(path) for path in batch_files])[:, None].to(args.device)
        with torch.inference_mode():
            # Token zero is the utterance-level representation.
            cls = model.extract_features(batch)[:, 0].float().cpu().numpy()
        ids.extend(path.stem for path in batch_files)
        embeddings.append(cls)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        ids=np.asarray(ids),
        embeddings=np.concatenate(embeddings) if embeddings else np.empty((0, 768)),
    )
    print(f"Saved {len(ids)} embeddings to {args.output}")


if __name__ == "__main__":
    main()
