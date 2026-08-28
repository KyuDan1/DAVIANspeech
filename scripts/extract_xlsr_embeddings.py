"""Cache mean-pooled AntiDeepfake embeddings on original audio.

The released classifier head was post-trained only on speech.  Caching its
foundation representation lets us test the paper's recommended task-specific
fine-tuning strategy with a small music head, without modifying the 2B encoder.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from pipeline import find_audio_files, load_audio  # noqa: E402
from presence import extract_segment, segment_starts  # noqa: E402
from xlsr_antideepfake import XlsrAntiDeepfake  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path,
                        default=ROOT / "models/xls-r-2b-anti-deepfake")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--window", type=int, default=64_000)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    args = parser.parse_args()

    files = find_audio_files(args.test_dir)[args.shard_index::args.num_shards]
    device = torch.device(args.device)
    model = XlsrAntiDeepfake.from_checkpoint(args.model_dir, device=device)
    ids, vectors = [], []
    for path in tqdm(files, desc="XLS-R embeddings"):
        audio = load_audio(path)
        per_window = []
        for start in segment_starts(len(audio), args.window):
            waveform = torch.from_numpy(
                extract_segment(audio, start, args.window)
            ).unsqueeze(0).to(device)
            with torch.inference_mode():
                vector = model.embedding(model.normalize(waveform))
            per_window.append(vector.float().cpu().numpy()[0])
        ids.append(path.stem)
        vectors.append(np.mean(per_window, axis=0))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, ids=np.asarray(ids), embeddings=np.asarray(vectors))
    print(f"Saved {len(ids)} embeddings to {args.output}")


if __name__ == "__main__":
    main()
