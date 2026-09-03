#!/usr/bin/env python3
"""Cache every XLS-R anti-deepfake window score for pooling audits."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import torch
from tqdm import tqdm


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audio-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--xlsr-dir", type=Path,
        default=ROOT / "models" / "xls-r-2b-anti-deepfake",
    )
    parser.add_argument("--source-dir", type=Path, default=ROOT / "src")
    parser.add_argument("--window", type=int, default=64_000)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--include-embeddings", action="store_true",
        help="Also cache the existing 1,920-D pooled XLS-R representation.",
    )
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output}")
    sys.path.insert(0, str(args.source_dir))
    from pipeline import (  # noqa: E402
        extract_segment, find_audio_files, load_audio, segment_starts,
    )
    from xlsr_antideepfake import XlsrAntiDeepfake  # noqa: E402

    files = find_audio_files(args.audio_dir)
    device = torch.device(args.device)
    detector = XlsrAntiDeepfake.from_checkpoint(args.xlsr_dir, device=device)
    ids: list[str] = []
    offsets = [0]
    scores: list[float] = []
    starts_flat: list[int] = []
    embeddings: list[np.ndarray] = []
    durations: list[float] = []
    for path in tqdm(files, desc="XLS-R window profiles"):
        audio = load_audio(path)
        starts = segment_starts(audio.size, args.window)
        windows = np.stack([
            extract_segment(audio, start, args.window) for start in starts
        ])
        item_scores: list[float] = []
        for offset in range(0, len(windows), args.batch_size):
            waveforms = torch.from_numpy(
                windows[offset:offset + args.batch_size]
            ).to(device)
            with torch.inference_mode():
                if args.include_embeddings:
                    pooled = detector.embedding(detector.normalize(waveforms))
                    probabilities = torch.softmax(
                        detector.proj_fc(pooled).float(), dim=-1
                    )[:, 0]
                    embeddings.extend(
                        pooled.to(dtype=torch.float16).cpu().numpy()
                    )
                else:
                    probabilities = detector.fake_probability(waveforms)
                item_scores.extend(probabilities.float().cpu().tolist())
        ids.append(path.stem)
        scores.extend(item_scores)
        starts_flat.extend(starts)
        offsets.append(len(scores))
        durations.append(audio.size / 16_000)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    arrays = dict(
        ids=np.asarray(ids),
        offsets=np.asarray(offsets, dtype=np.int64),
        scores=np.asarray(scores, dtype=np.float32),
        starts=np.asarray(starts_flat, dtype=np.int64),
        durations=np.asarray(durations, dtype=np.float32),
        window=np.asarray(args.window, dtype=np.int64),
    )
    if args.include_embeddings:
        arrays["embeddings"] = np.asarray(embeddings, dtype=np.float16)
    np.savez_compressed(args.output, **arrays)
    print(
        f"Saved {len(ids)} files / {len(scores)} windows to {args.output}",
        flush=True,
    )


if __name__ == "__main__":
    main()
