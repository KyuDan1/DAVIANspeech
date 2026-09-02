#!/usr/bin/env python3
"""Extract source-separation-free MERT embeddings into deterministic shards."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from sofia_mert_detector import SofiaMertDetector  # noqa: E402
from train_dual_domain_head import truth_path  # noqa: E402


def audio_dir(name: str) -> Path:
    if name.startswith("factorial_eval_1200_v2"):
        return ROOT / "data/eval/factorial_eval_1200_v2/audio"
    return ROOT / "data/eval" / name / "audio"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", required=True)
    parser.add_argument("--model-root", type=Path,
                        default=ROOT / "models/sofia-mert-v1")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--shard-index", type=int, required=True)
    parser.add_argument("--num-shards", type=int, required=True)
    args = parser.parse_args()
    if not 0 <= args.shard_index < args.num_shards:
        parser.error("shard-index must be in [0, num-shards)")

    entries = []
    extensions = {".aac", ".flac", ".m4a", ".mp3", ".ogg", ".opus", ".wav"}
    for name in args.datasets:
        truth = pd.read_csv(truth_path(name), dtype={"ID": str})
        paths = {
            path.stem: path for path in audio_dir(name).iterdir()
            if path.is_file() and path.suffix.lower() in extensions
        }
        for row in truth.itertuples(index=False):
            path = paths.get(str(row.ID))
            if path is None:
                raise FileNotFoundError(f"No audio for {name}/{row.ID}")
            entries.append((name, str(row.ID), path))
    selected = [
        item for index, item in enumerate(entries)
        if index % args.num_shards == args.shard_index
    ]
    detector = SofiaMertDetector(
        args.model_root / "mert", args.model_root / "sofia_g1_mert_head.pt",
        device=args.device, pad_to_release_length=False,
    )
    names, ids, embeddings = [], [], []
    for index, (name, item, path) in enumerate(selected, start=1):
        names.append(name)
        ids.append(item)
        embeddings.append(detector.embed_path(path))
        if index % 100 == 0:
            print(f"shard {args.shard_index}: {index}/{len(selected)}", flush=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / f"shard_{args.shard_index}.npz"
    np.savez_compressed(
        output, datasets=np.asarray(names), ids=np.asarray(ids),
        embeddings=np.asarray(embeddings, dtype=np.float32),
    )
    print(output)


if __name__ == "__main__":
    main()
