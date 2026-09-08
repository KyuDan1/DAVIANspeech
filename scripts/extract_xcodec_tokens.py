#!/usr/bin/env python3
"""Extract compact X-Codec mini token streams into deterministic shards."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from data_guard import assert_no_locked_eval_leakage  # noqa: E402
from pipeline import find_audio_files, load_audio  # noqa: E402
from xcodec_tokens import encode_selected, load_xcodec  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-root", type=Path,
                        default=ROOT / "models/xcodec_mini_infer")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--music-present-only", action="store_true")
    parser.add_argument("--training-datasets", nargs="*", default=[])
    args = parser.parse_args()
    if not 0 <= args.shard_index < args.num_shards:
        parser.error("shard-index must be in [0, num-shards)")

    for name in args.training_datasets:
        if name not in args.datasets:
            parser.error(f"training dataset is not extracted: {name}")
        assert_no_locked_eval_leakage(
            ROOT / "data/eval" / name / "truth.csv",
            ROOT / "configs/data_partitions.yaml",
        )

    entries: list[tuple[str, str, Path]] = []
    for name in args.datasets:
        root = ROOT / "data/eval" / name
        truth = pd.read_csv(root / "truth.csv", dtype={"ID": str})
        if args.music_present_only:
            truth = truth.loc[truth["MUSIC_PRESENT"].eq(1)]
        paths = {path.stem: path for path in find_audio_files(root / "audio")}
        for item in truth["ID"].astype(str):
            path = paths.get(item)
            if path is None:
                raise FileNotFoundError(f"No audio for {name}/{item}")
            entries.append((name, item, path))
    entries = entries[args.shard_index::args.num_shards]

    device = torch.device(args.device)
    model = load_xcodec(args.model_root, device)
    names: list[str] = []
    ids: list[str] = []
    token_batches: list[np.ndarray] = []
    for offset in tqdm(range(0, len(entries), args.batch_size),
                       desc=f"X-Codec shard {args.shard_index}"):
        batch = entries[offset:offset + args.batch_size]
        encoded = encode_selected(model, [load_audio(x[2]) for x in batch], device)
        names.extend(x[0] for x in batch)
        ids.extend(x[1] for x in batch)
        token_batches.append(encoded)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / f"shard_{args.shard_index}.npz"
    tokens = np.concatenate(token_batches) if token_batches else np.empty((0, 4, 500), np.int16)
    np.savez_compressed(
        output,
        datasets=np.asarray(names),
        ids=np.asarray(ids),
        tokens=tokens,
        codebooks=np.asarray([0, 1, 10, 11], dtype=np.int16),
        crop_seconds=np.asarray(10, dtype=np.int16),
    )
    print(f"Saved {len(ids)} examples to {output}", flush=True)


if __name__ == "__main__":
    main()
