#!/usr/bin/env python3
"""Cache all-layer EAT statistics from original, unseparated audio."""

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
sys.path.insert(0, str(ROOT / "scripts"))

from data_guard import assert_no_locked_eval_leakage  # noqa: E402
from dual_domain_stats import crop_or_pad, pad_views, temporal_starts  # noqa: E402
from eat_detector import EatMusicDetector, _load_local_model  # noqa: E402
from eat_hierarchical import gaussian_projection, hierarchical_statistics  # noqa: E402
from pipeline import find_audio_files, load_audio  # noqa: E402


MAX_VIEWS = 3
SAMPLES = EatMusicDetector.SAMPLES


@torch.inference_mode()
def score_batch(
    model,
    audios: list[np.ndarray],
    projection: torch.Tensor,
    device: torch.device,
) -> list[tuple[np.ndarray, np.ndarray]]:
    grouped = []
    for audio in audios:
        starts = temporal_starts(len(audio), SAMPLES, MAX_VIEWS)
        grouped.append([crop_or_pad(audio, start, SAMPLES) for start in starts])
    features = torch.stack([
        EatMusicDetector._fbank(view) for views in grouped for view in views
    ])[:, None].to(device)
    statistics = hierarchical_statistics(model, features, projection)
    result, offset = [], 0
    tail = tuple(statistics.shape[1:])
    for views in grouped:
        count = len(views)
        result.append(pad_views(
            [statistics[index] for index in range(offset, offset + count)],
            MAX_VIEWS,
            tail,
        ))
        offset += count
    return result


def extract_files(
    model,
    projection: torch.Tensor,
    files: list[Path],
    output: Path,
    device: torch.device,
    batch_size: int = 8,
) -> None:
    """Extract one ordered file list using an already-loaded encoder."""
    if batch_size <= 0:
        raise ValueError("batch size must be positive")
    ids, matrices, masks = [], [], []
    for offset in tqdm(
        range(0, len(files), batch_size),
        desc=f"hierarchical EAT statistics ({output.parent.parent.name})",
    ):
        batch = files[offset:offset + batch_size]
        audios = [load_audio(path) for path in batch]
        for path, (matrix, mask) in zip(
            batch, score_batch(model, audios, projection, device)
        ):
            ids.append(path.stem)
            matrices.append(matrix)
            masks.append(mask)
    output.parent.mkdir(parents=True, exist_ok=True)
    shape = (MAX_VIEWS, len(model.model.blocks), 5, projection.shape[1])
    np.savez_compressed(
        output,
        ids=np.asarray(ids),
        statistics=(np.stack(matrices) if matrices else np.empty((0, *shape), np.float16)),
        view_mask=(np.stack(masks) if masks else np.empty((0, MAX_VIEWS), bool)),
        projection=projection.numpy(),
    )
    print(f"Saved {len(ids)} hierarchical EAT examples to {output}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--ids-csv", type=Path)
    parser.add_argument("--training-truth", type=Path)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--projection-width", type=int, default=128)
    parser.add_argument("--projection-seed", type=int, default=20260904)
    args = parser.parse_args()
    if args.num_shards <= 0 or not 0 <= args.shard_index < args.num_shards:
        parser.error("invalid shard configuration")
    if args.training_truth is not None:
        assert_no_locked_eval_leakage(
            args.training_truth, ROOT / "configs/data_partitions.yaml"
        )

    files = find_audio_files(args.test_dir)
    if args.ids_csv is not None:
        allowed = set(pd.read_csv(args.ids_csv, dtype={"ID": str}).ID)
        files = [path for path in files if path.stem in allowed]
        missing = allowed.difference(path.stem for path in files)
        if missing:
            raise FileNotFoundError(
                f"{len(missing)} IDs have no audio: {sorted(missing)[:5]}"
            )
    files = files[args.shard_index::args.num_shards]
    device = torch.device(args.device)
    model = _load_local_model(ROOT / "models/eat-base-as2m", device)
    projection = gaussian_projection(
        output_width=args.projection_width, seed=args.projection_seed
    )
    extract_files(
        model, projection, files, args.output, device, args.batch_size,
    )


if __name__ == "__main__":
    main()
