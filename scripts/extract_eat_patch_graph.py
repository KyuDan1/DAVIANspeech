#!/usr/bin/env python3
"""Cache patch-preserving EAT time/frequency nodes from original audio."""

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
from eat_patch_graph import (  # noqa: E402
    DEFAULT_LAYERS, gaussian_patch_projection, patch_graph_features,
)
from pipeline import find_audio_files, load_audio  # noqa: E402


SAMPLES = EatMusicDetector.SAMPLES


@torch.inference_mode()
def score_batch(
    model,
    audios: list[np.ndarray],
    projection: torch.Tensor,
    device: torch.device,
    layers: tuple[int, ...] = DEFAULT_LAYERS,
    max_views: int = 3,
) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    grouped = []
    for audio in audios:
        starts = temporal_starts(len(audio), SAMPLES, max_views)
        grouped.append([crop_or_pad(audio, start, SAMPLES) for start in starts])
    features = torch.stack([
        EatMusicDetector._fbank(view) for views in grouped for view in views
    ])[:, None].to(device)
    temporal, spectral = patch_graph_features(
        model, features, projection, layers
    )
    result, offset = [], 0
    temporal_tail = tuple(temporal.shape[1:])
    spectral_tail = tuple(spectral.shape[1:])
    for views in grouped:
        count = len(views)
        temporal_matrix, mask = pad_views(
            [temporal[index] for index in range(offset, offset + count)],
            max_views, temporal_tail,
        )
        spectral_matrix, spectral_mask = pad_views(
            [spectral[index] for index in range(offset, offset + count)],
            max_views, spectral_tail,
        )
        if not np.array_equal(mask, spectral_mask):
            raise RuntimeError("EAT patch graph view masks differ")
        result.append((temporal_matrix, spectral_matrix, mask))
        offset += count
    return result


def extract_files(
    model,
    projection: torch.Tensor,
    files: list[Path],
    output: Path,
    device: torch.device,
    layers: tuple[int, ...],
    batch_size: int = 8,
    max_views: int = 3,
) -> None:
    ids, temporal, spectral, masks = [], [], [], []
    shapes = None
    for offset in tqdm(
        range(0, len(files), batch_size),
        desc=f"EAT patch graph ({output.parent.name})",
    ):
        batch = files[offset:offset + batch_size]
        audios = [load_audio(path) for path in batch]
        results = score_batch(
            model, audios, projection, device, layers, max_views
        )
        for path, (time_nodes, frequency_nodes, mask) in zip(batch, results):
            ids.append(path.stem)
            temporal.append(time_nodes)
            spectral.append(frequency_nodes)
            masks.append(mask)
            shapes = (time_nodes.shape, frequency_nodes.shape)
    if shapes is None:
        raise ValueError("no audio files were selected")
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        ids=np.asarray(ids),
        temporal=np.stack(temporal).astype(np.float16),
        spectral=np.stack(spectral).astype(np.float16),
        view_mask=np.stack(masks),
        projection=projection.numpy(),
        layers=np.asarray(layers, dtype=np.int64),
    )
    print(f"Saved {len(ids)} EAT patch-graph examples to {output}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ids-csv", type=Path)
    parser.add_argument("--training-truth", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-views", type=int, default=3)
    parser.add_argument("--layers", type=int, nargs="+", default=list(DEFAULT_LAYERS))
    parser.add_argument("--projection-width", type=int, default=128)
    parser.add_argument("--projection-seed", type=int, default=20260904)
    args = parser.parse_args()
    if args.batch_size <= 0 or args.max_views <= 0:
        parser.error("batch size and max views must be positive")
    layers = tuple(args.layers)
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
    device = torch.device(args.device)
    model = _load_local_model(ROOT / "models/eat-base-as2m", device)
    projection = gaussian_patch_projection(
        output_width=args.projection_width, seed=args.projection_seed
    )
    extract_files(
        model, projection, files, args.output, device, layers,
        args.batch_size, args.max_views,
    )


if __name__ == "__main__":
    main()
