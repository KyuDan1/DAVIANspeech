#!/usr/bin/env python3
"""Cache compact shallow-layer SPEAR temporal bins from original audio."""

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
from dual_domain_stats import crop_or_pad, temporal_starts  # noqa: E402
from extract_spear_embeddings import load_spear  # noqa: E402
from pipeline import find_audio_files, load_audio  # noqa: E402
from spear_temporal_bins import (  # noqa: E402
    DEFAULT_BINS,
    DEFAULT_LAYERS,
    DEFAULT_SEED,
    DEFAULT_WIDTH,
    audio_bin_ranges,
    projected_temporal_bins,
    random_projection,
)
from telephone_channel import apply_channel  # noqa: E402


SAMPLES = 160_000
MAX_VIEWS = 3


@torch.inference_mode()
def score_batch(model, audios: list[np.ndarray], device: torch.device,
                projection: torch.Tensor, bins: int):
    grouped_views, grouped_ranges, grouped_masks = [], [], []
    for audio in audios:
        starts = temporal_starts(len(audio), SAMPLES, MAX_VIEWS)
        grouped_views.append([crop_or_pad(audio, start, SAMPLES) for start in starts])
        ranges, masks = zip(*[
            audio_bin_ranges(len(audio), start, SAMPLES, bins) for start in starts
        ])
        grouped_ranges.append(list(ranges))
        grouped_masks.append(list(masks))
    waveform = torch.from_numpy(np.stack([
        view for views in grouped_views for view in views
    ])).to(device)
    lengths = torch.full(
        (len(waveform),), SAMPLES, dtype=torch.long, device=device
    )
    output = model(waveform, lengths)
    features = projected_temporal_bins(
        output["hidden_states"], projection, DEFAULT_LAYERS, bins
    )
    results, offset = [], 0
    feature_tail = (bins, len(DEFAULT_LAYERS), 4, projection.shape[1])
    for views, ranges, masks in zip(grouped_views, grouped_ranges, grouped_masks):
        count = len(views)
        padded = np.zeros((MAX_VIEWS, *feature_tail), dtype=np.float16)
        padded_ranges = np.zeros((MAX_VIEWS, bins, 2), dtype=np.int32)
        padded_mask = np.zeros((MAX_VIEWS, bins), dtype=bool)
        padded[:count] = features[offset:offset + count].cpu().numpy().astype(np.float16)
        padded_ranges[:count] = np.asarray(ranges, dtype=np.int32)
        padded_mask[:count] = np.asarray(masks, dtype=bool)
        results.append((padded, padded_ranges, padded_mask))
        offset += count
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--test-dir", type=Path)
    source.add_argument(
        "--datasets", nargs="+",
        help="Repository data/eval dataset names to extract in one model load.",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--channel-variant", default="clean")
    parser.add_argument("--ffmpeg", type=Path, default=Path("ffmpeg"))
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=6)
    parser.add_argument("--bins", type=int, default=DEFAULT_BINS)
    parser.add_argument("--projection-width", type=int, default=DEFAULT_WIDTH)
    parser.add_argument("--projection-seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--ids-csv", type=Path)
    parser.add_argument("--training-truth", type=Path)
    parser.add_argument(
        "--training-datasets", nargs="*", default=[],
        help="Dataset names in --datasets that must pass the no-leakage guard.",
    )
    args = parser.parse_args()
    if args.training_truth is not None:
        assert_no_locked_eval_leakage(
            args.training_truth, ROOT / "configs/data_partitions.yaml"
        )
    for name in args.training_datasets:
        if args.datasets is None or name not in args.datasets:
            parser.error(f"training dataset is not extracted: {name}")
        assert_no_locked_eval_leakage(
            ROOT / "data/eval" / name / "truth.csv",
            ROOT / "configs/data_partitions.yaml",
        )

    if args.datasets is None:
        records = [("", path) for path in find_audio_files(args.test_dir)]
    else:
        records = []
        for name in args.datasets:
            directory = ROOT / "data/eval" / name / "audio"
            if not directory.is_dir():
                raise FileNotFoundError(f"missing dataset audio: {directory}")
            records.extend((name, path) for path in find_audio_files(directory))
    if args.ids_csv is not None:
        allowed = set(pd.read_csv(args.ids_csv, dtype={"ID": str})["ID"])
        records = [(name, path) for name, path in records if path.stem in allowed]
        missing = allowed - {path.stem for _, path in records}
        if missing:
            raise FileNotFoundError(
                f"{len(missing)} requested IDs have no audio: {sorted(missing)[:5]}"
            )
    records = records[args.shard_index::args.num_shards]
    device = torch.device(args.device)
    model = load_spear(ROOT / "models/spear-xlarge-speech-audio-v2", device)
    matrix = random_projection(1280, args.projection_width, args.projection_seed)
    projection = torch.from_numpy(matrix).to(device)
    ids, dataset_names, features, ranges, masks = [], [], [], [], []
    for offset in tqdm(range(0, len(records), args.batch_size), desc="SPEAR temporal bins"):
        batch_records = records[offset:offset + args.batch_size]
        paths = [path for _, path in batch_records]
        audios = []
        for path in paths:
            audio = load_audio(path)
            if args.channel_variant != "clean":
                audio = apply_channel(
                    audio, args.channel_variant, ffmpeg=args.ffmpeg,
                    key=sum(path.stem.encode("utf-8")),
                )
            audios.append(audio)
        for (dataset_name, path), (feature, sample_ranges, mask) in zip(
            batch_records, score_batch(model, audios, device, projection, args.bins)
        ):
            ids.append(path.stem)
            dataset_names.append(dataset_name)
            features.append(feature)
            ranges.append(sample_ranges)
            masks.append(mask)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    tail = (MAX_VIEWS, args.bins, len(DEFAULT_LAYERS), 4, args.projection_width)
    np.savez_compressed(
        args.output,
        ids=np.asarray(ids),
        datasets=np.asarray(dataset_names),
        features=(np.stack(features) if features else np.empty((0, *tail), np.float16)),
        ranges=(np.stack(ranges) if ranges else np.empty((0, MAX_VIEWS, args.bins, 2), np.int32)),
        mask=(np.stack(masks) if masks else np.empty((0, MAX_VIEWS, args.bins), bool)),
        projection=matrix,
        layers=np.asarray(DEFAULT_LAYERS, dtype=np.int16),
        bins=np.asarray(args.bins),
        projection_seed=np.asarray(args.projection_seed),
    )
    print(f"Saved {len(ids)} examples to {args.output}")


if __name__ == "__main__":
    main()
