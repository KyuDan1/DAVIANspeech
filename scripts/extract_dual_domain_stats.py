#!/usr/bin/env python3
"""Cache temporal EAT or SPEAR token statistics from original audio mixtures."""

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
from dual_domain_stats import (  # noqa: E402
    crop_or_pad,
    pad_views,
    sequence_statistics,
    temporal_starts,
)
from eat_detector import EatMusicDetector, _load_local_model  # noqa: E402
from extract_spear_embeddings import load_spear  # noqa: E402
from pipeline import find_audio_files, load_audio  # noqa: E402
from telephone_channel import apply_channel  # noqa: E402


SPEAR_SAMPLES = 160_000
EAT_SAMPLES = EatMusicDetector.SAMPLES
MAX_VIEWS = 3


def eat_stats_batch(
    model, audios: list[np.ndarray], device: torch.device
) -> list[tuple[np.ndarray, np.ndarray]]:
    grouped_views = []
    for audio in audios:
        starts = temporal_starts(len(audio), EAT_SAMPLES, MAX_VIEWS)
        grouped_views.append([crop_or_pad(audio, start, EAT_SAMPLES) for start in starts])
    mel = torch.stack([
        EatMusicDetector._fbank(view)
        for views in grouped_views for view in views
    ])[:, None].to(device)
    with torch.inference_mode():
        # Exclude CLS: frame/patch tokens retain local texture information.
        tokens = model.extract_features(mel)[:, 1:]
        stats = sequence_statistics(tokens)
    result, offset = [], 0
    for views in grouped_views:
        count = len(views)
        values = [stats[index] for index in range(offset, offset + count)]
        result.append(pad_views(values, MAX_VIEWS, (4, 768)))
        offset += count
    return result


def spear_stats_batch(
    model, audios: list[np.ndarray], device: torch.device
) -> list[tuple[np.ndarray, np.ndarray]]:
    grouped_views = []
    for audio in audios:
        starts = temporal_starts(len(audio), SPEAR_SAMPLES, MAX_VIEWS)
        grouped_views.append([
            crop_or_pad(audio, start, SPEAR_SAMPLES) for start in starts
        ])
    waveform = torch.from_numpy(np.stack([
        view for views in grouped_views for view in views
    ])).to(device)
    lengths = torch.full(
        (waveform.shape[0],), waveform.shape[1], dtype=torch.long, device=device
    )
    with torch.inference_mode():
        output = model(waveform, lengths)
        # [views, layers, time, channels] -> [views, layers, stats, channels]
        layers = torch.stack(output["hidden_states"], dim=1)
        stats = sequence_statistics(layers)
    result, offset = [], 0
    for views in grouped_views:
        count = len(views)
        values = [stats[index] for index in range(offset, offset + count)]
        result.append(pad_views(values, MAX_VIEWS, (13, 4, 1280)))
        offset += count
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stream", choices=("eat", "spear"), required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--channel-variant", default="clean")
    parser.add_argument("--ffmpeg", type=Path, default=Path("ffmpeg"))
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument(
        "--ids-csv", type=Path,
        help="Optional manifest whose ID column limits extraction to an explicit split.",
    )
    parser.add_argument(
        "--training-truth", type=Path,
        help="When extracting train data, enforce the configured no-leakage guard.",
    )
    args = parser.parse_args()
    if args.training_truth is not None:
        assert_no_locked_eval_leakage(
            args.training_truth, ROOT / "configs" / "data_partitions.yaml"
        )

    files = find_audio_files(args.test_dir)
    if args.ids_csv is not None:
        allowed = set(pd.read_csv(args.ids_csv, dtype={"ID": str})["ID"])
        files = [path for path in files if path.stem in allowed]
        missing = allowed - {path.stem for path in files}
        if missing:
            raise FileNotFoundError(
                f"{len(missing)} requested IDs have no audio file: {sorted(missing)[:5]}"
            )
    files = files[args.shard_index::args.num_shards]
    device = torch.device(args.device)
    if args.stream == "eat":
        model = _load_local_model(ROOT / "models/eat-base-as2m", device)
        scorer = eat_stats_batch
    else:
        model = load_spear(ROOT / "models/spear-xlarge-speech-audio-v2", device)
        scorer = spear_stats_batch

    ids, matrices, masks = [], [], []
    for offset in tqdm(
        range(0, len(files), args.batch_size),
        desc=f"{args.stream.upper()} token statistics",
    ):
        batch_paths = files[offset:offset + args.batch_size]
        audios = []
        for path in batch_paths:
            audio = load_audio(path)
            if args.channel_variant != "clean":
                audio = apply_channel(
                    audio, args.channel_variant, ffmpeg=args.ffmpeg,
                    key=sum(path.stem.encode("utf-8")),
                )
            audios.append(audio)
        for path, (matrix, mask) in zip(
            batch_paths, scorer(model, audios, device)
        ):
            ids.append(path.stem)
            matrices.append(matrix)
            masks.append(mask)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    if matrices:
        statistics = np.stack(matrices)
        view_mask = np.stack(masks)
    else:
        tail = (4, 768) if args.stream == "eat" else (13, 4, 1280)
        statistics = np.empty((0, MAX_VIEWS, *tail), dtype=np.float16)
        view_mask = np.empty((0, MAX_VIEWS), dtype=bool)
    np.savez_compressed(
        args.output,
        ids=np.asarray(ids), statistics=statistics, view_mask=view_mask,
        stream=np.asarray(args.stream), channel=np.asarray(args.channel_variant),
    )
    print(f"Saved {len(ids)} {args.stream} examples to {args.output}")


if __name__ == "__main__":
    main()
