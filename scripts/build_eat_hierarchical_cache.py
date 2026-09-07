#!/usr/bin/env python3
"""Build train/development/locked hierarchical EAT caches with one model load."""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from data_guard import assert_no_locked_eval_leakage  # noqa: E402
from eat_detector import _load_local_model  # noqa: E402
from eat_hierarchical import gaussian_projection  # noqa: E402
from extract_eat_hierarchical_stats import extract_files  # noqa: E402
from pipeline import find_audio_files  # noqa: E402


@dataclass(frozen=True)
class Specification:
    name: str
    audio_dataset: str
    truth: Path
    train: bool = False


def direct(name: str, train: bool = False) -> Specification:
    return Specification(
        name, name, ROOT / "data" / "eval" / name / "truth.csv", train
    )


SPECS = (
    direct("external_mixed_train_v1", True),
    direct("mixed_devvoice_train_v1", True),
    direct("mixed_fmc_music_train_v1", True),
    direct("mixfake_music_train_v1", True),
    direct("telephone_mixed_train_v1", True),
    direct("temporal_mixed_train_v2", True),
    direct("channel_invariant_factorial_train_v1", True),
    direct("multigen_music_presence_train_v1", True),
    direct("phone_presence_factorial_train_v1", True),
    direct("mixfake_music_dev_v1"),
    direct("external_mixed_v1"),
    direct("external_mixed_v1_telephone_v1"),
    direct("source_disjoint_mixed_v1"),
    direct("source_disjoint_mixed_v1_telephone_v1"),
    direct("source_disjoint_mixed_equal_v1"),
    direct("source_disjoint_mixed_equal_v1_telephone_v1"),
    direct("source_disjoint_music_v1"),
    Specification(
        "factorial_eval_1200_v2_dev", "factorial_eval_1200_v2",
        ROOT / "data/eval/factorial_eval_1200_v2/truth_dev.csv",
    ),
    direct("telephone_mixed_dev_v1"),
    Specification(
        "factorial_eval_1200_v2_holdout", "factorial_eval_1200_v2",
        ROOT / "data/eval/factorial_eval_1200_v2/truth_holdout.csv",
    ),
    direct("phone_factorial_1200_v1"),
    direct("yue_cross_component_audit_v1"),
    direct("source_disjoint_music_telephone_v1"),
    direct("suno_vocals_v1"),
    direct("suno_vocals_codec_stress_v1"),
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-root", type=Path,
        default=ROOT / "output/eat_hierarchical_stats_v1",
    )
    parser.add_argument("--datasets", nargs="+")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--projection-width", type=int, default=128)
    parser.add_argument("--projection-seed", type=int, default=20260904)
    parser.add_argument("--max-views", type=int, default=3)
    parser.add_argument(
        "--view-strategy", choices=("endpoints", "segments"), default="endpoints"
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    selected = set(args.datasets or [spec.name for spec in SPECS])
    unknown = selected.difference(spec.name for spec in SPECS)
    if unknown:
        parser.error(f"unknown datasets: {sorted(unknown)}")
    if args.max_views <= 0:
        parser.error("--max-views must be positive")

    device = torch.device(args.device)
    model = _load_local_model(ROOT / "models/eat-base-as2m", device)
    projection = gaussian_projection(
        output_width=args.projection_width, seed=args.projection_seed
    )
    for spec in SPECS:
        if spec.name not in selected:
            continue
        output = args.output_root / spec.name / "shard_0.npz"
        if output.exists() and not args.overwrite:
            print(f"Keeping existing {output}")
            continue
        if spec.train:
            assert_no_locked_eval_leakage(
                spec.truth, ROOT / "configs/data_partitions.yaml"
            )
        truth = pd.read_csv(spec.truth, dtype={"ID": str})
        directory = ROOT / "data/eval" / spec.audio_dataset / "audio"
        by_id = {path.stem: path for path in find_audio_files(directory)}
        missing = set(truth.ID).difference(by_id)
        if missing:
            raise FileNotFoundError(
                f"{spec.name} has {len(missing)} missing audio files: "
                f"{sorted(missing)[:5]}"
            )
        files = [by_id[item] for item in truth.ID]
        extract_files(
            model, projection, files, output, device, args.batch_size,
            max_views=args.max_views, view_strategy=args.view_strategy,
        )


if __name__ == "__main__":
    main()
