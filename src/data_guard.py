"""Prevent locked evaluation samples or their source components from training."""

from __future__ import annotations

from pathlib import Path
import argparse

import pandas as pd
import yaml


IDENTITY_COLUMNS = (
    "ID", "GROUP_ID", "VOICE_SOURCE_ID", "MUSIC_SOURCE_ID", "SOURCE_FILE",
)


def identity_tokens(frame: pd.DataFrame) -> set[str]:
    tokens: set[str] = set()
    for column in IDENTITY_COLUMNS:
        if column in frame:
            tokens.update(frame[column].dropna().astype(str))
    return tokens


def assert_no_locked_eval_leakage(
    training_truth: Path,
    partition_config: Path,
) -> None:
    """Raise if training rows reuse a locked sample or source component."""
    config = yaml.safe_load(partition_config.read_text("utf-8"))
    config_dir = partition_config.resolve().parent
    root = config_dir.parent if config_dir.name == "configs" else config_dir
    protected_roles = ("development", "locked_eval", "ood_holdout", "stress_eval")
    locked_paths = [
        root / path
        for role in protected_roles
        for path in config.get(role, [])
    ]
    if not locked_paths:
        raise ValueError("Partition config must contain protected evaluation data")

    train_tokens = identity_tokens(pd.read_csv(training_truth, dtype=str))
    locked_tokens: set[str] = set()
    for path in locked_paths:
        if not path.is_file():
            raise FileNotFoundError(f"Locked truth is missing: {path}")
        locked_tokens.update(identity_tokens(pd.read_csv(path, dtype=str)))
    overlap = sorted(train_tokens & locked_tokens)
    if overlap:
        raise ValueError(
            f"TRAIN/EVAL LEAKAGE: {len(overlap)} locked identity tokens reused: "
            f"{overlap[:10]}"
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/data_partitions.yaml"))
    parser.add_argument("--training-truth", type=Path, action="append", default=[])
    args = parser.parse_args()
    paths = args.training_truth
    if not paths:
        config = yaml.safe_load(args.config.read_text("utf-8"))
        config_dir = args.config.resolve().parent
        root = config_dir.parent if config_dir.name == "configs" else config_dir
        paths = [
            root / path
            for role in ("train", "router_train")
            for path in config.get(role, [])
        ]
    if not paths:
        raise ValueError("No training truth files were declared")
    for path in paths:
        assert_no_locked_eval_leakage(path, args.config)
        print(f"PASS: {path}")


if __name__ == "__main__":
    main()
