#!/usr/bin/env python3
"""Audit only the explicit v57 train/development split and rendered audio."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import sys

import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from data_guard import IDENTITY_COLUMNS as REGISTERED_IDENTITY_COLUMNS  # noqa: E402
from train_three_stream_anchor_residual import (  # noqa: E402
    MUSIC_IDENTITY_COLUMNS,
    PARENT_IDENTITY_COLUMNS,
    VOICE_IDENTITY_COLUMNS,
    assert_component_split_contract,
    authorized_partitions,
)


AUDIO_EXTENSIONS = {".aac", ".flac", ".m4a", ".mp3", ".ogg", ".opus", ".wav", ".wma"}
# The generic guard predates several v57 banks. These additional manifest
# columns are genuine source/artist/archive identities rather than domains.
V57_ADDITIONAL_IDENTITY_COLUMNS = (
    "SPEAKER", "FMA_TRACK_ID", "ORIGINAL_AUDIO", "BASE_ID",
)
ALL_IDENTITY_COLUMNS = tuple(dict.fromkeys(
    REGISTERED_IDENTITY_COLUMNS
    + VOICE_IDENTITY_COLUMNS
    + MUSIC_IDENTITY_COLUMNS
    + PARENT_IDENTITY_COLUMNS
    + V57_ADDITIONAL_IDENTITY_COLUMNS
))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_declared(
    matrix_path: Path, partition_config: Path, role: str, key_name: str,
) -> tuple[list[object], pd.DataFrame]:
    matrix = yaml.safe_load(matrix_path.read_text("utf-8")) or {}
    key = matrix.get(key_name)
    requested = matrix.get(key) if isinstance(key, str) else None
    if not isinstance(requested, list) or not requested:
        raise ValueError(f"matrix does not declare {key_name}")
    partitions = authorized_partitions(partition_config, role, requested)
    frames = []
    for partition in partitions:
        frame = pd.read_csv(partition.truth_path, dtype=str)
        frame.insert(0, "DATASET", partition.name)
        frames.append(frame)
    return partitions, pd.concat(frames, ignore_index=True, sort=False)


def identity_tokens(frame: pd.DataFrame) -> tuple[set[str], dict[str, int]]:
    tokens: set[str] = set()
    counts: dict[str, int] = {}
    for column in ALL_IDENTITY_COLUMNS:
        if column not in frame:
            continue
        values = frame[column].dropna().astype(str).str.strip()
        current = set(values.loc[values.ne("")])
        tokens.update(current)
        counts[column] = len(current)
    return tokens, counts


def excluded_rows(
    frame: pd.DataFrame, exclusions: set[str],
) -> tuple[pd.DataFrame, list[str]]:
    rejected = pd.Series(False, index=frame.index)
    for column in ALL_IDENTITY_COLUMNS:
        if column in frame:
            rejected |= frame[column].fillna("").astype(str).str.strip().isin(exclusions)
    return frame.loc[~rejected].reset_index(drop=True), frame.loc[rejected, "ID"].tolist()


def audio_paths(
    partitions: list[object], excluded_ids: set[str] | None = None,
) -> list[tuple[str, str, Path]]:
    records: list[tuple[str, str, Path]] = []
    excluded_ids = excluded_ids or set()
    for partition in partitions:
        ids = pd.read_csv(
            partition.truth_path, usecols=["ID"], dtype={"ID": str},
        ).ID.astype(str).str.strip().tolist()
        directory = partition.truth_path.parent / "audio"
        mapping: dict[str, Path] = {}
        for path in directory.iterdir():
            if path.is_file() and path.suffix.lower() in AUDIO_EXTENSIONS:
                if path.stem in mapping:
                    raise ValueError(f"duplicate audio stem {path.stem}: {directory}")
                mapping[path.stem] = path.resolve(strict=True)
        missing = sorted(set(ids).difference(mapping))
        if missing:
            raise ValueError(f"audio missing for {partition.name}: {missing[:5]}")
        records.extend(
            (partition.name, sample_id, mapping[sample_id])
            for sample_id in ids if sample_id not in excluded_ids
        )
    return records


def hash_audio(
    records: list[tuple[str, str, Path]], workers: int,
) -> tuple[set[str], int]:
    with ThreadPoolExecutor(max_workers=workers) as executor:
        hashes = list(executor.map(lambda item: sha256(item[2]), records))
    return set(hashes), len(hashes)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", type=Path, required=True)
    parser.add_argument("--partition-config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--train-identity-exclusions", type=Path)
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"refusing to overwrite {args.output}")

    train_parts, train = load_declared(
        args.matrix, args.partition_config, "train", "train_set",
    )
    dev_parts, development = load_declared(
        args.matrix, args.partition_config, "development", "development_set",
    )
    exclusions = set()
    if args.train_identity_exclusions is not None:
        exclusions = {
            line.strip()
            for line in args.train_identity_exclusions.resolve(strict=True)
            .read_text("utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        }
    train, excluded_ids = excluded_rows(train, exclusions)
    train_tokens, train_counts = identity_tokens(train)
    dev_tokens, dev_counts = identity_tokens(development)
    identity_overlap = sorted(train_tokens & dev_tokens)
    if identity_overlap:
        raise ValueError(
            f"v57 train/development identity overlap: {identity_overlap[:10]}"
        )
    component = assert_component_split_contract(train, development)

    train_audio = audio_paths(train_parts, set(excluded_ids))
    dev_audio = audio_paths(dev_parts)
    train_hashes, train_rows = hash_audio(train_audio, args.workers)
    dev_hashes, dev_rows = hash_audio(dev_audio, args.workers)
    audio_overlap = sorted(train_hashes & dev_hashes)
    if audio_overlap:
        raise ValueError(
            f"v57 train/development rendered-audio overlap: {audio_overlap[:5]}"
        )

    report = {
        "status": "PASS",
        "scope": "explicit matrix train/development only",
        "matrix_sha256": sha256(args.matrix),
        "partition_config_sha256": sha256(args.partition_config),
        "train": {
            "datasets": [item.name for item in train_parts],
            "rows": len(train), "identity_tokens": len(train_tokens),
            "identity_columns": train_counts,
            "audio_rows": train_rows, "unique_audio_sha256": len(train_hashes),
            "excluded_rows": excluded_ids,
        },
        "development": {
            "datasets": [item.name for item in dev_parts],
            "rows": len(development), "identity_tokens": len(dev_tokens),
            "identity_columns": dev_counts,
            "audio_rows": dev_rows, "unique_audio_sha256": len(dev_hashes),
        },
        "overlap": {"identity_tokens": 0, "rendered_audio_sha256": 0},
        "component_contract": component,
        "non_scope_roles_read": [],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
