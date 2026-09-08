#!/usr/bin/env python3
"""Materialize 16 kHz ArtifactBench AI music with source-disjoint roles.

The original Parquet audio is decoded and resampled once, but never cropped;
training can therefore choose a fresh four-second view each epoch.  AIME and
MoM collections are eligible for training.  SONICS and post-freeze CDN/extra
collections are held out as a cross-collection/new-version audit.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from io import BytesIO
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import soundfile as sf
import soxr


ROOT = Path(__file__).resolve().parents[1]
TRAIN_PREFIXES = ("aime_", "mom_")
HOLDOUT_PREFIXES = ("sonics_", "suno_", "udio_")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def decode(blob: bytes) -> tuple[np.ndarray, int]:
    audio, sample_rate = sf.read(BytesIO(blob), dtype="float32", always_2d=True)
    audio = audio.mean(axis=1)
    if sample_rate != 16_000:
        audio = soxr.resample(audio, sample_rate, 16_000, quality="HQ")
    audio = np.asarray(audio, dtype=np.float32)
    if not len(audio) or not np.isfinite(audio).all():
        raise ValueError("invalid embedded audio")
    return audio, int(sample_rate)


def role(source: str) -> str:
    if source.startswith(TRAIN_PREFIXES):
        return "train"
    if source.startswith(HOLDOUT_PREFIXES):
        return "holdout"
    raise ValueError(f"unrecognized source {source!r}")


def materialize(args: argparse.Namespace) -> None:
    shards = sorted(args.input.glob("shard_*.parquet"))
    selected = shards[args.shard_index :: args.num_shards]
    if not selected:
        raise ValueError("worker received no shards")
    args.output.mkdir(parents=True, exist_ok=True)
    clips = args.output / "audio"
    clips.mkdir(exist_ok=True)
    rows = []
    for shard in selected:
        table = pq.read_table(shard, columns=[
            "track_id", "audio_bytes", "source", "generator", "format",
        ]).to_pydict()
        for index in range(len(table["track_id"])):
            item_id = "artifactbench_" + str(table["track_id"][index])
            destination = clips / f"{item_id}.flac"
            audio, source_rate = decode(table["audio_bytes"][index])
            sf.write(destination, audio, 16_000, format="FLAC", subtype="PCM_16")
            source = str(table["source"][index])
            rows.append({
                "ID": item_id, "PATH": str(destination.resolve()),
                "SOURCE": source, "GENERATOR": str(table["generator"][index]),
                "ORIGINAL_FORMAT": str(table["format"][index]),
                "ORIGINAL_SAMPLE_RATE": source_rate,
                "DURATION": len(audio) / 16_000,
                "ROLE": role(source), "PARQUET": shard.name,
                "AUDIO_SHA256": sha256(destination),
            })
    frame = pd.DataFrame(rows)
    frame.to_csv(args.output / f"metadata_{args.shard_index:02d}.csv", index=False)
    print(json.dumps({
        "worker": args.shard_index, "shards": len(selected), "rows": len(frame),
        "role_counts": frame.ROLE.value_counts().to_dict(),
    }, sort_keys=True), flush=True)


def finalize(args: argparse.Namespace) -> None:
    parts = sorted(args.output.glob("metadata_*.csv"))
    if len(parts) != args.num_shards:
        raise ValueError(f"expected {args.num_shards} metadata parts, found {len(parts)}")
    metadata = pd.concat([pd.read_csv(path, dtype={"ID": str}) for path in parts],
                         ignore_index=True)
    if len(metadata) != 4_400 or metadata.ID.duplicated().any():
        raise ValueError("unexpected ArtifactBench cardinality or duplicate ID")
    if set(metadata.ROLE) != {"train", "holdout"}:
        raise ValueError("both roles are required")
    if missing := [path for path in metadata.PATH if not Path(path).is_file()]:
        raise FileNotFoundError(missing[0])
    metadata.sort_values("ID").to_csv(args.output / "metadata.csv", index=False)

    columns = pd.read_csv(args.base_train, nrows=0).columns
    base = pd.read_csv(args.base_train, dtype={"ID": str}, low_memory=False)
    train_metadata = metadata.loc[metadata.ROLE.eq("train")].copy()
    cross_role_hashes = metadata.groupby("AUDIO_SHA256").ROLE.nunique()
    cross_role_hashes = cross_role_hashes.loc[cross_role_hashes.gt(1)].index
    if len(cross_role_hashes):
        raise ValueError(
            f"{len(cross_role_hashes)} audio hashes cross the train/holdout boundary"
        )
    duplicate_train_rows = int(train_metadata.AUDIO_SHA256.duplicated().sum())
    train_metadata = train_metadata.drop_duplicates("AUDIO_SHA256", keep="first")

    additions = []
    for row in train_metadata.itertuples(index=False):
        item = {column: np.nan for column in columns}
        item.update({
            "ID": row.ID, "FILE_FAKE": 1, "VOICE_FAKE": np.nan,
            "MUSIC_FAKE": 1, "VOICE_PRESENT": np.nan, "MUSIC_PRESENT": 1,
            "AUDIO_TYPE": "music_or_song", "SPLIT": "train",
            "MUSIC_SOURCE_ID": row.ID, "MUSIC_SOURCE_BANK": "artifactbench_v1",
            "MUSIC_GENERATOR": f"artifactbench:{row.SOURCE}",
            "MUSIC_GROUP": row.ID, "PARENT_ID": row.ID,
            "DURATION": row.DURATION, "ROLE": "external_train",
            "DATASET": "artifactbench_v1_train", "PATH": row.PATH,
            "COMPONENT_CASE": "AB_FAKE", "SOURCE": row.SOURCE,
            "SOURCE_BANK": "artifactbench_v1", "CHANNEL": "original_16k",
        })
        additions.append(item)
    augmented = pd.concat([base, pd.DataFrame(additions, columns=columns)],
                          ignore_index=True)
    if augmented.ID.duplicated().any() or set(base.ID) & set(metadata.ID):
        raise ValueError("augmented training IDs overlap")
    augmented.to_csv(args.output / "train_augmented.csv", index=False)

    holdout = metadata.loc[metadata.ROLE.eq("holdout")].copy()
    holdout["target"] = 1
    holdout[["ID", "PATH", "target", "SOURCE", "GENERATOR", "DURATION"]].to_csv(
        args.output / "holdout_fake.csv", index=False,
    )
    holdout_eval = pd.DataFrame({
        "ID": holdout.ID,
        "FILE_FAKE": 1,
        "VOICE_FAKE": np.nan,
        "MUSIC_FAKE": 1,
        "VOICE_PRESENT": np.nan,
        "MUSIC_PRESENT": 1,
        "AUDIO_TYPE": "music_or_song",
        "SOURCE": holdout.SOURCE,
        "GENERATOR": holdout.GENERATOR,
        "DATASET": "artifactbench_v1_holdout",
        "PATH": holdout.PATH,
        "COMPONENT_CASE": "AB_FAKE_HOLDOUT",
        "CHANNEL": "original_16k",
        "DURATION": holdout.DURATION,
    })
    holdout_eval.to_csv(args.output / "holdout_fake_eval.csv", index=False)
    report = {
        "schema": "artifactbench_external_v106", "rows": len(metadata),
        "train_rows": int(metadata.ROLE.eq("train").sum()),
        "train_unique_audio_rows": len(train_metadata),
        "train_duplicate_audio_rows_excluded": duplicate_train_rows,
        "holdout_rows": int(metadata.ROLE.eq("holdout").sum()),
        "train_sources": sorted(metadata.loc[metadata.ROLE.eq("train"), "SOURCE"].unique()),
        "holdout_sources": sorted(metadata.loc[metadata.ROLE.eq("holdout"), "SOURCE"].unique()),
        "base_train_rows": len(base), "augmented_train_rows": len(augmented),
        "base_train_sha256": sha256(args.base_train),
        "train_augmented_sha256": sha256(args.output / "train_augmented.csv"),
        "audio_sha256_unique": bool(metadata.AUDIO_SHA256.is_unique),
        "license": "CC-BY-NC-4.0",
        "source": "https://huggingface.co/datasets/intrect/artifactbench",
    }
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path,
                        default=ROOT / "data/external/artifactbench_v1/ai_tracks")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "data/external/artifactbench_v1/prepared_v106")
    parser.add_argument("--base-train", type=Path,
                        default=ROOT / "reports/training_inventory_v78/train.csv")
    parser.add_argument("--num-shards", type=int, default=8)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--finalize", action="store_true")
    args = parser.parse_args()
    if args.finalize:
        finalize(args)
    else:
        if not 0 <= args.shard_index < args.num_shards:
            parser.error("invalid shard index")
        materialize(args)


if __name__ == "__main__":
    main()
