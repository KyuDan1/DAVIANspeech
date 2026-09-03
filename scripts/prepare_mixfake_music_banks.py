#!/usr/bin/env python3
"""Create balanced, protected MixFake music-mixture train/dev manifests.

The source archive remains external data.  This script writes only deterministic
manifests and archive target lists; extraction is a separate, resumable step.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from data_guard import identity_tokens  # noqa: E402


def music_generator(path: str, authenticity: str) -> str:
    if authenticity == "bonafide":
        return "FMA"
    value = Path(path)
    if "FakeMusicCaps" in value.parts:
        return value.parent.name
    stem = value.stem.lower()
    for name in ("suno", "udio"):
        if f"_{name}_" in stem:
            return name
    return "unknown"


def protected_tokens(config_path: Path) -> set[str]:
    config = yaml.safe_load(config_path.read_text("utf-8"))
    result: set[str] = set()
    for role in ("development", "ood_holdout", "stress_eval", "locked_eval"):
        for relative in config.get(role, []):
            path = ROOT / relative
            if path.is_file():
                result.update(identity_tokens(pd.read_csv(path, dtype=str)))
    return result


def balanced_sample(frame: pd.DataFrame, per_cell: int, seed: int) -> pd.DataFrame:
    selected = []
    for cell, group in frame.groupby("CELL", sort=True):
        rng = np.random.default_rng(seed + sum(cell.encode("utf-8")))
        if group.MUSIC_FAKE.iloc[0] == 1:
            generators = sorted(group.MUSIC_GENERATOR.unique())
            target = per_cell // len(generators)
            parts = []
            for generator in generators:
                candidates = group[group.MUSIC_GENERATOR.eq(generator)]
                take = min(target, len(candidates))
                parts.append(candidates.iloc[rng.choice(len(candidates), take, replace=False)])
            choice = pd.concat(parts)
            remaining = group.drop(choice.index)
            needed = per_cell - len(choice)
            if needed > 0:
                choice = pd.concat([
                    choice,
                    remaining.iloc[rng.choice(len(remaining), needed, replace=False)],
                ])
        else:
            choice = group.iloc[rng.choice(len(group), per_cell, replace=False)]
        selected.append(choice)
    return pd.concat(selected).sort_values("ID").reset_index(drop=True)


def build_bank(
    details: pd.DataFrame, split: str, per_cell: int, output_dir: Path,
    blocked: set[str], seed: int,
) -> dict[str, object]:
    frame = details[
        details.split.eq(split) & details.back_type.eq("music")
    ].copy()
    frame["ID"] = frame.syn_file.map(lambda value: Path(value).stem)
    frame["VOICE_FAKE"] = frame.fore_authenticity.eq("spoof").astype(int)
    frame["MUSIC_FAKE"] = frame.back_authenticity.eq("spoof").astype(int)
    frame["FILE_FAKE"] = frame[["VOICE_FAKE", "MUSIC_FAKE"]].max(axis=1)
    frame["VOICE_SOURCE_ID"] = frame.fore_file.map(lambda value: Path(value).stem)
    frame["MUSIC_SOURCE_ID"] = frame.back_file.map(lambda value: Path(value).stem)
    frame["MUSIC_GENERATOR"] = [
        music_generator(path, authenticity)
        for path, authenticity in zip(frame.back_file, frame.back_authenticity)
    ]
    frame = frame[~frame.MUSIC_GENERATOR.eq("unknown")]
    identity_columns = ["ID", "VOICE_SOURCE_ID", "MUSIC_SOURCE_ID"]
    overlap = frame[identity_columns].isin(blocked).any(axis=1)
    frame = frame[~overlap].copy()
    frame["CELL"] = (
        frame.VOICE_FAKE.map({0: "R", 1: "F"})
        + frame.MUSIC_FAKE.map({0: "R", 1: "F"})
    )
    frame = balanced_sample(frame, per_cell, seed)

    truth = pd.DataFrame({
        "ID": frame.ID,
        "FILE_FAKE": frame.FILE_FAKE,
        "VOICE_FAKE": frame.VOICE_FAKE,
        "MUSIC_FAKE": frame.MUSIC_FAKE,
        "VOICE_PRESENT": 1,
        "MUSIC_PRESENT": 1,
        "AUDIO_TYPE": "mixed",
        "CONDITION": "concurrent",
        "SOURCE": "MixFake",
        "CODEC": "wav16k",
        "VOICE_SOURCE_ID": frame.VOICE_SOURCE_ID,
        "MUSIC_SOURCE_ID": frame.MUSIC_SOURCE_ID,
        "VOICE_GENERATOR": np.where(frame.VOICE_FAKE.eq(1), "ASVspoof2019-LA", "bonafide"),
        "MUSIC_GENERATOR": frame.MUSIC_GENERATOR,
        "SNR_DB": frame.snr_db,
        "ORIGINAL_SPLIT": split,
        "SOURCE_FILE": frame.syn_file,
    })
    targets = [
        f"MixFake/mixed_dataset/{split}/audio/{item}.wav" for item in truth.ID
    ]
    output_dir.mkdir(parents=True, exist_ok=True)
    truth.to_csv(output_dir / "truth.csv", index=False)
    (output_dir / "archive_targets.txt").write_text(
        "\n".join(targets) + "\n", encoding="utf-8"
    )
    summary = {
        "split": split,
        "samples": len(truth),
        "cells": truth.groupby(["VOICE_FAKE", "MUSIC_FAKE"]).size().to_dict(),
        "music_generators": truth.MUSIC_GENERATOR.value_counts().to_dict(),
        "excluded_protected_rows": int(overlap.sum()),
        "license": "CC-BY-4.0",
        "source": "https://huggingface.co/datasets/Tnxts/MixFake",
    }
    serializable = {
        **summary,
        "cells": {f"{voice}{music}": value for (voice, music), value in summary["cells"].items()},
    }
    (output_dir / "summary.json").write_text(
        json.dumps(serializable, indent=2), encoding="utf-8"
    )
    return serializable


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--details", type=Path,
        default=ROOT / "data/external/mixfake/MixFake/protocols/mixed_details.csv",
    )
    parser.add_argument("--train-output", type=Path,
                        default=ROOT / "data/eval/mixfake_music_train_v1")
    parser.add_argument("--dev-output", type=Path,
                        default=ROOT / "data/eval/mixfake_music_dev_v1")
    parser.add_argument("--train-per-cell", type=int, default=2_000)
    parser.add_argument("--dev-per-cell", type=int, default=400)
    parser.add_argument("--seed", type=int, default=20260902)
    args = parser.parse_args()
    blocked = protected_tokens(ROOT / "configs/data_partitions.yaml")
    details = pd.read_csv(args.details)
    train = build_bank(
        details, "train", args.train_per_cell, args.train_output,
        blocked, args.seed,
    )
    # The official dev split is disjoint from official train; also prevent an
    # accidental identity collision with the newly selected train manifest.
    blocked = blocked | identity_tokens(pd.read_csv(args.train_output / "truth.csv", dtype=str))
    dev = build_bank(
        details, "dev", args.dev_per_cell, args.dev_output,
        blocked, args.seed + 1,
    )
    print(json.dumps({"train": train, "dev": dev}, indent=2))


if __name__ == "__main__":
    main()
