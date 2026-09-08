#!/usr/bin/env python3
"""Create a fixed uniform-logit ensemble of three-stream predictions."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


AUTHENTICITY = (
    "FILE_FAKE_PROB", "VOICE_FAKE_PROB", "MUSIC_FAKE_PROB",
)
PRESENCE = ("VOICE_PRESENT_PROB", "MUSIC_PRESENT_PROB")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def logit(values: np.ndarray) -> np.ndarray:
    values = np.clip(np.asarray(values, np.float64), 1e-6, 1 - 1e-6)
    return np.log(values) - np.log1p(-values)


def sigmoid(values: np.ndarray) -> np.ndarray:
    return np.exp(-np.logaddexp(0.0, -np.asarray(values, np.float64)))


def ensemble(paths: list[Path]) -> pd.DataFrame:
    if len(paths) < 2:
        raise ValueError("uniform ensemble requires at least two members")
    frames = [pd.read_csv(path, dtype={"ID": str}) for path in paths]
    required = {"DATASET", "ID", *AUTHENTICITY, *PRESENCE}
    for path, frame in zip(paths, frames):
        if missing := required.difference(frame):
            raise ValueError(f"{path} misses columns: {sorted(missing)}")
        if frame[["DATASET", "ID"]].duplicated().any():
            raise ValueError(f"{path} has duplicate DATASET/ID keys")
    first = frames[0].sort_values(["DATASET", "ID"]).reset_index(drop=True)
    aligned = [first]
    for path, frame in zip(paths[1:], frames[1:]):
        frame = frame.sort_values(["DATASET", "ID"]).reset_index(drop=True)
        if not frame[["DATASET", "ID"]].equals(first[["DATASET", "ID"]]):
            raise ValueError(f"prediction IDs differ: {path}")
        for column in PRESENCE:
            if not np.array_equal(
                frame[column].to_numpy(), first[column].to_numpy()
            ):
                raise ValueError(f"presence changed across members: {column}")
        aligned.append(frame)
    output = first.copy()
    for column in AUTHENTICITY:
        member_logits = np.stack([
            logit(frame[column].to_numpy(np.float64)) for frame in aligned
        ])
        output[column] = sigmoid(member_logits.mean(axis=0))
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prediction", type=Path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite {args.output_dir}")
    for path in args.prediction:
        if not path.is_file():
            raise FileNotFoundError(path)
    result = ensemble(args.prediction)
    args.output_dir.mkdir(parents=True)
    output = args.output_dir / "predictions.csv"
    result.to_csv(output, index=False)
    provenance = {
        "method": "uniform_logit_mean",
        "members": [
            {"path": str(path.resolve()), "sha256": sha256_file(path)}
            for path in args.prediction
        ],
        "rows": len(result),
        "prediction_sha256": sha256_file(output),
    }
    (args.output_dir / "provenance.json").write_text(
        json.dumps(provenance, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Wrote {len(result):,} uniform-logit ensemble rows to {output}")


if __name__ == "__main__":
    main()
