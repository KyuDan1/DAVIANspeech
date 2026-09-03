#!/usr/bin/env python3
"""Apply frozen SPEAR binary/joint probe heads to cached embeddings."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

LAYERS = 13
DIM = 1280


def sigmoid(value: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(value, -40, 40)))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--embedding-dir", type=Path, required=True)
    parser.add_argument("--probe-head", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--layer", type=int, default=2)
    parser.add_argument("--existing-voice-head", type=Path)
    parser.add_argument("--existing-music-head", type=Path)
    args = parser.parse_args()
    if not 0 <= args.layer < LAYERS:
        parser.error(f"--layer must be between 0 and {LAYERS - 1}")

    ids, vectors = [], []
    for path in sorted(args.embedding_dir.glob("shard_*.npz")):
        shard = np.load(path)
        ids.extend(shard["ids"].astype(str))
        vectors.append(shard["embeddings"])
    if not vectors:
        raise FileNotFoundError(f"No embedding shards in {args.embedding_dir}")
    matrix = np.concatenate(vectors).reshape(-1, LAYERS, DIM)
    head = np.load(args.probe_head)
    normalized = (matrix - head["mean"]) / head["std"]
    layer = args.layer
    binary_logits = (
        normalized[:, layer] @ head["binary_weight"][layer]
        + head["binary_bias"][layer]
    )
    joint_logits = (
        normalized[:, layer] @ head["joint_weight"][layer]
        + head["joint_bias"][layer]
    )
    joint_logits -= joint_logits.max(axis=1, keepdims=True)
    joint_probabilities = np.exp(joint_logits)
    joint_probabilities /= joint_probabilities.sum(axis=1, keepdims=True)
    frame = pd.DataFrame({
        "ID": ids,
        "SPEAR_BINARY_VOICE_PROB": sigmoid(binary_logits[:, 0]),
        "SPEAR_BINARY_MUSIC_PROB": sigmoid(binary_logits[:, 1]),
        "SPEAR_JOINT_VOICE_PROB": (
            joint_probabilities[:, 2] + joint_probabilities[:, 3]
        ),
        "SPEAR_JOINT_MUSIC_PROB": (
            joint_probabilities[:, 1] + joint_probabilities[:, 3]
        ),
        "SPEAR_JOINT_FILE_PROB": 1.0 - joint_probabilities[:, 0],
    })
    for name, path in (
        ("SPEAR_EXISTING_VOICE_PROB", args.existing_voice_head),
        ("SPEAR_EXISTING_MUSIC_PROB", args.existing_music_head),
    ):
        if path:
            existing = np.load(path)
            frame[name] = sigmoid(
                matrix.reshape(len(matrix), -1) @ existing["weight"]
                + existing["bias"]
            )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    frame.sort_values("ID").to_csv(args.output, index=False)
    print(f"Saved {len(frame)} scores to {args.output}")


if __name__ == "__main__":
    main()
