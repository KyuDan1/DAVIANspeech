#!/usr/bin/env python3
"""Export trained long-horizon heads into a deterministic NumPy checkpoint."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from train_long_horizon_music_probe import RandomSslProjection  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--head", type=Path, nargs="+", required=True)
    parser.add_argument("--member-weights", type=float, nargs="+", required=True)
    parser.add_argument(
        "--phone-member-weights", type=float, nargs="+", required=True,
    )
    parser.add_argument("--music-weight", type=float, default=0.40)
    parser.add_argument("--file-weight", type=float, default=0.20)
    parser.add_argument("--file-music-presence-threshold", type=float, default=0.70)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not (len(args.head) == len(args.member_weights)
            == len(args.phone_member_weights)):
        parser.error("head and both member-weight lists must have equal lengths")
    for values in (args.member_weights, args.phone_member_weights):
        if any(value < 0 for value in values) or not np.isclose(sum(values), 1):
            parser.error("member weights must be non-negative and sum to one")
    for value in (
        args.music_weight, args.file_weight, args.file_music_presence_threshold,
    ):
        if not 0 <= value <= 1:
            parser.error("fusion weights and threshold must be in [0, 1]")

    checkpoints = [
        torch.load(path, map_location="cpu", weights_only=False)
        for path in args.head
    ]
    first = checkpoints[0]
    signature = (
        first.get("projection_kind"), first.get("projection_width"),
        first.get("projection_seed"), first.get("feature_mode"),
        tuple(first.get("layers", [])),
    )
    if signature[0] != "random" or signature[3] != "both":
        raise ValueError("Deployment requires random-projection, mean+dispersion heads")
    if signature[4] != tuple(range(13)):
        raise ValueError("Deployment requires all 13 SPEAR layers")
    for checkpoint in checkpoints[1:]:
        other = (
            checkpoint.get("projection_kind"), checkpoint.get("projection_width"),
            checkpoint.get("projection_seed"), checkpoint.get("feature_mode"),
            tuple(checkpoint.get("layers", [])),
        )
        if other != signature:
            raise ValueError("All heads must share the same frozen projection")

    projection = RandomSslProjection(
        int(signature[1]), int(signature[2]), torch.device("cpu")
    )
    payload: dict[str, np.ndarray] = {
        "format_version": np.asarray(1, dtype=np.int64),
        "eat_matrix": projection.eat_matrix.cpu().numpy(),
        "spear_matrix": projection.spear_matrix.cpu().numpy(),
        "member_count": np.asarray(len(checkpoints), dtype=np.int64),
        "member_logit_weights": np.asarray(args.member_weights, dtype=np.float64),
        "phone_member_logit_weights": np.asarray(
            args.phone_member_weights, dtype=np.float64
        ),
        "music_weight": np.asarray(args.music_weight, dtype=np.float64),
        "file_weight": np.asarray(args.file_weight, dtype=np.float64),
        "file_music_presence_threshold": np.asarray(
            args.file_music_presence_threshold, dtype=np.float64
        ),
    }
    for index, checkpoint in enumerate(checkpoints):
        classifier = checkpoint["classifier"]
        payload[f"member_{index}_mean"] = np.asarray(
            checkpoint["mean"], dtype=np.float32
        )
        payload[f"member_{index}_std"] = np.asarray(
            checkpoint["std"], dtype=np.float32
        )
        payload[f"member_{index}_weight"] = (
            classifier["weight"].detach().cpu().numpy().reshape(-1).astype(np.float32)
        )
        payload[f"member_{index}_bias"] = np.asarray(
            classifier["bias"].detach().cpu().numpy().reshape(()), dtype=np.float32
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.output, **payload)
    print(f"Exported {len(checkpoints)} heads to {args.output}")


if __name__ == "__main__":
    main()
