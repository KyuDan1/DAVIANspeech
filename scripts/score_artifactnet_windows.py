#!/usr/bin/env python3
"""Cache ArtifactNet window scores for pooling/generalization ablations."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
from tqdm import tqdm


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from artifactnet_detector import ArtifactNetMusicDetector  # noqa: E402


def logmeanexp(probabilities: np.ndarray, temperature: float) -> float:
    scaled = temperature * np.clip(probabilities, 0, 1)
    peak = float(scaled.max())
    return float((peak + np.log(np.mean(np.exp(scaled - peak)))) / temperature)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", required=True)
    parser.add_argument(
        "--model-dir", type=Path, default=ROOT / "models/artifactnet"
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--provider", default="CUDAExecutionProvider")
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    args = parser.parse_args()
    if not 0 <= args.shard_index < args.num_shards:
        parser.error("shard-index must satisfy 0 <= index < num-shards")

    detector = ArtifactNetMusicDetector(
        args.model_dir,
        providers=[args.provider, "CPUExecutionProvider"],
    )
    rows: list[dict] = []
    for dataset in args.datasets:
        root = ROOT / "data" / "eval" / dataset
        truth_path = root / "truth.csv"
        if not truth_path.is_file():
            raise FileNotFoundError(truth_path)
        truth = pd.read_csv(truth_path, dtype={"ID": str})
        selected = truth.iloc[args.shard_index::args.num_shards]
        paths = {
            path.stem: path for path in (root / "audio").iterdir()
            if path.is_file()
        }
        for row in tqdm(
            selected.itertuples(index=False), total=len(selected), desc=dataset,
        ):
            path = paths.get(row.ID)
            if path is None:
                raise FileNotFoundError(f"{dataset}: missing audio for {row.ID}")
            audio, _ = librosa.load(
                path, sr=16_000, mono=True, dtype=np.float32
            )
            scores = detector.window_probabilities(audio)
            top_k = min(3, len(scores))
            rows.append({
                "DATASET": dataset,
                "ID": row.ID,
                "WINDOW_COUNT": len(scores),
                "MEDIAN": float(np.median(scores)),
                "MEAN": float(scores.mean()),
                "MAX": float(scores.max()),
                "TOP3": float(np.sort(scores)[-top_k:].mean()),
                "LME_T2": logmeanexp(scores, 2),
                "LME_T5": logmeanexp(scores, 5),
                "Q75": float(np.quantile(scores, .75)),
                "WINDOW_SCORES": json.dumps(scores.tolist(), separators=(",", ":")),
            })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.output, index=False)
    print(f"Saved {len(rows)} rows to {args.output}")


if __name__ == "__main__":
    main()
