#!/usr/bin/env python3
"""Extract XLSR-SLS scores across multiple leakage-controlled evaluation banks."""

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

from xlsr_sls_detector import (  # noqa: E402
    SAMPLE_RATE, XlsrSLSDetector, logmeanexp_probability,
)


AUDIO_EXTENSIONS = {".aac", ".flac", ".m4a", ".mp3", ".ogg", ".opus", ".wav"}


def audio_index(dataset_dir: Path, audio_dir: Path | None = None) -> dict[str, Path]:
    audio_root = audio_dir if audio_dir is not None else dataset_dir / "audio"
    paths = [
        path for path in audio_root.rglob("*")
        if path.is_file() and path.suffix.lower() in AUDIO_EXTENSIONS
    ]
    result = {path.stem: path for path in paths}
    if len(result) != len(paths):
        raise ValueError(f"Duplicate audio stems in {dataset_dir}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, nargs="+", required=True)
    parser.add_argument(
        "--model", type=Path,
        default=ROOT / "models" / "xlsr-sls" / "xlsr-sls.onnx",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--audio-dir", type=Path,
        help="Optional audio override; valid only when exactly one dataset is given.",
    )
    parser.add_argument(
        "--split", action="append", default=[], metavar="COLUMN=VALUE",
        help="Filter each truth manifest before inference (repeatable).",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()

    if args.audio_dir is not None and len(args.dataset) != 1:
        parser.error("--audio-dir requires exactly one --dataset")
    filters = []
    for expression in args.split:
        if "=" not in expression:
            parser.error(f"invalid --split {expression!r}; expected COLUMN=VALUE")
        filters.append(expression.split("=", 1))

    detector = XlsrSLSDetector(args.model, args.device, args.batch_size)
    rows = []
    for dataset_dir in args.dataset:
        truth = pd.read_csv(dataset_dir / "truth.csv", dtype={"ID": str})
        for column, value in filters:
            if column not in truth.columns:
                raise KeyError(f"{dataset_dir}: missing split column {column!r}")
            truth = truth[truth[column].astype(str) == value].copy()
        if truth.empty:
            raise ValueError(f"{dataset_dir}: split filters selected no rows")
        paths = audio_index(dataset_dir, args.audio_dir)
        missing = sorted(set(truth.ID) - set(paths))
        if missing:
            raise FileNotFoundError(f"{dataset_dir}: missing audio for {missing[:5]}")
        for sample_id in tqdm(truth.ID, desc=f"XLSR-SLS {dataset_dir.name}"):
            audio, _ = librosa.load(
                paths[sample_id], sr=SAMPLE_RATE, mono=True, dtype=np.float32,
            )
            scores = detector.window_probabilities(audio)
            rows.append({
                "DATASET": dataset_dir.name,
                "ID": sample_id,
                "N_WINDOWS": len(scores),
                "SLS_FIRST": float(scores[0]),
                "SLS_MEAN": float(scores.mean()),
                "SLS_MAX": float(scores.max()),
                "SLS_LME5": logmeanexp_probability(scores, 5.0),
                "SLS_WINDOWS": json.dumps(scores.tolist(), separators=(",", ":")),
            })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.output, index=False)
    print(f"Saved {len(rows)} scores to {args.output}")


if __name__ == "__main__":
    main()
