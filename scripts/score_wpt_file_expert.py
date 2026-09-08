#!/usr/bin/env python3
"""Export five-view WPT task probabilities for an ordered audio bank."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pipeline import find_audio_files, order_by_submission  # noqa: E402
from wpt_spectra_inference import predict_wpt_tasks  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audio-dir", type=Path, required=True)
    parser.add_argument("--sample-submission", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=6)
    parser.add_argument("--file-views", type=int, default=5)
    parser.add_argument("--file-temperature", type=float, default=2.0)
    args = parser.parse_args()

    with args.sample_submission.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    paths = order_by_submission(find_audio_files(args.audio_dir), rows)
    probabilities = predict_wpt_tasks(
        paths, args.model_dir, args.checkpoint,
        device=args.device, file_batch_size=args.batch_size,
        file_views=args.file_views,
        file_temperature=args.file_temperature,
    )
    if probabilities.shape != (len(rows), 3):
        raise RuntimeError("unexpected WPT prediction shape")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(".tmp.csv")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "ID", "VOICE_FAKE_PROB", "MUSIC_FAKE_PROB",
                "FILE_FAKE_PROB",
            ],
        )
        writer.writeheader()
        for row, values in zip(rows, np.asarray(probabilities)):
            writer.writerow({
                "ID": row["ID"],
                "VOICE_FAKE_PROB": float(values[0]),
                "MUSIC_FAKE_PROB": float(values[1]),
                "FILE_FAKE_PROB": float(values[2]),
            })
    temporary.replace(args.output)


if __name__ == "__main__":
    main()
