#!/usr/bin/env python3
"""Score fixed start/middle/end views of ArtifactBench with ArtifactNet.

The diagnostic deliberately limits each long track to three four-second views,
matching the bounded evidence available in competition inference rather than
letting very long songs contribute dozens of windows.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from artifactnet_detector import ArtifactNetMusicDetector  # noqa: E402
from pipeline import load_audio  # noqa: E402
from wpt_spectra_inference import fixed_windows  # noqa: E402


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--views", type=int, default=3)
    parser.add_argument("--max-per-source", type=int, default=None)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    frame = pd.read_csv(args.manifest, dtype={"ID": str})
    if args.max_per_source is not None:
        frame = frame.groupby("SOURCE", sort=True, group_keys=False).head(
            args.max_per_source
        ).reset_index(drop=True)
    detector = ArtifactNetMusicDetector(
        args.model_dir, providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
    )
    rows = []
    for index, row in enumerate(frame.itertuples(index=False)):
        audio = load_audio(Path(row.PATH))
        windows = fixed_windows(audio, 4 * 16_000, args.views)
        scores = detector.window_probabilities(windows.reshape(-1), source_sr=16_000)
        rows.append({
            "ID": row.ID, "SOURCE": row.SOURCE,
            "score": float(np.median(scores)),
            "score_mean": float(np.mean(scores)),
            "score_max": float(np.max(scores)),
        })
        if index == 0 or (index + 1) % 100 == 0 or index + 1 == len(frame):
            print(json.dumps({"done": index + 1, "total": len(frame)}), flush=True)
    result = pd.DataFrame(rows)
    source_report = {}
    for source, group in result.groupby("SOURCE"):
        score = group.score.to_numpy()
        source_report[source] = {
            "rows": len(group), "mean": float(score.mean()),
            "p10": float(np.quantile(score, .1)),
            "fraction_above_0_5": float(np.mean(score > .5)),
        }
    args.output.mkdir(parents=True)
    result.to_csv(args.output / "predictions.csv", index=False)
    report = {
        "rows": len(result), "views": args.views,
        "manifest_sha256": sha256(args.manifest),
        "model_sha256": sha256(args.model_dir / "artifactnet_v94_full.onnx"),
        "all_fake_mean": float(result.score.mean()),
        "all_fake_fraction_above_0_5": float(np.mean(result.score > .5)),
        "by_source": source_report,
    }
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
