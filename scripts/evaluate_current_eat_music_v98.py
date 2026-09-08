#!/usr/bin/env python3
"""Score the frozen v83 EAT Music branch on a registered development manifest."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import librosa
import numpy as np
import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from src.common_eat_adaptation_inference import CommonEatAdaptationPredictor  # noqa: E402
from evaluate_mixfake_multistream_prompt_v94 import (  # noqa: E402
    attach_audio_paths, finite_eer, subgroup_eer,
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--audio-root", type=Path, required=True)
    parser.add_argument("--flat-audio-dir", type=Path, default=None)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--score-name", default="clean")
    parser.add_argument("--checkpoint", type=Path, default=(
        ROOT / "reports/common_eat_adaptation/full/adapted/head.pt"
    ))
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if "locked" in " ".join(map(str, (
        args.manifest, args.flat_audio_dir, args.output,
    ))).lower():
        raise ValueError("locked data are forbidden during candidate selection")
    frame = pd.read_csv(args.manifest, dtype={"ID": str})
    required = {"ID", "COMPONENT_CASE", "MUSIC_FAKE"}
    if missing := required.difference(frame.columns):
        raise ValueError(f"manifest misses {sorted(missing)}")
    if frame.ID.duplicated().any():
        raise ValueError("manifest IDs are not unique")
    frame = attach_audio_paths(frame, args.audio_root, args.flat_audio_dir)
    model = CommonEatAdaptationPredictor(
        args.checkpoint, ROOT, device=args.device
    )
    scores = []
    started = time.monotonic()
    for index, row in enumerate(frame.itertuples(index=False)):
        audio, _ = librosa.load(
            Path(row.PATH), sr=16000, mono=True, dtype=np.float32
        )
        scores.append(float(model(audio)[2]))
        if index == 0 or (index + 1) % 100 == 0 or index + 1 == len(frame):
            print(json.dumps({
                "done": index + 1, "total": len(frame),
                "seconds": time.monotonic() - started,
            }), flush=True)
    scores = np.asarray(scores, np.float32)
    labels = frame.MUSIC_FAKE.to_numpy(np.int64)
    report = {
        "status": "complete", "rows": len(frame), "score_name": args.score_name,
        "music_eer": finite_eer(labels, scores),
        "subgroup_eer": subgroup_eer(frame, scores, "music"),
        "manifest": str(args.manifest),
        "manifest_sha256": sha256_file(args.manifest),
        "checkpoint_sha256": sha256_file(args.checkpoint),
        "locked_scores_read": False,
        "seconds": time.monotonic() - started,
    }
    args.output.mkdir(parents=True)
    np.savez_compressed(
        args.output / "predictions.npz", ids=frame.ID.to_numpy(dtype=str),
        **{args.score_name: scores},
    )
    pd.DataFrame({"ID": frame.ID, "MUSIC_FAKE_PROB": scores}).to_csv(
        args.output / "predictions.csv", index=False
    )
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
