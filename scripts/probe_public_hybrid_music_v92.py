#!/usr/bin/env python3
"""Retrospective mechanism probe for the public Suno/GTZAN hybrid model."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
BANK = ROOT / "data/eval/all_type_blind_v91"
ANCHOR_RUN = ROOT / "reports/file_global_moe_v91/prospective_v91"
CHECKPOINT = ROOT / "models/ai-music-deepfake-detector/best_model.pth"


def sha(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def worker(output: Path, shard: int) -> None:
    destination = output / f"shard_{shard}"
    destination.mkdir(parents=True, exist_ok=False)
    import numpy as np
    import pandas as pd
    import torch
    from src.hybrid_suno_music_detector_v92 import HybridSunoDetector, lme_probability
    from src.pipeline import load_audio

    torch.set_num_threads(2)
    detector = HybridSunoDetector(CHECKPOINT)
    neutral = pd.read_csv(BANK / "sample_submission.csv", dtype={"ID": str})
    selected = neutral.iloc[shard::4]
    path = destination / "predictions.csv"
    started = time.monotonic()
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["ID", "WINDOWS", "FIRST", "MEAN", "LME5", "MAX"])
        writer.writeheader()
        for index, identity in enumerate(selected.ID):
            scores = detector.window_probabilities(load_audio(BANK / "audio" / f"{identity}.flac"))
            writer.writerow({"ID": identity, "WINDOWS": len(scores), "FIRST": scores[0],
                             "MEAN": scores.mean(), "LME5": lme_probability(scores, 5),
                             "MAX": scores.max()})
            if index % 25 == 0:
                print(json.dumps({"shard": shard, "files": index + 1,
                                  "seconds": time.monotonic() - started}), flush=True)
    report = {"status": "complete", "shard": shard, "rows": len(selected),
              "seconds": time.monotonic() - started, "prediction_sha256": sha(path),
              "checkpoint_sha256": sha(CHECKPOINT)}
    (destination / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)


def score(output: Path) -> None:
    import numpy as np
    import pandas as pd
    from src.evaluate_diagnostic import official_eer
    parts = []
    for shard in range(4):
        directory = output / f"shard_{shard}"
        report = json.loads((directory / "report.json").read_text())
        path = directory / "predictions.csv"
        if (report["status"] != "complete" or report["prediction_sha256"] != sha(path)
                or report["checkpoint_sha256"] != sha(CHECKPOINT)):
            raise ValueError("invalid worker")
        parts.append(pd.read_csv(path, dtype={"ID": str}))
    predictions = pd.concat(parts, ignore_index=True)
    truth = pd.read_csv(BANK / "truth.csv", dtype={"ID": str})
    anchors = []
    for shard in range(4):
        anchors.append(pd.read_csv(ANCHOR_RUN / f"anchor_{shard}/output/submission.csv",
                                   dtype={"ID": str}))
    anchor = pd.concat(anchors, ignore_index=True)
    frame = truth.merge(predictions, on="ID", validate="one_to_one").merge(
        anchor[["ID", "FILE_FAKE_PROB", "MUSIC_FAKE_PROB"]], on="ID", validate="one_to_one")
    models = ["MUSIC_FAKE_PROB", "FIRST", "MEAN", "LME5", "MAX"]
    rows = []
    axes = {"ALL": None, "CHANNEL": "CHANNEL", "AUDIO_TYPE": "AUDIO_TYPE",
            "MIX_MODE": "MIX_MODE", "MUSIC_GENERATOR": "MUSIC_GENERATOR"}
    music = frame.loc[frame.MUSIC_PRESENT.eq(1)].copy()
    for axis, column in axes.items():
        groups = [("ALL", music)] if column is None else music.groupby(column, sort=True)
        for group, block in groups:
            labels = block.MUSIC_FAKE.to_numpy(int)
            if np.unique(labels).size != 2:
                continue
            for model in models:
                rows.append({"AXIS": axis, "GROUP": str(group), "MODEL": model, "N": len(block),
                             "REAL_N": int((labels == 0).sum()), "FAKE_N": int((labels == 1).sum()),
                             "MUSIC_EER": official_eer(labels, block[model].to_numpy(float))})
    metrics = pd.DataFrame(rows)
    metrics.to_csv(output / "metrics.csv", index=False)
    overall = metrics.loc[metrics.AXIS.eq("ALL")].to_dict("records")
    report = {"status": "complete_retrospective_mechanism_probe", "rows": len(frame),
              "results": overall, "selection_allowed": False,
              "automatic_submission_allowed": False,
              "checkpoint_source": "Huzaifanasir95/AI-Music-DeepFake-Detector (MIT)",
              "caveat": "v91 labels were already opened; results may reject or motivate a new design only."}
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["worker", "score"])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--shard", type=int, choices=range(4))
    args = parser.parse_args()
    if args.mode == "worker":
        if args.shard is None:
            raise ValueError("--shard required")
        worker(args.output, args.shard)
    else:
        score(args.output)


if __name__ == "__main__":
    main()
