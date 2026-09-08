#!/usr/bin/env python3
"""Evaluate the frozen v86 File readouts on the exposed v70 component bank.

This is a mechanism audit only.  The v70 sources were used by earlier work, so
the result may reject a candidate but must not, by itself, authorize a submit.
"""
import argparse
import csv
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.check_file_local_v86 import sha


RUN = ROOT / "reports/file_local_readout_v86/full"
BANK = ROOT / "data/eval/long_component_stress_v70"
EAT = ROOT / "reports/long_component_stress_v70_comparison"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["prepare", "worker", "score"])
    parser.add_argument("--shard", type=int, choices=range(4))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if args.mode == "prepare":
        if args.output.exists():
            raise FileExistsError(args.output)
        truth = BANK / "truth.csv"
        validation = BANK / "validation.json"
        valid = json.loads(validation.read_text())
        if not valid["passed"] or valid["selection_allowed"]:
            raise ValueError("v70 must remain a stress-only bank")
        if sha(truth) != valid["truth_sha256"]:
            raise ValueError("v70 truth changed")
        completed = json.loads((RUN / "report.json").read_text())
        if completed["status"] != "complete":
            raise ValueError("v86 training is incomplete")
        artifacts = [
            Path(__file__), truth, validation, RUN / "frozen.json",
            RUN / "report.json", RUN / "parent_development.csv",
            RUN / "global.pt", RUN / "local.pt",
            ROOT / "src/file_local_inference_v86.py",
            ROOT / "src/file_local_readout_v86.py",
            ROOT / "src/common_encoder_probe.py",
            ROOT / "src/xlsr_antideepfake.py",
            EAT / "eat_control/predictions.csv",
            EAT / "eat_adapted/predictions.csv",
        ]
        truth_frame = __import__("pandas").read_csv(truth, dtype={"ID": str})
        audio = []
        for identity in truth_frame.ID:
            path = BANK / "audio" / f"{identity}.flac"
            audio.append(dict(ID=identity, PATH=str(path), SHA256=sha(path)))
        args.output.mkdir(parents=True)
        frozen = dict(
            schema="file_local_component_long_v90",
            rows=len(audio), shards=4, inputs=audio,
            artifacts_sha256={str(p): sha(p) for p in artifacts},
            scope="PREVIOUSLY EXPOSED synthetic component stress; rejection audit only",
            selection_allowed=False, automatic_submission_allowed=False,
        )
        (args.output / "frozen.json").write_text(json.dumps(frozen, indent=2) + "\n")
        print(json.dumps(dict(status="prepared", rows=len(audio))), flush=True)
        return

    frozen_path = args.output / "frozen.json"
    frozen = json.loads(frozen_path.read_text())
    if any(sha(p) != digest for p, digest in frozen["artifacts_sha256"].items()):
        raise ValueError("frozen artifact changed")

    if args.mode == "worker":
        if args.shard is None:
            raise ValueError("--shard required")
        import torch
        from src.file_local_inference_v86 import FileLocalPredictor
        from src.pipeline import load_audio

        torch.set_num_threads(2)
        destination = args.output / f"shard_{args.shard}"
        destination.mkdir(exist_ok=False)
        predictor = FileLocalPredictor(RUN, ROOT)
        selected = frozen["inputs"][args.shard::frozen["shards"]]
        started = time.monotonic()
        prediction_path = destination / "predictions.csv"
        with prediction_path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["ID", "parent", "global", "local"])
            writer.writeheader()
            for index, row in enumerate(selected):
                path = Path(row["PATH"])
                if sha(path) != row["SHA256"]:
                    raise ValueError("audio changed")
                writer.writerow(dict(ID=row["ID"], **predictor(load_audio(path))))
                if index % 100 == 0:
                    print(json.dumps(dict(shard=args.shard, files=index + 1,
                                          seconds=time.monotonic() - started)), flush=True)
        report = dict(status="complete", rows=len(selected), seconds=time.monotonic() - started,
                      predictions_sha256=sha(prediction_path), frozen_sha256=sha(frozen_path))
        (destination / "report.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report), flush=True)
        return

    import numpy as np
    import pandas as pd
    from src.evaluate_diagnostic import official_eer

    parts = []
    for shard in range(frozen["shards"]):
        directory = args.output / f"shard_{shard}"
        report = json.loads((directory / "report.json").read_text())
        if (report["status"] != "complete"
                or report["predictions_sha256"] != sha(directory / "predictions.csv")
                or report["frozen_sha256"] != sha(frozen_path)):
            raise ValueError("invalid shard")
        parts.append(pd.read_csv(directory / "predictions.csv", dtype={"ID": str}))
    predictions = pd.concat(parts, ignore_index=True)
    if len(predictions) != frozen["rows"] or predictions.ID.duplicated().any():
        raise ValueError("invalid prediction identities")
    truth = pd.read_csv(BANK / "truth.csv", dtype={"ID": str})
    frame = truth.merge(predictions, on="ID", validate="one_to_one")
    for name in ["eat_control", "eat_adapted"]:
        source = pd.read_csv(EAT / name / "predictions.csv", dtype={"ID": str})
        frame = frame.merge(source[["ID", "FILE_FAKE_PROB"]].rename(
            columns={"FILE_FAKE_PROB": name}), on="ID", validate="one_to_one")
    models = ["parent", "global", "local", "eat_control", "eat_adapted"]
    rows = []
    axes = [[], ["DURATION"], ["CHANNEL"], ["AUDIO_TYPE"], ["MIX_MODE"],
            ["COMPONENT_CASE"], ["DURATION", "CHANNEL"],
            ["DURATION", "AUDIO_TYPE"], ["MIX_MODE", "COMPONENT_CASE"]]
    for columns in axes:
        groups = [("ALL", frame)] if not columns else frame.groupby(columns)
        for keys, selected in groups:
            if columns and not isinstance(keys, tuple):
                keys = (keys,)
            group = "ALL" if not columns else "|".join(map(str, keys))
            axis = "ALL" if not columns else "|".join(columns)
            for model in models:
                labels = selected.FILE_FAKE.to_numpy()
                scores = selected[model].to_numpy(float)
                if len(np.unique(labels)) < 2:
                    continue
                rows.append(dict(axis=axis, group=group, model=model, n=len(selected),
                                 real_n=int((labels == 0).sum()), fake_n=int((labels == 1).sum()),
                                 FILE_EER=official_eer(labels, scores)))
    metrics = pd.DataFrame(rows)
    metrics.to_csv(args.output / "metrics.csv", index=False)
    overall = metrics.loc[metrics.axis.eq("ALL")].to_dict("records")
    report = dict(status="complete_exposed_component_audit", rows=len(frame), results=overall,
                  selection_allowed=False, automatic_submission_allowed=False,
                  caveat="Development-derived synthetic sources; reject failures, never claim natural-call OOD confirmation.")
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
