#!/usr/bin/env python3
"""Compare the exact current File anchor with frozen v86 heads on v70."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def logit(value):
    value = np.clip(np.asarray(value, dtype=np.float64), 1e-6, 1 - 1e-6)
    return np.log(value / (1 - value))


def sigmoid(value):
    return 1 / (1 + np.exp(-np.asarray(value, dtype=np.float64)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--v86", type=Path, required=True)
    args = parser.parse_args()
    frozen_path = args.run / "frozen.json"
    frozen = json.loads(frozen_path.read_text())
    if sha(frozen["truth"]) != frozen["truth_sha256"]:
        raise ValueError("truth changed")
    parts = []
    for record in frozen["runners"]:
        runner = Path(record["root"])
        report = json.loads((runner / "report.json").read_text())
        output = runner / "output/submission.csv"
        if (report["status"] != "complete" or report["frozen_sha256"] != sha(frozen_path)
                or report["predictions_sha256"] != sha(output)):
            raise ValueError("invalid current-anchor shard")
        parts.append(pd.read_csv(output, dtype={"ID": str})[["ID", "FILE_FAKE_PROB"]])
    anchor = pd.concat(parts, ignore_index=True).rename(columns={"FILE_FAKE_PROB": "current_anchor"})
    if len(anchor) != frozen["rows"] or anchor.ID.duplicated().any():
        raise ValueError("invalid current-anchor identities")

    v86_parts = []
    for shard in range(4):
        directory = args.v86 / f"shard_{shard}"
        report = json.loads((directory / "report.json").read_text())
        path = directory / "predictions.csv"
        if report["status"] != "complete" or report["predictions_sha256"] != sha(path):
            raise ValueError("invalid v86 shard")
        v86_parts.append(pd.read_csv(path, dtype={"ID": str}))
    v86 = pd.concat(v86_parts, ignore_index=True)
    truth = pd.read_csv(frozen["truth"], dtype={"ID": str})
    frame = truth.merge(anchor, on="ID", validate="one_to_one").merge(v86, on="ID", validate="one_to_one")
    if len(frame) != len(truth):
        raise ValueError("identity mismatch")

    for expert in ["global", "local"]:
        for weight in [.025, .05, .10, .20]:
            frame[f"anchor_{expert}_{weight:g}"] = sigmoid(
                (1 - weight) * logit(frame.current_anchor) + weight * logit(frame[expert]))
    models = [name for name in frame.columns if name in {"current_anchor", "parent", "global", "local"}
              or name.startswith("anchor_")]

    from src.evaluate_diagnostic import official_eer
    results = []

    def add(scope, axis, group, selected):
        labels = selected.FILE_FAKE.to_numpy(int)
        if len(np.unique(labels)) != 2:
            return
        for model in models:
            results.append(dict(scope=scope, axis=axis, group=group, model=model,
                                n=len(selected), real_n=int((labels == 0).sum()),
                                fake_n=int((labels == 1).sum()),
                                FILE_EER=official_eer(labels, selected[model].to_numpy(float))))

    add("all", "ALL", "ALL", frame)
    for axis in ["DURATION", "CHANNEL", "AUDIO_TYPE", "MIX_MODE"]:
        for group, selected in frame.groupby(axis):
            add("all", axis, str(group), selected)
    pair_cases = {
        "RR_vs_FR": ("RR", "FR"),
        "RR_vs_RF": ("RR", "RF"),
        "RR_vs_FF": ("RR", "FF"),
        "VR_vs_VF": ("VR", "VF"),
        "MR_vs_MF": ("MR", "MF"),
    }
    for pair, cases in pair_cases.items():
        selected = frame.loc[frame.COMPONENT_CASE.isin(cases)]
        add("pair", "ALL", pair, selected)
        for axis in ["DURATION", "CHANNEL", "MIX_MODE"]:
            for group, block in selected.groupby(axis):
                add("pair", axis, f"{pair}|{group}", block)
    metrics = pd.DataFrame(results)
    metrics.to_csv(args.run / "metrics.csv", index=False)
    overall = metrics.loc[(metrics.scope == "all") & (metrics.axis == "ALL")].to_dict("records")
    report = dict(status="complete_exposed_anchor_comparison", rows=len(frame), results=overall,
                  selection_allowed=False, automatic_submission_allowed=False,
                  interpretation="Fixed blends diagnose complementarity only; v70 cannot select a submission weight.",
                  input_sha256={"anchor_frozen": sha(frozen_path),
                                "v86_frozen": sha(args.v86 / "frozen.json")})
    (args.run / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
