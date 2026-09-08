#!/usr/bin/env python3
"""One-shot evaluation of the frozen v101 candidate on the reserved v93 lockbox.

The architecture, weights and checkpoints are constants in this file.  The
lockbox may validate the frozen candidate once; it must never select a weight.
"""
from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "reports/mixfake_alltype_v93_design/reservation/locked.csv"
AUDIO_ROOT = ROOT / "data/external/mixfake_alltype_v93"
PACKAGE = ROOT / "reports/music_only_submission_v82/full_package_smoke"
JOINT = ROOT / (
    "reports/mixfake_multistream_v97/seed34_joint_robust/"
    "multistream_prompt.pt"
)
MUSIC = ROOT / (
    "reports/mixfake_multistream_v96/seed35_music_exactcodec_paperwarm/"
    "multistream_prompt.pt"
)
OUTPUT = ROOT / "reports/mixfake_joint_music_moe_v101/locked_one_shot"
EXPECTED_MANIFEST_SHA256 = (
    "d7409df0ae6e36936eb79ab8c8d6377d14f4ef7433f60c75ed0b4a0cc0a0a8d3"
)
WEIGHTS = {"file": .40, "voice": .12, "music": .28}
FILE_OR_WEIGHT = .15
COLUMNS = (
    "ID", "FILE_FAKE_PROB", "VOICE_FAKE_PROB", "MUSIC_FAKE_PROB",
    "VOICE_PRESENT_PROB", "MUSIC_PRESENT_PROB",
)


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def eer(labels, scores):
    labels = np.asarray(labels, np.int64)
    scores = np.asarray(scores, np.float64)
    fpr, tpr, _ = roc_curve(
        labels, scores, pos_label=1, drop_intermediate=False
    )
    fnr = 1 - tpr
    index = int(np.argmin(np.abs(fpr - fnr)))
    return float((fpr[index] + fnr[index]) / 2)


def metrics(truth, prediction):
    joined = truth.merge(prediction, on="ID", validate="one_to_one")
    result = {}
    for task in ("file", "voice", "music"):
        key = task.upper()
        mask = np.ones(len(joined), dtype=bool)
        if task != "file":
            mask = joined[f"{key}_PRESENT"].eq(1).to_numpy()
        result[f"{task}_eer"] = eer(
            joined.loc[mask, f"{key}_FAKE"],
            joined.loc[mask, f"{key}_FAKE_PROB"],
        )
    result["ads"] = (
        .5 * (1 - result["file_eer"])
        + .2 * (1 - result["voice_eer"])
        + .3 * (1 - result["music_eer"])
    )
    result["mean_scores_by_case"] = {
        case: {
            task: float(block[f"{task.upper()}_FAKE_PROB"].mean())
            for task in ("file", "voice", "music")
        }
        for case, block in joined.groupby("COMPONENT_CASE")
    }
    return result


def main():
    if OUTPUT.exists():
        raise FileExistsError(
            f"one-shot lockbox output already exists; refusing rerun: {OUTPUT}"
        )
    if sha256_file(MANIFEST) != EXPECTED_MANIFEST_SHA256:
        raise ValueError("locked manifest changed")
    for path in (PACKAGE / "script.py", PACKAGE / "model", JOINT, MUSIC):
        if not path.exists():
            raise FileNotFoundError(path)
    truth = pd.read_csv(MANIFEST, dtype={"ID": str})
    if len(truth) != 480 or truth.ID.duplicated().any():
        raise ValueError("unexpected locked manifest")
    OUTPUT.mkdir(parents=True)
    runner = OUTPUT / "runner"
    test_dir = runner / "data/test"
    test_dir.mkdir(parents=True)
    (runner / "output").mkdir()
    os.symlink((PACKAGE / "model").resolve(), runner / "model")
    for row in truth.itertuples(index=False):
        source = (AUDIO_ROOT / str(row.ARCHIVE_MEMBER)).resolve()
        if not source.is_file():
            raise FileNotFoundError(source)
        os.symlink(source, test_dir / f"{row.ID}.wav")
    sample = runner / "data/sample_submission.csv"
    with sample.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS)
        writer.writeheader()
        for identity in truth.ID:
            writer.writerow({"ID": identity, **{name: .5 for name in COLUMNS[1:]}})
    protocol = {
        "schema": "mixfake_v101_locked_one_shot",
        "purpose": "validation_only_no_selection",
        "rows": len(truth),
        "manifest_sha256": sha256_file(MANIFEST),
        "base_script_sha256": sha256_file(PACKAGE / "script.py"),
        "joint_checkpoint_sha256": sha256_file(JOINT),
        "music_checkpoint_sha256": sha256_file(MUSIC),
        "weights": WEIGHTS,
        "file_or_weight": FILE_OR_WEIGHT,
        "views": {"joint": 3, "music": 1},
        "future_weight_tuning_from_this_result_forbidden": True,
    }
    (OUTPUT / "frozen_protocol.json").write_text(
        json.dumps(protocol, indent=2) + "\n"
    )

    frozen_script = PACKAGE / "script.py"
    spec = importlib.util.spec_from_file_location("frozen_v83_for_v101", frozen_script)
    if spec is None or spec.loader is None:
        raise ImportError(frozen_script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.BASE_DIR = runner
    sys.argv = [str(frozen_script), "--device", "cuda:3"]
    started = time.monotonic()
    module.main()
    submission = runner / "output/submission.csv"
    anchor = pd.read_csv(submission, dtype={"ID": str})
    anchor.to_csv(OUTPUT / "anchor_predictions.csv", index=False)

    sys.path.insert(0, str(ROOT / "src"))
    from mixfake_multistream_file_inference_v99 import (  # noqa: E402
        apply_mixfake_joint_music_moe,
    )
    apply_mixfake_joint_music_moe(
        test_dir, submission, PACKAGE / "model/spectra-aasist", JOINT, MUSIC,
        device="cuda:3", file_weight=WEIGHTS["file"],
        voice_weight=WEIGHTS["voice"], music_weight=WEIGHTS["music"],
        batch_size=12, joint_views=3, music_views=1,
        file_or_weight=FILE_OR_WEIGHT,
    )
    candidate = pd.read_csv(submission, dtype={"ID": str})
    candidate.to_csv(OUTPUT / "candidate_predictions.csv", index=False)
    report = {
        "status": "complete_locked_one_shot",
        "seconds": time.monotonic() - started,
        "anchor": metrics(truth, anchor),
        "candidate": metrics(truth, candidate),
        "locked_used_for_selection": False,
        "rerun_allowed": False,
    }
    (OUTPUT / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
