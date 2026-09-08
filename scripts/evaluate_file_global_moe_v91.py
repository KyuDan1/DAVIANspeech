#!/usr/bin/env python3
"""One-shot prospective evaluation of the frozen v91 File-only MoE.

The candidate formula and acceptance thresholds live in the already frozen
v91 manifest.  ``prepare`` and both worker modes only consume the neutral
sample manifest; labels are opened for the first and only time by ``score``.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

BANK = ROOT / "data/eval/all_type_blind_v91"
VALIDATION = ROOT / "reports/all_type_blind_v91_design/bank_validation/validation.json"
CANDIDATE = ROOT / "reports/file_global_moe_v91/frozen_candidate.json"
PACKAGE = ROOT / "reports/music_only_submission_v82/full_package_smoke"
V86 = ROOT / "reports/file_local_readout_v86/full"
OUTPUT_COLUMNS = [
    "FILE_FAKE_PROB", "VOICE_FAKE_PROB", "MUSIC_FAKE_PROB",
    "VOICE_PRESENT_PROB", "MUSIC_PRESENT_PROB",
]


def sha(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_candidate() -> dict:
    frozen = json.loads(CANDIDATE.read_text())
    changed = {
        name: {"expected": expected, "actual": sha(ROOT / name)}
        for name, expected in frozen["artifacts_sha256"].items()
        if sha(ROOT / name) != expected
    }
    if changed:
        raise ValueError(f"frozen candidate artifacts changed: {changed}")
    if frozen["fusion"] != {"domain": "logit", "global_expert_weight": 0.1, "routing": "none"}:
        raise ValueError("unexpected frozen v91 formula")
    return frozen


def prepare(output: Path) -> None:
    if output.exists():
        raise FileExistsError(output)
    candidate = verify_candidate()
    validation = json.loads(VALIDATION.read_text())
    if (validation.get("status") != "PASS"
            or validation.get("checks_passed") != validation.get("checks_total")
            or validation.get("authenticity_detector_scores_read") is not False
            or validation.get("score_open_count") != 0):
        raise ValueError("prospective bank has not passed its frozen pre-score validation")
    truth = BANK / "truth.csv"
    if sha(truth) != validation["truth_sha256"]:
        raise ValueError("bank truth changed after validation")

    # This neutral manifest contains IDs and constant placeholders only.
    import pandas as pd
    neutral = pd.read_csv(BANK / "sample_submission.csv", dtype={"ID": str})
    if len(neutral) != validation["rendered_rows"] or neutral.ID.duplicated().any():
        raise ValueError("invalid neutral sample manifest")
    if not all(neutral[column].eq(0.5).all() for column in OUTPUT_COLUMNS):
        raise ValueError("sample manifest is not neutral")

    script = PACKAGE / "script_anchor.py"
    output.mkdir(parents=True)
    runners = []
    for shard in range(4):
        runner = output / f"anchor_{shard}"
        (runner / "data/test").mkdir(parents=True)
        (runner / "output").mkdir()
        os.symlink(PACKAGE.joinpath("model").resolve(), runner / "model")
        selected = neutral.iloc[shard::4]
        for identity in selected.ID:
            source = BANK / "audio" / f"{identity}.flac"
            os.symlink(source.resolve(), runner / "data/test" / source.name)
        selected.to_csv(runner / "data/sample_submission.csv", index=False)
        runners.append({
            "shard": shard,
            "root": str(runner.resolve()),
            "sample_sha256": sha(runner / "data/sample_submission.csv"),
        })

    artifacts = {
        str(Path(name)): expected
        for name, expected in candidate["artifacts_sha256"].items()
    }
    artifacts.update({
        str(VALIDATION.relative_to(ROOT)): sha(VALIDATION),
        str(BANK.joinpath("truth.csv").relative_to(ROOT)): sha(BANK / "truth.csv"),
        str(BANK.joinpath("sample_submission.csv").relative_to(ROOT)): sha(BANK / "sample_submission.csv"),
        str(PACKAGE.joinpath("script_anchor.py").relative_to(ROOT)): sha(script),
        str(Path(__file__).resolve().relative_to(ROOT)): sha(Path(__file__)),
    })
    frozen = {
        "schema": "file_global_moe_v91_one_shot",
        "status": "frozen_before_inference_or_scoring",
        "rows": len(neutral),
        "shards": 4,
        "runners": runners,
        "artifacts_sha256": artifacts,
        "candidate_sha256": sha(CANDIDATE),
        "truth_open_policy": "workers use neutral IDs only; score opens labels once",
        "score_open_count": 0,
        "selection_allowed_after_score": False,
        "automatic_submission_allowed": False,
    }
    (output / "frozen.json").write_text(json.dumps(frozen, indent=2) + "\n")
    print(json.dumps({"status": "prepared", "rows": len(neutral), "shards": 4}), flush=True)


def load_frozen(output: Path) -> tuple[Path, dict]:
    verify_candidate()
    path = output / "frozen.json"
    frozen = json.loads(path.read_text())
    if frozen["score_open_count"] != 0:
        raise ValueError("prospective truth was already opened")
    for name, expected in frozen["artifacts_sha256"].items():
        if sha(ROOT / name) != expected:
            raise ValueError(f"frozen artifact changed: {name}")
    return path, frozen


def anchor_worker(output: Path, shard: int) -> None:
    frozen_path, frozen = load_frozen(output)
    record = frozen["runners"][shard]
    runner = Path(record["root"])
    if sha(runner / "data/sample_submission.csv") != record["sample_sha256"]:
        raise ValueError("runner sample changed")
    destination = runner / "output/submission.csv"
    if destination.exists():
        raise FileExistsError(destination)
    script = PACKAGE / "script_anchor.py"
    spec = importlib.util.spec_from_file_location(f"anchor_v91_{shard}", script)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    module.BASE_DIR = runner
    original_run = module.run

    def run_without_music_cache(runtime_args):
        runtime_args.xlsr_window_embeddings_output = None
        return original_run(runtime_args)

    module.run = run_without_music_cache
    module.apply_spectra_voice_fusion = lambda *args, **kwargs: None
    module.apply_sparse_call_consensus = lambda *args, **kwargs: None
    module.apply_three_stream_anchor_residual = lambda *args, **kwargs: None
    sys.argv = [str(script)]
    started = time.monotonic()
    module.main()
    elapsed = time.monotonic() - started
    prediction = __import__("pandas").read_csv(destination, dtype={"ID": str})
    expected_rows = (frozen["rows"] + 3 - shard) // 4
    if len(prediction) != expected_rows or prediction.ID.duplicated().any():
        raise ValueError("invalid anchor shard")
    report = {
        "status": "complete", "kind": "anchor", "shard": shard,
        "rows": len(prediction), "seconds": elapsed,
        "prediction_sha256": sha(destination), "frozen_sha256": sha(frozen_path),
    }
    (runner / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)


def expert_worker(output: Path, shard: int) -> None:
    frozen_path, frozen = load_frozen(output)
    destination = output / f"expert_{shard}"
    destination.mkdir(exist_ok=False)
    import torch
    from src.file_local_inference_v86 import FileLocalPredictor
    from src.pipeline import load_audio

    torch.set_num_threads(2)
    predictor = FileLocalPredictor(V86, ROOT)
    neutral = __import__("pandas").read_csv(BANK / "sample_submission.csv", dtype={"ID": str})
    selected = neutral.iloc[shard::frozen["shards"]]
    path = destination / "predictions.csv"
    started = time.monotonic()
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["ID", "global"])
        writer.writeheader()
        for index, identity in enumerate(selected.ID):
            audio_path = BANK / "audio" / f"{identity}.flac"
            result = predictor(load_audio(audio_path))
            writer.writerow({"ID": identity, "global": result["global"]})
            if index % 50 == 0:
                print(json.dumps({"kind": "expert", "shard": shard,
                                  "files": index + 1,
                                  "seconds": time.monotonic() - started}), flush=True)
    report = {
        "status": "complete", "kind": "expert", "shard": shard,
        "rows": len(selected), "seconds": time.monotonic() - started,
        "prediction_sha256": sha(path), "frozen_sha256": sha(frozen_path),
    }
    (destination / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)


def eer(labels, scores) -> float:
    import numpy as np
    from sklearn.metrics import roc_curve
    fpr, tpr, _ = roc_curve(labels, scores, pos_label=1, drop_intermediate=False)
    fnr = 1 - tpr
    index = int(np.argmin(np.abs(fpr - fnr)))
    return float((fpr[index] + fnr[index]) / 2)


def logit(values):
    import numpy as np
    values = np.clip(np.asarray(values, dtype=np.float64), 1e-6, 1 - 1e-6)
    return np.log(values / (1 - values))


def sigmoid(values):
    import numpy as np
    return 1 / (1 + np.exp(-np.asarray(values, dtype=np.float64)))


def score(output: Path) -> None:
    frozen_path, frozen = load_frozen(output)
    import numpy as np
    import pandas as pd
    anchor_parts, expert_parts = [], []
    worker_reports = []
    for shard in range(frozen["shards"]):
        runner = Path(frozen["runners"][shard]["root"])
        anchor_path = runner / "output/submission.csv"
        anchor_report = json.loads((runner / "report.json").read_text())
        expert_dir = output / f"expert_{shard}"
        expert_path = expert_dir / "predictions.csv"
        expert_report = json.loads((expert_dir / "report.json").read_text())
        for report, path, kind in ((anchor_report, anchor_path, "anchor"),
                                   (expert_report, expert_path, "expert")):
            if (report["status"] != "complete" or report["kind"] != kind
                    or report["prediction_sha256"] != sha(path)
                    or report["frozen_sha256"] != sha(frozen_path)):
                raise ValueError(f"invalid {kind} shard {shard}")
            worker_reports.append(report)
        anchor_parts.append(pd.read_csv(anchor_path, dtype={"ID": str}))
        expert_parts.append(pd.read_csv(expert_path, dtype={"ID": str}))
    anchor = pd.concat(anchor_parts, ignore_index=True)
    expert = pd.concat(expert_parts, ignore_index=True)
    if (len(anchor) != frozen["rows"] or len(expert) != frozen["rows"]
            or anchor.ID.duplicated().any() or expert.ID.duplicated().any()):
        raise ValueError("invalid combined predictions")

    # First label access in this workflow occurs here, after both predictions
    # are immutable.  The candidate formula remains the precommitted 10%.
    truth = pd.read_csv(BANK / "truth.csv", dtype={"ID": str})
    frame = truth.merge(anchor[["ID", "FILE_FAKE_PROB"]], on="ID", validate="one_to_one")
    frame = frame.merge(expert, on="ID", validate="one_to_one")
    frame["candidate"] = sigmoid(
        0.9 * logit(frame.FILE_FAKE_PROB) + 0.1 * logit(frame["global"])
    )

    rows = []
    axes = {"ALL": None, "CHANNEL": "CHANNEL", "AUDIO_TYPE": "AUDIO_TYPE",
            "MIX_MODE": "MIX_MODE", "COMPONENT_CASE": "COMPONENT_CASE"}
    for axis, column in axes.items():
        groups = [("ALL", frame)] if column is None else frame.groupby(column, sort=True)
        for group, block in groups:
            labels = block.FILE_FAKE.to_numpy(int)
            if np.unique(labels).size != 2:
                continue
            anchor_eer = eer(labels, block.FILE_FAKE_PROB)
            candidate_eer = eer(labels, block.candidate)
            rows.append({"AXIS": axis, "GROUP": str(group), "N": len(block),
                         "REAL_N": int((labels == 0).sum()), "FAKE_N": int((labels == 1).sum()),
                         "ANCHOR_FILE_EER": anchor_eer, "CANDIDATE_FILE_EER": candidate_eer,
                         "EER_DELTA": candidate_eer - anchor_eer})
    pair_specs = {"RR_vs_FR": ("RR", "FR"), "RR_vs_RF": ("RR", "RF")}
    for group, cases in pair_specs.items():
        block = frame.loc[frame.COMPONENT_CASE.isin(cases)]
        labels = block.FILE_FAKE.to_numpy(int)
        anchor_eer = eer(labels, block.FILE_FAKE_PROB)
        candidate_eer = eer(labels, block.candidate)
        rows.append({"AXIS": "PAIR", "GROUP": group, "N": len(block),
                     "REAL_N": int((labels == 0).sum()), "FAKE_N": int((labels == 1).sum()),
                     "ANCHOR_FILE_EER": anchor_eer, "CANDIDATE_FILE_EER": candidate_eer,
                     "EER_DELTA": candidate_eer - anchor_eer})
    metrics = pd.DataFrame(rows)
    overall = metrics.loc[(metrics.AXIS == "ALL") & (metrics.GROUP == "ALL")].iloc[0]
    channel_max = float(metrics.loc[metrics.AXIS == "CHANNEL", "EER_DELTA"].max())
    pair_max = float(metrics.loc[metrics.AXIS == "PAIR", "EER_DELTA"].max())
    gain = float(overall.ANCHOR_FILE_EER - overall.CANDIDATE_FILE_EER)
    gate = {"overall_file_eer_minimum_absolute_gain": 0.005,
            "maximum_channel_eer_regression": 0.01,
            "maximum_rr_vs_fr_or_rf_regression": 0.02}
    accepted = bool(gain >= gate["overall_file_eer_minimum_absolute_gain"]
                    and channel_max <= gate["maximum_channel_eer_regression"]
                    and pair_max <= gate["maximum_rr_vs_fr_or_rf_regression"])
    scored = output / "score"
    scored.mkdir(exist_ok=False)
    metrics.to_csv(scored / "metrics.csv", index=False)
    anchor.to_csv(scored / "anchor_predictions.csv", index=False)
    candidate_predictions = anchor.copy()
    candidate_predictions["FILE_FAKE_PROB"] = frame.set_index("ID").loc[candidate_predictions.ID, "candidate"].to_numpy()
    candidate_predictions.to_csv(scored / "candidate_predictions.csv", index=False)
    report = {
        "status": "complete_one_shot", "accepted_by_precommitted_statistical_gate": accepted,
        "packaging_promotion_allowed": accepted, "automatic_submission_allowed": False,
        "score_open_count": 1, "rows": len(frame), "gate": gate,
        "anchor_file_eer": float(overall.ANCHOR_FILE_EER),
        "candidate_file_eer": float(overall.CANDIDATE_FILE_EER),
        "absolute_file_eer_gain": gain,
        "maximum_channel_eer_regression": channel_max,
        "maximum_rr_vs_fr_or_rf_regression": pair_max,
        "prediction_hashes": {"anchor": sha(scored / "anchor_predictions.csv"),
                              "candidate": sha(scored / "candidate_predictions.csv")},
        "worker_seconds": worker_reports,
        "caveat": "Synthetic source-disjoint evidence; not an estimate of private leaderboard EER.",
    }
    (scored / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["prepare", "anchor-worker", "expert-worker", "score"])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--shard", type=int, choices=range(4))
    args = parser.parse_args()
    if args.mode == "prepare":
        prepare(args.output)
    elif args.mode == "anchor-worker":
        if args.shard is None:
            raise ValueError("--shard required")
        anchor_worker(args.output, args.shard)
    elif args.mode == "expert-worker":
        if args.shard is None:
            raise ValueError("--shard required")
        expert_worker(args.output, args.shard)
    else:
        score(args.output)


if __name__ == "__main__":
    main()
