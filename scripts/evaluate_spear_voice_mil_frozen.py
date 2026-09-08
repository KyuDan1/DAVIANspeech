#!/usr/bin/env python3
"""One-shot retrospective scoring for an already frozen Voice MIL residual."""

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
sys.path.insert(0, str(ROOT / "scripts"))

from score_prospective_paired_eval import (  # noqa: E402
    OUTPUT_FILES, score_prospective_paired_eval,
)
from spear_voice_mil_inference import (  # noqa: E402
    _logit, _sigmoid, predict_spear_voice_mil,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--truth", type=Path, required=True)
    parser.add_argument("--anchor", type=Path, required=True)
    parser.add_argument("--statistics", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, action="append", required=True)
    parser.add_argument("--voice-weight", type=float, default=.15)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to repeat frozen diagnostic: {args.output_dir}")
    if not 0 <= args.voice_weight <= 1:
        parser.error("voice weight must lie in [0,1]")
    truth = pd.read_csv(args.truth, dtype={"ID": str})
    anchor = pd.read_csv(args.anchor, dtype={"ID": str})
    ids, expert = predict_spear_voice_mil(
        args.statistics, args.checkpoint, device=args.device,
    )
    if set(anchor.ID) != set(ids):
        raise ValueError("anchor and frozen Voice expert IDs differ")
    lookup = dict(zip(ids, expert))
    candidate = anchor.copy()
    evidence = np.asarray([lookup[item] for item in candidate.ID], dtype=np.float64)
    candidate["VOICE_FAKE_PROB"] = _sigmoid(
        (1 - args.voice_weight) * _logit(candidate.VOICE_FAKE_PROB)
        + args.voice_weight * _logit(evidence)
    )
    reports = {}
    for model, prediction in (("anchor", anchor), ("voice_mil_v53", candidate)):
        for key, frame in score_prospective_paired_eval(truth, prediction).items():
            block = frame.copy(); block.insert(0, "MODEL", model)
            reports.setdefault(key, []).append(block)
    args.output_dir.mkdir(parents=True)
    candidate.to_csv(args.output_dir / "predictions.csv", index=False)
    for key, filename in OUTPUT_FILES.items():
        pd.concat(reports[key], ignore_index=True).to_csv(
            args.output_dir / filename, index=False,
        )
    (args.output_dir / "provenance.json").write_text(json.dumps({
        "frozen_before_diagnostic": True,
        "selection_or_sweep_supported": False,
        "voice_weight": args.voice_weight,
        "truth": {"path": str(args.truth), "sha256": sha256(args.truth)},
        "anchor": {"path": str(args.anchor), "sha256": sha256(args.anchor)},
        "statistics": {
            "path": str(args.statistics), "sha256": sha256(args.statistics),
        },
        "checkpoints": [
            {"path": str(path), "sha256": sha256(path)} for path in args.checkpoint
        ],
    }, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
