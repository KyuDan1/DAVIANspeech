#!/usr/bin/env python3
"""Audit the fixed five-percent EAT patch-graph residual over exact v18."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from evaluate_diagnostic import score_frame  # noqa: E402
from evaluate_wpt_v38_residual import reconstruct_v18_locked  # noqa: E402


AUDITS = {
    "factorial": "factorial_eval_1200_v2_holdout",
    "phone": "phone_factorial_1200_v1",
    "yue": "yue_cross_component_audit_v1",
}
DEFAULT_PREDICTIONS = (
    ROOT / "reports/eat_patch_graph_v1/seed01_balanced_fm_locked"
    / "predictions.csv"
)


def logit(values: pd.Series | np.ndarray) -> np.ndarray:
    clipped = np.clip(np.asarray(values, dtype=np.float64), 1e-6, 1 - 1e-6)
    return np.log(clipped) - np.log1p(-clipped)


def sigmoid(values: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-np.clip(values, -60, 60)))


def fuse(anchor: pd.DataFrame, expert: pd.DataFrame, weight: float) -> pd.DataFrame:
    result = anchor.copy()
    for column in ("FILE_FAKE_PROB", "MUSIC_FAKE_PROB"):
        result[column] = sigmoid(
            (1 - weight) * logit(anchor[column]) + weight * logit(expert[column])
        )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, default=DEFAULT_PREDICTIONS)
    parser.add_argument(
        "--output-dir", type=Path,
        default=ROOT / "reports/eat_patch_graph_v1/v44_residual_audit",
    )
    parser.add_argument("--weight", type=float, default=.05)
    args = parser.parse_args()
    if not 0 <= args.weight <= .05:
        parser.error("the official-ablation guard limits the residual to 5%")

    expert_all = pd.read_csv(args.predictions, dtype={"ID": str})
    rows: list[dict[str, object]] = []
    for short, dataset in AUDITS.items():
        truth, anchor = reconstruct_v18_locked(short)
        expert = expert_all.loc[expert_all.DATASET.eq(dataset)].set_index("ID")
        missing = anchor.index.difference(expert.index)
        if len(missing):
            raise ValueError(f"{dataset} is missing {len(missing)} expert rows")
        expert = expert.loc[anchor.index]
        for method, prediction in (
            ("v18", anchor),
            ("eat_patch_graph_direct", expert),
            ("v44_patch_residual", fuse(anchor, expert, args.weight)),
        ):
            rows.append({
                "DATASET": short,
                "METHOD": method,
                "FILE_WEIGHT": 0. if method != "v44_patch_residual" else args.weight,
                "MUSIC_WEIGHT": 0. if method != "v44_patch_residual" else args.weight,
                **score_frame(truth.join(prediction)),
            })

    metrics = pd.DataFrame(rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(args.output_dir / "locked_audit.csv", index=False)

    suno = expert_all.loc[expert_all.DATASET.eq("suno_vocals_v1")]
    summary = {
        "predictions": str(args.predictions.relative_to(ROOT)),
        "residual_weight": args.weight,
        "locked_datasets": list(AUDITS.values()),
        "suno_count": int(len(suno)),
        "suno_file_above_half": int((suno.FILE_FAKE_PROB > .5).sum()),
        "suno_music_above_half": int((suno.MUSIC_FAKE_PROB > .5).sum()),
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(metrics.round(6).to_string(index=False))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
