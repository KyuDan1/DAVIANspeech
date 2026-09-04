#!/usr/bin/env python3
"""Select a seed-robust unified expert and audit a tiny residual over v18.

The official leaderboard rejects aggressive routing/weighting after v18.  This
script therefore uses development banks only to choose a seed subset, limits
both File and Music residual weights to five percent, and opens the three audit
banks only after the subset and weights have been fixed.
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from evaluate_diagnostic import score_frame  # noqa: E402
from evaluate_wpt_v38_residual import (  # noqa: E402
    grouped_score, logit, reconstruct_dev_anchors, reconstruct_v18_locked,
    sigmoid,
)
from train_unified_dual_ssl_head import truth_for  # noqa: E402


DEVELOPMENT = (
    "mixfake_music_dev_v1",
    "external_mixed_v1",
    "source_disjoint_mixed_v1",
    "source_disjoint_mixed_equal_v1",
    "source_disjoint_music_v1",
    "factorial_eval_1200_v2_dev",
    "telephone_mixed_dev_v1",
)
AUDITS = {
    "factorial": "factorial_eval_1200_v2_holdout",
    "phone": "phone_factorial_1200_v1",
    "yue": "yue_cross_component_audit_v1",
}
TASK_COLUMNS = (
    "VOICE_FAKE_PROB", "MUSIC_FAKE_PROB", "FILE_FAKE_PROB",
)


def load_development(paths: list[Path]) -> tuple[pd.DataFrame, list[pd.DataFrame]]:
    members = [pd.read_csv(path, dtype={"ID": str}) for path in paths]
    reference = members[0][["DATASET", "ID"]]
    for path, member in zip(paths[1:], members[1:]):
        if not reference.equals(member[["DATASET", "ID"]]):
            raise ValueError(f"development order differs in {path}")
    return reference, members


def ensemble(members: list[pd.DataFrame], subset: tuple[int, ...]) -> pd.DataFrame:
    result = members[0][["DATASET", "ID"]].copy()
    for column in TASK_COLUMNS:
        values = np.stack([
            logit(members[index][column].to_numpy(np.float64))
            for index in subset
        ])
        result[column] = sigmoid(values.mean(axis=0))
    return result


def development_metrics(
    expert: pd.DataFrame, subset: tuple[int, ...]
) -> list[dict[str, object]]:
    rows = []
    for dataset in DEVELOPMENT:
        block = expert.loc[expert.DATASET.eq(dataset)].set_index("ID")
        truth = truth_for(dataset, "dev").set_index("ID").loc[block.index]
        metric = score_frame(truth.join(block[list(TASK_COLUMNS)]))
        rows.append({"SUBSET": "+".join(map(str, subset)), "DATASET": dataset,
                     **metric})
    return rows


def locked_member_frames(path: Path, count: int) -> list[pd.DataFrame]:
    frame = pd.read_csv(path, dtype={"ID": str})
    members = []
    for index in range(count):
        current = frame[["DATASET", "ID"]].copy()
        for task in ("VOICE", "MUSIC", "FILE"):
            current[f"{task}_FAKE_PROB"] = frame[
                f"MEMBER_{index:02d}_{task}_FAKE_PROB"
            ]
        members.append(current)
    return members


def fuse_v18(
    anchor: pd.DataFrame, expert: pd.DataFrame,
    file_weight: float, music_weight: float,
) -> pd.DataFrame:
    result = anchor.copy()
    result["FILE_FAKE_PROB"] = sigmoid(
        (1 - file_weight) * logit(anchor.FILE_FAKE_PROB)
        + file_weight * logit(expert.FILE_FAKE_PROB)
    )
    result["MUSIC_FAKE_PROB"] = sigmoid(
        (1 - music_weight) * logit(anchor.MUSIC_FAKE_PROB)
        + music_weight * logit(expert.MUSIC_FAKE_PROB)
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--member-dev", type=Path, nargs="+", required=True)
    parser.add_argument("--locked-members", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--weights", type=float, nargs="+", default=[0, .025, .05],
    )
    args = parser.parse_args()
    if len(args.member_dev) < 2:
        parser.error("at least two members are required")
    if any(value < 0 or value > .05 for value in args.weights):
        parser.error("v43 residual weights must lie in [0, 0.05]")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    _, development_members = load_development(args.member_dev)
    subsets = [
        subset for size in range(1, len(development_members) + 1)
        for subset in itertools.combinations(range(len(development_members)), size)
    ]
    dev_rows = []
    experts = {}
    for subset in subsets:
        current = ensemble(development_members, subset)
        experts[subset] = current
        dev_rows.extend(development_metrics(current, subset))
    dev = pd.DataFrame(dev_rows)
    dev.to_csv(args.output_dir / "subset_development_metrics.csv", index=False)
    summary = dev.groupby("SUBSET", sort=False).agg(
        MEAN_ADS=("ADS", "mean"), WORST_ADS=("ADS", "min"),
        MEAN_MUSIC_EER=("MUSIC_EER", "mean"),
        WORST_MUSIC_EER=("MUSIC_EER", "max"),
    ).reset_index()
    summary["MUSIC_SELECTION"] = 1 - .5 * (
        summary.MEAN_MUSIC_EER + summary.WORST_MUSIC_EER
    )
    summary["MEMBERS"] = summary.SUBSET.str.count(r"\+") + 1
    summary = summary.sort_values(
        ["MUSIC_SELECTION", "MEAN_ADS", "MEMBERS"],
        ascending=[False, False, False],
    )
    summary.to_csv(args.output_dir / "subset_summary.csv", index=False)
    selected_subset = tuple(map(int, summary.iloc[0].SUBSET.split("+")))

    _truth, v18, _v37 = reconstruct_dev_anchors()
    truth, _, _ = reconstruct_dev_anchors()
    selected_dev = experts[selected_subset]
    expert_dev = selected_dev.loc[
        selected_dev.DATASET.eq("factorial_eval_1200_v2_dev")
    ].set_index("ID").loc[v18.index]
    baseline, baseline_worst, baseline_groups = grouped_score(truth, v18)
    weight_rows = []
    for file_weight, music_weight in itertools.product(args.weights, repeat=2):
        candidate = fuse_v18(v18, expert_dev, file_weight, music_weight)
        metric, worst, groups = grouped_score(truth, candidate)
        group_delta = min(
            item["ADS"] - next(
                base["ADS"] for base in baseline_groups
                if base["CHANNEL"] == item["CHANNEL"]
            )
            for item in groups
        )
        weight_rows.append({
            "FILE_WEIGHT": file_weight, "MUSIC_WEIGHT": music_weight,
            "ADS": metric["ADS"], "WORST_CHANNEL_ADS": worst,
            "MIN_CHANNEL_DELTA": group_delta,
            "ROBUST_SELECTION": .5 * metric["ADS"] + .5 * worst,
        })
    weight_sweep = pd.DataFrame(weight_rows)
    weight_sweep.to_csv(args.output_dir / "v18_tiny_weight_sweep.csv", index=False)
    safe = weight_sweep.loc[weight_sweep.MIN_CHANNEL_DELTA >= -1e-12]
    pool = safe if len(safe) else weight_sweep
    selected_weight = pool.sort_values(
        ["ROBUST_SELECTION", "ADS", "FILE_WEIGHT", "MUSIC_WEIGHT"],
        ascending=[False, False, True, True],
    ).iloc[0]

    locked_members = locked_member_frames(
        args.locked_members, len(development_members)
    )
    locked_expert = ensemble(locked_members, selected_subset)
    audit_rows = []
    for short, dataset in AUDITS.items():
        audit_truth, anchor = reconstruct_v18_locked(short)
        expert = locked_expert.loc[
            locked_expert.DATASET.eq(dataset)
        ].set_index("ID").loc[anchor.index]
        for method, file_weight, music_weight in (
            ("v18", 0., 0.),
            ("selected_tiny_residual", float(selected_weight.FILE_WEIGHT),
             float(selected_weight.MUSIC_WEIGHT)),
        ):
            prediction = fuse_v18(anchor, expert, file_weight, music_weight)
            audit_rows.append({
                "DATASET": short, "METHOD": method,
                "FILE_WEIGHT": file_weight, "MUSIC_WEIGHT": music_weight,
                **score_frame(audit_truth.join(prediction)),
            })
    audit = pd.DataFrame(audit_rows)
    audit.to_csv(args.output_dir / "locked_audit.csv", index=False)
    metadata = {
        "selected_subset": list(selected_subset),
        "selected_file_weight": float(selected_weight.FILE_WEIGHT),
        "selected_music_weight": float(selected_weight.MUSIC_WEIGHT),
        "development_anchor_ads": float(baseline["ADS"]),
        "development_anchor_worst_channel_ads": float(baseline_worst),
    }
    (args.output_dir / "selection.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    print(summary.head(10).round(6).to_string(index=False))
    print("\n", json.dumps(metadata, indent=2))
    print("\n", audit.round(6).to_string(index=False))


if __name__ == "__main__":
    main()
