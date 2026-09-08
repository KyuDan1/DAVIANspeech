#!/usr/bin/env python3
"""Select a fixed logit fusion between the exact anchor and one v94 expert."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve


def finite_eer(labels, scores) -> float:
    labels = np.asarray(labels, np.int64)
    scores = np.asarray(scores, np.float64)
    if len(np.unique(labels)) != 2:
        return float("nan")
    fpr, tpr, _ = roc_curve(
        labels, scores, pos_label=1, drop_intermediate=False
    )
    fnr = 1 - tpr
    index = int(np.argmin(np.abs(fpr - fnr)))
    return float((fpr[index] + fnr[index]) / 2)


def subgroup_eer(frame, scores, task):
    result = {}
    if task == "music":
        groups = {"voice_real": ("RR", "RF"), "voice_fake": ("FR", "FF")}
    elif task == "voice":
        groups = {"music_real": ("RR", "FR"), "music_fake": ("RF", "FF")}
    else:
        groups = {}
    for name, cases in groups.items():
        mask = frame.COMPONENT_CASE.isin(cases).to_numpy()
        result[name] = finite_eer(
            frame.loc[mask, f"{task.upper()}_FAKE"], scores[mask]
        )
    return result


def logit(values: np.ndarray) -> np.ndarray:
    values = np.clip(np.asarray(values, np.float64), 1e-5, 1 - 1e-5)
    return np.log(values) - np.log1p(-values)


def sigmoid(values: np.ndarray) -> np.ndarray:
    return np.exp(-np.logaddexp(0, -np.asarray(values, np.float64)))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--anchor", type=Path, required=True)
    parser.add_argument("--expert", type=Path, required=True)
    parser.add_argument("--channel", required=True)
    parser.add_argument("--task", choices=("voice", "music", "file"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=101)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if "locked" in " ".join(map(str, (args.manifest, args.anchor, args.expert, args.output))).lower():
        raise ValueError("locked data are forbidden during fusion selection")
    truth = pd.read_csv(args.manifest, dtype={"ID": str})
    # Component EER is defined only where that component is present.  Broad
    # inventory manifests also contain absent-component rows, whereas the
    # component evaluators intentionally omit them from their score archives.
    if args.task in {"voice", "music"}:
        present_column = f"{args.task.upper()}_PRESENT"
        truth = truth.loc[
            pd.to_numeric(truth[present_column], errors="coerce").eq(1)
        ].reset_index(drop=True)
    anchor = pd.read_csv(args.anchor, dtype={"ID": str})
    if not set(truth.ID).issubset(set(anchor.ID)) or anchor.ID.duplicated().any():
        raise ValueError("anchor does not cover all truth IDs")
    anchor = anchor.set_index("ID").loc[truth.ID].reset_index()
    with np.load(args.expert, allow_pickle=False) as archive:
        expert_ids = archive["ids"].astype(str)
        if args.channel not in archive:
            raise ValueError(f"expert archive lacks channel {args.channel}")
        expert_scores = archive[args.channel].astype(np.float64)
        if expert_scores.ndim == 2:
            task_index = {"voice": 0, "music": 1, "file": 2}[args.task]
            if expert_scores.shape[1] != 3:
                raise ValueError("joint expert scores must have three task columns")
            expert_scores = expert_scores[:, task_index]
    if len(set(expert_ids)) != len(expert_ids) or not set(truth.ID).issubset(
        set(expert_ids)
    ):
        raise ValueError("expert does not cover all truth IDs")
    expert_by_id = dict(zip(expert_ids, expert_scores))
    expert_scores = np.asarray([expert_by_id[item] for item in truth.ID])
    label = truth[f"{args.task.upper()}_FAKE"].to_numpy(np.int64)
    anchor_scores = anchor[f"{args.task.upper()}_FAKE_PROB"].to_numpy(np.float64)
    anchor_logit, expert_logit = logit(anchor_scores), logit(expert_scores)
    rows = []
    for weight in np.linspace(0, 1, args.steps):
        fused = sigmoid((1 - weight) * anchor_logit + weight * expert_logit)
        rows.append({"expert_weight": float(weight), "eer": finite_eer(label, fused)})
    sweep = pd.DataFrame(rows)
    best_eer = float(sweep.eer.min())
    # Select the smallest weight on the best plateau; it alters the proven
    # anchor least when several weights induce the same EER ranking.
    best_weight = float(sweep.loc[sweep.eer.eq(best_eer), "expert_weight"].min())
    fused = sigmoid((1 - best_weight) * anchor_logit + best_weight * expert_logit)
    report = {
        "task": args.task, "channel": args.channel, "rows": len(truth),
        "anchor_eer": finite_eer(label, anchor_scores),
        "expert_eer": finite_eer(label, expert_scores),
        "best_fused_eer": best_eer, "selected_expert_weight": best_weight,
        "anchor_subgroup_eer": subgroup_eer(truth, anchor_scores, args.task),
        "expert_subgroup_eer": subgroup_eer(truth, expert_scores, args.task),
        "fused_subgroup_eer": subgroup_eer(truth, fused, args.task),
        "locked_scores_read": False,
    }
    args.output.mkdir(parents=True)
    sweep.to_csv(args.output / "sweep.csv", index=False)
    pd.DataFrame({"ID": truth.ID, "anchor": anchor_scores,
                  "expert": expert_scores, "fused": fused,
                  "label": label}).to_csv(args.output / "predictions.csv", index=False)
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
