#!/usr/bin/env python3
"""Robustness audit for replacing or extending the v101 Music expert."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve


def eer(label, score) -> float:
    label = np.asarray(label, np.int64)
    score = np.asarray(score, np.float64)
    if len(np.unique(label)) != 2:
        return float("nan")
    fpr, tpr, _ = roc_curve(label, score, pos_label=1, drop_intermediate=False)
    fnr = 1 - tpr
    index = int(np.argmin(np.abs(fpr - fnr)))
    return float((fpr[index] + fnr[index]) / 2)


def logit(value):
    value = np.clip(np.asarray(value, np.float64), 1e-5, 1 - 1e-5)
    return np.log(value) - np.log1p(-value)


def sigmoid(value):
    return np.exp(-np.logaddexp(0, -np.asarray(value, np.float64)))


def fuse(first, second, weight):
    return sigmoid((1 - weight) * logit(first) + weight * logit(second))


def best_weight(label, anchor, expert):
    rows = [(weight, eer(label, fuse(anchor, expert, weight)))
            for weight in np.linspace(0, 1, 101)]
    best = min(value for _, value in rows)
    return min(weight for weight, value in rows if value == best), best


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--v101-input", type=Path, required=True)
    parser.add_argument("--expert", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--replace-weight", type=float, default=.50)
    parser.add_argument("--extend-weight", type=float, default=.34)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    truth = pd.read_csv(args.manifest, dtype={"ID": str}, low_memory=False)
    truth = truth.loc[truth.MUSIC_PRESENT.eq(1)].reset_index(drop=True)
    old = pd.read_csv(args.v101_input, dtype={"ID": str}).set_index("ID").loc[truth.ID]
    with np.load(args.expert, allow_pickle=False) as archive:
        by_id = dict(zip(archive["ids"].astype(str), archive["clean"]))
    new = np.asarray([by_id[item] for item in truth.ID], np.float64)
    v83 = old.anchor.to_numpy(np.float64)
    seed35 = old.expert.to_numpy(np.float64)
    v101 = fuse(v83, seed35, .28)
    replace = fuse(v83, new, args.replace_weight)
    extend = fuse(v101, new, args.extend_weight)
    label = truth.MUSIC_FAKE.to_numpy(np.int64)
    variants = {
        "v83": v83, "v101": v101, "new_expert": new,
        "replace": replace, "extend": extend,
    }
    groups = []
    for column in ("DATASET", "CHANNEL", "MIX_MODE", "COMPONENT_CASE"):
        if column not in truth:
            continue
        values = truth[column].fillna("missing").astype(str)
        for value in sorted(values.unique()):
            mask = values.eq(value).to_numpy()
            if len(np.unique(label[mask])) != 2:
                continue
            row = {"column": column, "value": value, "rows": int(mask.sum())}
            row.update({name: eer(label[mask], score[mask])
                        for name, score in variants.items()})
            groups.append(row)
    group_frame = pd.DataFrame(groups)
    datasets = truth.DATASET.fillna("missing").astype(str)
    lodo = []
    for heldout in sorted(datasets.unique()):
        train = datasets.ne(heldout).to_numpy()
        test = ~train
        if len(np.unique(label[test])) != 2:
            continue
        for anchor_name, anchor in (("v83", v83), ("v101", v101)):
            weight, train_eer = best_weight(label[train], anchor[train], new[train])
            lodo.append({
                "heldout": heldout, "anchor": anchor_name, "weight": weight,
                "train_eer": train_eer,
                "heldout_anchor_eer": eer(label[test], anchor[test]),
                "heldout_fused_eer": eer(label[test], fuse(anchor[test], new[test], weight)),
            })
    report = {
        "rows": len(truth),
        "global_eer": {name: eer(label, score) for name, score in variants.items()},
        "new_score_correlation": {
            "v83": float(np.corrcoef(new, v83)[0, 1]),
            "seed35": float(np.corrcoef(new, seed35)[0, 1]),
        },
        "replace_weight_and_eer": best_weight(label, v83, new),
        "extend_weight_and_eer": best_weight(label, v101, new),
        "lodo_improved": {
            anchor: int(sum(
                row["heldout_fused_eer"] < row["heldout_anchor_eer"]
                for row in lodo if row["anchor"] == anchor
            ))
            for anchor in ("v83", "v101")
        },
        "lodo_comparable": {
            anchor: int(sum(row["anchor"] == anchor for row in lodo))
            for anchor in ("v83", "v101")
        },
    }
    args.output.mkdir(parents=True)
    group_frame.to_csv(args.output / "group_eer.csv", index=False)
    pd.DataFrame(lodo).to_csv(args.output / "leave_one_dataset_out.csv", index=False)
    pd.DataFrame({"ID": truth.ID, "label": label, **variants}).to_csv(
        args.output / "predictions.csv", index=False
    )
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
