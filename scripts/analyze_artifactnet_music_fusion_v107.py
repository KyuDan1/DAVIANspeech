#!/usr/bin/env python3
"""Measure whether already-computed ArtifactNet evidence repairs final Music."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve


def eer(y, score):
    y, score = np.asarray(y, np.int64), np.asarray(score, np.float64)
    if len(np.unique(y)) != 2:
        return float("nan")
    fpr, tpr, _ = roc_curve(y, score, pos_label=1, drop_intermediate=False)
    fnr = 1 - tpr
    i = int(np.argmin(np.abs(fpr - fnr)))
    return float((fpr[i] + fnr[i]) / 2)


def logit(x):
    x = np.clip(np.asarray(x, np.float64), 1e-5, 1 - 1e-5)
    return np.log(x) - np.log1p(-x)


def fuse(a, b, weight):
    value = (1 - weight) * logit(a) + weight * logit(b)
    return np.exp(-np.logaddexp(0, -value))


def select(y, a, b):
    rows = [(w, eer(y, fuse(a, b, w))) for w in np.linspace(0, .5, 101)]
    best = min(value for _, value in rows)
    return min(w for w, value in rows if value == best), best


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--prompt-input", type=Path, required=True)
    parser.add_argument("--new-prompt", type=Path, required=True)
    parser.add_argument("--artifactnet", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    truth = pd.read_csv(args.manifest, dtype={"ID": str}, low_memory=False)
    truth = truth.loc[truth.MUSIC_PRESENT.eq(1)].reset_index(drop=True)
    prompt = pd.read_csv(args.prompt_input, dtype={"ID": str}).set_index("ID").loc[truth.ID]
    artifact = pd.read_csv(args.artifactnet, dtype={"ID": str}).set_index("ID").loc[truth.ID]
    with np.load(args.new_prompt, allow_pickle=False) as archive:
        new_by_id = dict(zip(archive["ids"].astype(str), archive["clean"]))
    new = np.asarray([new_by_id[item] for item in truth.ID])
    v83 = prompt.anchor.to_numpy(float)
    seed35 = prompt.expert.to_numpy(float)
    v101 = fuse(v83, seed35, .28)
    replace = fuse(v83, new, .50)
    artifact = artifact.score.to_numpy(float)
    y = truth.MUSIC_FAKE.to_numpy(np.int64)
    bases = {"v83": v83, "v101": v101, "replace_prompt_w50": replace}
    selected = {name: select(y, base, artifact) for name, base in bases.items()}
    variants = dict(bases)
    for name, base in bases.items():
        variants[f"{name}_artifact"] = fuse(base, artifact, selected[name][0])
    group_rows = []
    for column in ("DATASET", "CHANNEL", "MIX_MODE", "COMPONENT_CASE"):
        if column not in truth:
            continue
        values = truth[column].fillna("missing").astype(str)
        for value in sorted(values.unique()):
            mask = values.eq(value).to_numpy()
            if len(np.unique(y[mask])) != 2:
                continue
            row = {"column": column, "value": value, "rows": int(mask.sum())}
            row.update({name: eer(y[mask], score[mask]) for name, score in variants.items()})
            group_rows.append(row)
    datasets = truth.DATASET.fillna("missing").astype(str)
    lodo = []
    for heldout in sorted(datasets.unique()):
        fit = datasets.ne(heldout).to_numpy(); test = ~fit
        if len(np.unique(y[test])) != 2:
            continue
        for name, base in bases.items():
            weight, fit_eer = select(y[fit], base[fit], artifact[fit])
            lodo.append({
                "heldout": heldout, "base": name, "weight": weight,
                "fit_eer": fit_eer, "base_eer": eer(y[test], base[test]),
                "fused_eer": eer(y[test], fuse(base[test], artifact[test], weight)),
            })
    report = {
        "rows": len(truth), "artifactnet_eer": eer(y, artifact),
        "selected": {name: {"weight": pair[0], "eer": pair[1]}
                     for name, pair in selected.items()},
        "base_eer": {name: eer(y, value) for name, value in bases.items()},
        "lodo": {
            name: {
                "improved": int(sum(row["fused_eer"] < row["base_eer"]
                                    for row in lodo if row["base"] == name)),
                "comparable": int(sum(row["base"] == name for row in lodo)),
            } for name in bases
        },
    }
    args.output.mkdir(parents=True)
    pd.DataFrame(group_rows).to_csv(args.output / "group_eer.csv", index=False)
    pd.DataFrame(lodo).to_csv(args.output / "leave_one_dataset_out.csv", index=False)
    pd.DataFrame({"ID": truth.ID, "label": y, "artifactnet": artifact,
                  **variants}).to_csv(args.output / "predictions.csv", index=False)
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
