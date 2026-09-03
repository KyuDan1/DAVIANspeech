#!/usr/bin/env python3
"""Train a leakage-guarded EAT presence probe and audit source transfer."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.linear_model import SGDClassifier

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from data_guard import assert_no_locked_eval_leakage  # noqa: E402


TRAIN_BANKS = (
    "phone_router_voice_train_v1",
    "multigen_music_presence_train_v1",
    "external_mixed_train_v1",
    "mixed_devvoice_train_v1",
    "mixed_fmc_music_train_v1",
)
DEV_BANKS = (
    "factorial_eval_1200_v2_dev",
    "presence_source_disjoint_dev_v1",
)
TRUTH_OVERRIDES = {
    "factorial_eval_1200_v2_dev":
        ROOT / "data/eval/factorial_eval_1200_v2/truth_dev.csv",
    "factorial_eval_1200_v2_holdout":
        ROOT / "data/eval/factorial_eval_1200_v2/truth_holdout.csv",
}


def truth_path(name: str) -> Path:
    return TRUTH_OVERRIDES.get(name, ROOT / "data/eval" / name / "truth.csv")


def load_features(stats_root: Path, name: str) -> tuple[np.ndarray, np.ndarray]:
    ids, values, masks = [], [], []
    directory = stats_root / name / "eat" / "clean"
    for path in sorted(directory.glob("shard_*.npz")):
        shard = np.load(path, allow_pickle=False)
        ids.append(shard["ids"].astype(str))
        values.append(shard["statistics"].astype(np.float32))
        masks.append(shard["view_mask"])
    if not ids:
        raise FileNotFoundError(f"No EAT statistics in {directory}")
    ids = np.concatenate(ids)
    values = np.concatenate(values)
    masks = np.concatenate(masks)
    valid = masks[:, :, None, None]
    count = valid.sum(axis=1).clip(min=1)
    view_mean = (values * valid).sum(axis=1) / count
    view_max = np.where(valid, values, -np.inf).max(axis=1)
    # Mean retains stable clip evidence; max captures sequential voice/music
    # that occurs in just one of the start/middle/end views.
    features = np.concatenate((view_mean, view_max), axis=1).reshape(len(ids), -1)
    return ids, features.astype(np.float32)


def load_bank(stats_root: Path, name: str) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    ids, features = load_features(stats_root, name)
    truth = pd.read_csv(truth_path(name), dtype={"ID": str}).set_index("ID")
    missing = set(ids) - set(truth.index)
    if missing:
        raise ValueError(f"Missing truth for {name}: {sorted(missing)[:5]}")
    return ids, features, truth.loc[ids].reset_index()


def auc(y: np.ndarray, score: np.ndarray) -> float:
    positive = y.astype(bool)
    negative = ~positive
    # Exact Mann-Whitney AUC without importing metric machinery in deployment.
    order = pd.Series(score).rank(method="average").to_numpy()
    return float((order[positive].sum() - positive.sum() * (positive.sum() + 1) / 2)
                 / (positive.sum() * negative.sum()))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stats-root", type=Path,
                        default=ROOT / "output/dual_domain_stats_v1")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[20260902, 20260903, 20260904])
    parser.add_argument("--alphas", type=float, nargs="+",
                        default=[1e-2, 3e-3, 1e-3, 3e-4, 1e-4])
    args = parser.parse_args()

    for name in TRAIN_BANKS:
        assert_no_locked_eval_leakage(
            truth_path(name), ROOT / "configs/data_partitions.yaml"
        )
    train = [load_bank(args.stats_root, name) for name in TRAIN_BANKS]
    dev = {name: load_bank(args.stats_root, name) for name in DEV_BANKS}
    train_x = np.concatenate([bank[1] for bank in train])
    mean = train_x.mean(axis=0, dtype=np.float64).astype(np.float32)
    std = train_x.std(axis=0, dtype=np.float64).astype(np.float32).clip(min=1e-4)
    train_x = np.clip((train_x - mean) / std, -8, 8)
    dev_x = {
        name: np.clip((bank[1] - mean) / std, -8, 8)
        for name, bank in dev.items()
    }

    records = []
    selected = {}
    for component in ("VOICE", "MUSIC"):
        target = np.concatenate([
            bank[2][f"{component}_PRESENT"].to_numpy(np.int64) for bank in train
        ])
        candidates = []
        for alpha in args.alphas:
            for seed in args.seeds:
                model = SGDClassifier(
                    loss="log_loss", penalty="l2", alpha=alpha,
                    class_weight="balanced", average=True, max_iter=2000,
                    tol=1e-5, random_state=seed,
                ).fit(train_x, target)
                scores = {}
                for name, bank in dev.items():
                    y = bank[2][f"{component}_PRESENT"].to_numpy(np.int64)
                    scores[name] = auc(y, model.decision_function(dev_x[name]))
                selection = 0.5 * np.mean(list(scores.values())) + 0.5 * min(scores.values())
                row = {
                    "COMPONENT": component, "ALPHA": alpha, "SEED": seed,
                    "SELECTION": selection, **scores,
                }
                records.append(row)
                candidates.append((selection, model, row))
        _, best, row = max(candidates, key=lambda item: item[0])
        selected[component] = {
            "coefficient": best.coef_.astype(np.float32),
            "intercept": best.intercept_.astype(np.float32),
            "row": row,
        }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records).to_csv(args.output_dir / "sweep.csv", index=False)
    np.savez_compressed(
        args.output_dir / "presence_head.npz",
        mean=mean, std=std,
        voice_coefficient=selected["VOICE"]["coefficient"],
        voice_intercept=selected["VOICE"]["intercept"],
        music_coefficient=selected["MUSIC"]["coefficient"],
        music_intercept=selected["MUSIC"]["intercept"],
    )
    summary = {name: value["row"] for name, value in selected.items()}
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    for name, (_, _, truth) in dev.items():
        output = pd.DataFrame({"ID": truth.ID})
        for component in ("VOICE", "MUSIC"):
            item = selected[component]
            decision = (
                dev_x[name] @ item["coefficient"].reshape(-1)
                + item["intercept"].item()
            )
            output[f"PROBE_{component}_PRESENT_PROB"] = expit(decision)
        output.to_csv(args.output_dir / f"{name}.csv", index=False)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
