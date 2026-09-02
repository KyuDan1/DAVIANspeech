#!/usr/bin/env python3
"""Fit a source-disjoint telephone Voice-presence probe on frozen EAT stats."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.linear_model import SGDClassifier

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from data_guard import assert_no_locked_eval_leakage  # noqa: E402
from experiment_presence_probe import auc  # noqa: E402


VARIANTS = ("resample8k", "g711_ulaw", "g726_24k", "opus_nb_8k")


def aggregate(paths: list[Path]) -> tuple[np.ndarray, np.ndarray]:
    ids, values, masks = [], [], []
    for path in paths:
        shard = np.load(path, allow_pickle=False)
        ids.append(shard["ids"].astype(str))
        values.append(shard["statistics"].astype(np.float32))
        masks.append(shard["view_mask"])
    if not ids:
        raise FileNotFoundError("No EAT statistic shards")
    ids = np.concatenate(ids)
    values = np.concatenate(values)
    masks = np.concatenate(masks)
    valid = masks[:, :, None, None]
    count = valid.sum(axis=1).clip(min=1)
    view_mean = (values * valid).sum(axis=1) / count
    view_max = np.where(valid, values, -np.inf).max(axis=1)
    features = np.concatenate((view_mean, view_max), axis=1).reshape(len(ids), -1)
    return ids, features.astype(np.float32)


def clean_bank(stats_root: Path, name: str) -> tuple[np.ndarray, np.ndarray]:
    return aggregate(sorted((stats_root / name / "eat" / "clean").glob("shard_*.npz")))


def augmented_bank(
    stats_root: Path, name: str,
) -> tuple[np.ndarray, np.ndarray]:
    ids, features = [], []
    for variant in VARIANTS:
        bank_ids, bank_features = aggregate(sorted(
            (stats_root / name / "eat" / variant).glob("shard_*.npz")
        ))
        ids.append(np.asarray([f"{item}__{variant}" for item in bank_ids]))
        features.append(bank_features)
    return np.concatenate(ids), np.concatenate(features)


def label(ids: np.ndarray, truth_path: Path) -> np.ndarray:
    truth = pd.read_csv(truth_path, dtype={"ID": str}).set_index("ID")
    base_ids = np.asarray([item.rsplit("__", 1)[0] for item in ids])
    missing = set(base_ids) - set(truth.index)
    if missing:
        raise ValueError(f"Missing truth rows: {sorted(missing)[:5]}")
    return truth.loc[base_ids, "VOICE_PRESENT"].to_numpy(np.int64)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stats-root", type=Path,
        default=ROOT / "output" / "dual_domain_stats_v1",
    )
    parser.add_argument(
        "--augmented-root", type=Path,
        default=ROOT / "output" / "dual_domain_stats_v1_phone_aug",
    )
    parser.add_argument(
        "--development-root", type=Path,
        default=ROOT / "output" / "dual_domain_stats_v1_phone_dev",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[20260902, 20260903, 20260904])
    parser.add_argument(
        "--alphas", type=float, nargs="+",
        default=[3e-2, 1e-2, 3e-3, 1e-3, 3e-4, 1e-4],
    )
    args = parser.parse_args()

    training = (
        "phone_router_voice_train_v1",
        "multigen_music_presence_train_v1",
        "telephone_mixed_train_v1",
        "phone_presence_factorial_train_v1",
    )
    for name in training:
        assert_no_locked_eval_leakage(
            ROOT / "data" / "eval" / name / "truth.csv",
            ROOT / "configs" / "data_partitions.yaml",
        )

    voice_ids, voice_x = augmented_bank(
        args.augmented_root, "phone_router_voice_train_v1"
    )
    music_ids, music_x = augmented_bank(
        args.augmented_root, "multigen_music_presence_train_v1"
    )
    mixed_ids, mixed_x = clean_bank(args.stats_root, "telephone_mixed_train_v1")
    factorial_ids, factorial_x = clean_bank(
        args.stats_root, "phone_presence_factorial_train_v1"
    )
    train_x = np.concatenate((voice_x, music_x, mixed_x, factorial_x))
    train_y = np.concatenate((
        label(
            voice_ids,
            ROOT / "data/eval/phone_router_voice_train_v1/truth.csv",
        ),
        label(
            music_ids,
            ROOT / "data/eval/multigen_music_presence_train_v1/truth.csv",
        ),
        pd.read_csv(
            ROOT / "data/eval/telephone_mixed_train_v1/truth.csv",
            dtype={"ID": str},
        ).set_index("ID").loc[mixed_ids, "VOICE_PRESENT"].to_numpy(np.int64),
        pd.read_csv(
            ROOT / "data/eval/phone_presence_factorial_train_v1/truth.csv",
            dtype={"ID": str},
        ).set_index("ID").loc[factorial_ids, "VOICE_PRESENT"].to_numpy(np.int64),
    ))

    mean = train_x.mean(axis=0, dtype=np.float64).astype(np.float32)
    std = train_x.std(axis=0, dtype=np.float64).astype(np.float32).clip(min=1e-4)
    train_x = np.clip((train_x - mean) / std, -8, 8)

    music_dev_ids, music_dev_x = clean_bank(
        args.development_root, "source_disjoint_music_telephone_v1"
    )
    voice_dev_ids, voice_dev_x = clean_bank(
        args.development_root, "asvspoof_voice_telephone_dev_v1"
    )
    mixed_dev_ids, mixed_dev_x = clean_bank(args.stats_root, "telephone_mixed_dev_v1")
    dev = {
        "pure_source_disjoint": (
            np.concatenate((music_dev_x, voice_dev_x)),
            np.concatenate((np.zeros(len(music_dev_x)), np.ones(len(voice_dev_x)))),
        ),
        "mixed_source_disjoint": (
            np.concatenate((music_dev_x, mixed_dev_x)),
            np.concatenate((np.zeros(len(music_dev_x)), np.ones(len(mixed_dev_x)))),
        ),
    }
    dev = {
        name: (np.clip((x - mean) / std, -8, 8), y.astype(np.int64))
        for name, (x, y) in dev.items()
    }

    records, candidates = [], []
    for alpha in args.alphas:
        for seed in args.seeds:
            model = SGDClassifier(
                loss="log_loss", penalty="l2", alpha=alpha,
                class_weight="balanced", average=True, max_iter=3000,
                tol=1e-6, random_state=seed,
            ).fit(train_x, train_y)
            scores = {
                name: auc(y, model.decision_function(x))
                for name, (x, y) in dev.items()
            }
            selection = 0.5 * np.mean(list(scores.values())) + 0.5 * min(scores.values())
            row = {
                "ALPHA": alpha, "SEED": seed, "SELECTION": selection, **scores,
            }
            records.append(row)
            candidates.append((selection, min(scores.values()), model, row))
    _, _, best, best_row = max(candidates, key=lambda item: (item[0], item[1]))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records).to_csv(args.output_dir / "sweep.csv", index=False)
    np.savez_compressed(
        args.output_dir / "phone_voice_presence_head.npz",
        mean=mean, std=std,
        coefficient=best.coef_.astype(np.float32),
        intercept=best.intercept_.astype(np.float32),
        variants=np.asarray(VARIANTS),
    )
    predictions = []
    for name, (x, y) in dev.items():
        score = expit(best.decision_function(x) / 5.0)
        predictions.append(pd.DataFrame({
            "BANK": name, "VOICE_PRESENT": y,
            "PHONE_PROBE_VOICE_PRESENT_PROB": score,
        }))
    pd.concat(predictions, ignore_index=True).to_csv(
        args.output_dir / "dev_predictions.csv", index=False
    )
    summary = {
        "selected": best_row,
        "train_rows": int(len(train_y)),
        "train_positive": int(train_y.sum()),
        "train_negative": int((train_y == 0).sum()),
        "protocol": (
            "factorial_eval_1200_v2 and phone_factorial_1200_v1 are excluded "
            "from fitting and model selection"
        ),
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
