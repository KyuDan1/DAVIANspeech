#!/usr/bin/env python3
"""Probe generator-agnostic real-only density scores on frozen EAT statistics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.decomposition import PCA
from sklearn.metrics import roc_curve
from sklearn.neighbors import NearestNeighbors


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from data_guard import assert_no_locked_eval_leakage  # noqa: E402


TRAIN_BANKS = (
    "external_mixed_train_v1",
    "mixed_devvoice_train_v1",
    "mixed_fmc_music_train_v1",
    "mixfake_music_train_v1",
    "telephone_mixed_train_v1",
)
DEV_BANKS = (
    "external_mixed_v1",
    "factorial_eval_1200_v2_dev",
    "source_disjoint_mixed_equal_v1",
    "source_disjoint_mixed_v1",
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
    paths = sorted((stats_root / name / "eat" / "clean").glob("shard_*.npz"))
    if not paths:
        raise FileNotFoundError(f"No EAT statistics for {name}")
    for path in paths:
        shard = np.load(path, allow_pickle=False)
        ids.append(shard["ids"].astype(str))
        values.append(shard["statistics"].astype(np.float32))
        masks.append(shard["view_mask"])
    ids = np.concatenate(ids)
    values = np.concatenate(values)
    masks = np.concatenate(masks)
    valid = masks[:, :, None, None]
    count = valid.sum(axis=1).clip(min=1)
    view_mean = (values * valid).sum(axis=1) / count
    view_max = np.where(valid, values, -np.inf).max(axis=1)
    features = np.concatenate((view_mean, view_max), axis=1).reshape(len(ids), -1)
    return ids, features.astype(np.float32)


def aligned_truth(name: str, ids: np.ndarray) -> pd.DataFrame:
    truth = pd.read_csv(truth_path(name), dtype={"ID": str}).set_index("ID")
    missing = set(ids) - set(truth.index)
    if missing:
        raise ValueError(f"Missing truth rows for {name}: {sorted(missing)[:5]}")
    return truth.loc[ids].reset_index()


def official_eer(labels: np.ndarray, scores: np.ndarray) -> float:
    fpr, tpr, _ = roc_curve(labels, scores, pos_label=1, drop_intermediate=False)
    fnr = 1 - tpr
    index = np.argmin(np.abs(fpr - fnr))
    return float((fpr[index] + fnr[index]) / 2)


def robust_location_scale(values: np.ndarray) -> tuple[float, float]:
    centre = float(np.median(values))
    scale = float(1.4826 * np.median(np.abs(values - centre)))
    return centre, max(scale, 1e-6)


class RealDensity:
    """PCA residual, Gaussian radius, and local-neighbour realness model."""

    def __init__(self, components: int, neighbors: int, seed: int) -> None:
        self.components = components
        self.neighbors = neighbors
        self.seed = seed

    def fit(self, features: np.ndarray) -> "RealDensity":
        self.mean = features.mean(axis=0, dtype=np.float64).astype(np.float32)
        self.scale = features.std(axis=0, dtype=np.float64).astype(np.float32)
        self.scale = np.maximum(self.scale, 1e-4)
        normalized = np.clip((features - self.mean) / self.scale, -8, 8)
        self.pca = PCA(
            n_components=min(self.components, len(features) - 1),
            svd_solver="randomized", iterated_power=3, random_state=self.seed,
        ).fit(normalized)
        latent = self.pca.transform(normalized)
        self.latent_scale = np.sqrt(np.maximum(self.pca.explained_variance_, 1e-6))
        unit = latent / np.maximum(np.linalg.norm(latent, axis=1, keepdims=True), 1e-6)
        self.reference = unit.astype(np.float32)
        self.knn = NearestNeighbors(
            n_neighbors=self.neighbors + 1, metric="cosine", n_jobs=6,
        ).fit(self.reference)
        raw = self._raw(normalized, training=True)
        self.calibration = {
            key: robust_location_scale(value) for key, value in raw.items()
        }
        return self

    def _raw(self, normalized: np.ndarray, training: bool = False) -> dict[str, np.ndarray]:
        latent = self.pca.transform(normalized)
        whitened = latent / self.latent_scale
        radius = np.mean(np.square(whitened), axis=1)
        reconstructed = self.pca.inverse_transform(latent)
        residual = np.mean(np.square(normalized - reconstructed), axis=1)
        unit = latent / np.maximum(np.linalg.norm(latent, axis=1, keepdims=True), 1e-6)
        count = self.neighbors + 1 if training else self.neighbors
        distances = self.knn.kneighbors(unit, n_neighbors=count)[0]
        if training:
            distances = distances[:, 1:]
        knn = distances.mean(axis=1)
        return {"radius": radius, "residual": residual, "knn": knn}

    def scores(self, features: np.ndarray) -> dict[str, np.ndarray]:
        normalized = np.clip((features - self.mean) / self.scale, -8, 8)
        raw = self._raw(normalized)
        standardized = {
            key: (value - self.calibration[key][0]) / self.calibration[key][1]
            for key, value in raw.items()
        }
        standardized["ensemble"] = np.mean(
            np.column_stack(tuple(standardized.values())), axis=1
        )
        return standardized


def balanced_real_rows(
    stats_root: Path, name: str, target: str, cap: int, seed: int,
) -> np.ndarray:
    ids, features = load_features(stats_root, name)
    truth = aligned_truth(name, ids)
    if target == "MUSIC":
        selected = truth.MUSIC_PRESENT.eq(1) & truth.MUSIC_FAKE.eq(0)
    else:
        selected = truth.FILE_FAKE.eq(0)
    indices = np.flatnonzero(selected.to_numpy())
    if len(indices) > cap:
        indices = np.random.default_rng(seed).choice(indices, cap, replace=False)
    return features[np.sort(indices)]


def evaluate(
    model: RealDensity, stats_root: Path, banks: tuple[str, ...], target: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    metrics, predictions = [], []
    label = f"{target}_FAKE"
    present = f"{target}_PRESENT"
    for name in banks:
        ids, features = load_features(stats_root, name)
        truth = aligned_truth(name, ids)
        selected = truth[present].eq(1).to_numpy() if present in truth else np.ones(len(truth), bool)
        scores = model.scores(features)
        row = {"BANK": name, "N": int(selected.sum())}
        for method, values in scores.items():
            row[f"EER_{method}"] = official_eer(
                truth.loc[selected, label].to_numpy(), values[selected]
            )
        metrics.append(row)
        frame = pd.DataFrame({"BANK": name, "ID": ids, label: truth[label]})
        for method, values in scores.items():
            frame[method] = values
            frame[f"prob_{method}"] = expit(np.clip(values / 5.0, -60, 60))
        predictions.append(frame)
    return pd.DataFrame(metrics), pd.concat(predictions, ignore_index=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stats-root", type=Path,
        default=ROOT / "output/dual_domain_stats_v1",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--components", type=int, default=128)
    parser.add_argument("--neighbors", type=int, default=10)
    parser.add_argument("--bank-cap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260902)
    args = parser.parse_args()

    for name in TRAIN_BANKS:
        assert_no_locked_eval_leakage(
            truth_path(name), ROOT / "configs/data_partitions.yaml"
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary = {}
    for target in ("FILE", "MUSIC"):
        train = np.concatenate([
            balanced_real_rows(
                args.stats_root, name, target, args.bank_cap, args.seed + index
            )
            for index, name in enumerate(TRAIN_BANKS)
        ])
        model = RealDensity(args.components, args.neighbors, args.seed).fit(train)
        metrics, predictions = evaluate(model, args.stats_root, DEV_BANKS, target)
        metrics.to_csv(args.output_dir / f"{target.lower()}_dev_metrics.csv", index=False)
        predictions.to_csv(
            args.output_dir / f"{target.lower()}_dev_predictions.csv", index=False
        )
        joblib.dump(model, args.output_dir / f"{target.lower()}_real_density.joblib")
        summary[target] = {
            "train_rows": len(train),
            "mean_eer": {
                column.removeprefix("EER_"): float(metrics[column].mean())
                for column in metrics if column.startswith("EER_")
            },
            "worst_eer": {
                column.removeprefix("EER_"): float(metrics[column].max())
                for column in metrics if column.startswith("EER_")
            },
        }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
