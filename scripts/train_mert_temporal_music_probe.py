#!/usr/bin/env python3
"""Train a generator-balanced music head on frozen temporal MERT statistics."""

from __future__ import annotations

import argparse
import copy
import json
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from data_guard import assert_no_locked_eval_leakage  # noqa: E402
from evaluate_diagnostic import official_eer  # noqa: E402
from train_dual_domain_head import truth_path  # noqa: E402
from train_long_horizon_music_probe import (  # noqa: E402
    DEV_DEFAULT,
    TRAIN_DEFAULT,
    component_partners,
    generator_environments,
    normalized_generator,
    sample_weights,
)


AUDIT_DEFAULT = (
    "factorial_eval_1200_v2_holdout",
    "phone_factorial_1200_v1",
    "yue_cross_component_audit_v1",
    "suno_vocals_v1",
)


def load_statistics(root: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    paths = sorted(root.glob("shard_*.npz"))
    if not paths:
        raise FileNotFoundError(f"No MERT temporal shards in {root}")
    datasets, ids, values = [], [], []
    for path in paths:
        with np.load(path, allow_pickle=False) as shard:
            if str(shard["representation"]) != "temporal_statistics":
                raise ValueError(f"Not a temporal-statistics shard: {path}")
            datasets.append(shard["datasets"].astype(str))
            ids.append(shard["ids"].astype(str))
            values.append(shard["embeddings"].astype(np.float16, copy=True))
    datasets = np.concatenate(datasets)
    ids = np.concatenate(ids)
    values = np.concatenate(values)
    keys = list(zip(datasets, ids))
    if len(keys) != len(set(keys)):
        raise ValueError("MERT temporal statistics contain duplicate dataset/ID keys")
    if values.shape[1:] != (3, 13, 2, 768):
        raise ValueError(f"Unexpected MERT temporal shape: {values.shape}")
    return datasets, ids, values


def load_metadata(datasets: np.ndarray, ids: np.ndarray) -> pd.DataFrame:
    frames = []
    for name in np.unique(datasets):
        truth = pd.read_csv(truth_path(str(name)), dtype={"ID": str})
        truth["DATASET"] = str(name)
        frames.append(truth)
    truth = pd.concat(frames, ignore_index=True).set_index(["DATASET", "ID"])
    keys = pd.MultiIndex.from_arrays([datasets, ids], names=["DATASET", "ID"])
    missing = keys.difference(truth.index)
    if len(missing):
        raise ValueError(f"Missing truth for {list(missing[:5])}")
    return truth.loc[keys].reset_index()


@torch.inference_mode()
def projected_features(
    statistics: np.ndarray,
    projection: torch.Tensor,
    device: torch.device,
    batch_size: int,
    feature_mode: str,
) -> np.ndarray:
    blocks = []
    for offset in range(0, len(statistics), batch_size):
        values = torch.from_numpy(statistics[offset:offset + batch_size]).to(
            device, dtype=torch.float32
        ).clamp_(-8, 8)
        values = F.layer_norm(values, (values.shape[-1],)) @ projection
        values = values.flatten(2, 4)
        mean = values.mean(dim=1)
        dispersion = values.std(dim=1, unbiased=False)
        if feature_mode == "mean":
            features = mean
        elif feature_mode == "dispersion":
            features = dispersion
        else:
            features = torch.cat((mean, dispersion), dim=1)
        count = torch.ones((len(values), 1), device=device, dtype=values.dtype)
        blocks.append(torch.cat((features, count), dim=1).cpu().numpy())
    return np.concatenate(blocks).astype(np.float32, copy=False)


def music_eer(frame: pd.DataFrame, scores: np.ndarray) -> float:
    labels = frame.MUSIC_FAKE.astype(int)
    if labels.nunique() < 2:
        return float("nan")
    return float(official_eer(labels, scores))


def evaluate(
    frame: pd.DataFrame, scores: np.ndarray, datasets: list[str],
) -> tuple[pd.DataFrame, float]:
    rows = []
    for name in datasets:
        mask = frame.DATASET.eq(name).to_numpy()
        rows.append({
            "DATASET": name, "N": int(mask.sum()),
            "MUSIC_EER": music_eer(frame.loc[mask], scores[mask]),
        })
    result = pd.DataFrame(rows)
    valid = result.MUSIC_EER.dropna()
    selection = 0.5 * (1 - valid.mean()) + 0.5 * (1 - valid.max())
    return result, float(selection)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--statistics", type=Path, default=ROOT / "output/mert_temporal_v2"
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--train-datasets", nargs="+", default=list(TRAIN_DEFAULT))
    parser.add_argument("--dev-datasets", nargs="+", default=list(DEV_DEFAULT))
    parser.add_argument("--audit-datasets", nargs="+", default=list(AUDIT_DEFAULT))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--projection-width", type=int, default=128)
    parser.add_argument("--projection-seed", type=int, default=20260904)
    parser.add_argument(
        "--feature-mode", choices=("mean", "dispersion", "both"), default="both"
    )
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=600)
    parser.add_argument("--eval-every", type=int, default=5)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--learning-rate", type=float, default=3e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--consistency-weight", type=float, default=0.05)
    parser.add_argument("--group-dro-temperature", type=float, default=0.0)
    parser.add_argument(
        "--exclude-music-generators", nargs="*", default=[],
        help="Remove named fake generators from training for strict LOGO audits.",
    )
    parser.add_argument("--seed", type=int, default=20260904)
    args = parser.parse_args()
    if args.group_dro_temperature < 0:
        parser.error("--group-dro-temperature must be non-negative")
    for name in args.train_datasets:
        assert_no_locked_eval_leakage(
            truth_path(name), ROOT / "configs/data_partitions.yaml"
        )

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device = torch.device(args.device)
    datasets, ids, statistics = load_statistics(args.statistics)
    frame = load_metadata(datasets, ids)
    expected = set(args.train_datasets + args.dev_datasets + args.audit_datasets)
    missing = expected.difference(frame.DATASET.unique())
    if missing:
        raise ValueError(f"Missing configured datasets: {sorted(missing)}")
    generator = torch.Generator(device="cpu").manual_seed(args.projection_seed)
    projection = (
        torch.randn(768, args.projection_width, generator=generator)
        / np.sqrt(768)
    ).to(device)
    features = projected_features(
        statistics, projection, device, args.batch_size, args.feature_mode
    )
    del statistics

    train_mask = frame.DATASET.isin(args.train_datasets).to_numpy(copy=True)
    excluded = {
        value.strip().lower() for value in args.exclude_music_generators
        if value.strip()
    }
    if excluded:
        generators = frame.apply(normalized_generator, axis=1)
        train_mask &= ~(
            frame.MUSIC_FAKE.eq(1).to_numpy()
            & generators.isin(excluded).to_numpy()
        )
    train_frame = frame.loc[train_mask].reset_index(drop=True)
    train_features = features[train_mask]
    mean = train_features.mean(axis=0, dtype=np.float64).astype(np.float32)
    std = train_features.std(axis=0, dtype=np.float64).clip(1e-4).astype(np.float32)
    features = np.clip((features - mean) / std, -8, 8).astype(np.float32)
    train_features = features[train_mask]
    weights = sample_weights(train_frame)
    pairs = component_partners(train_frame)
    environments = generator_environments(train_frame)
    print(json.dumps({
        "examples": len(frame), "train": len(train_frame),
        "features": features.shape[1], "pairs": len(pairs),
        "environments": len(environments),
    }), flush=True)

    x = torch.from_numpy(train_features).to(device)
    y = torch.from_numpy(
        train_frame.MUSIC_FAKE.astype(np.float32).to_numpy(copy=True)
    ).to(device)
    sample_weight = torch.from_numpy(weights).to(device)
    pair_index = torch.from_numpy(pairs).to(device)
    environment_indices = [
        (name, torch.from_numpy(positive).to(device),
         torch.from_numpy(negative).to(device))
        for name, positive, negative in environments
    ]
    evaluation_x = torch.from_numpy(features).to(device)
    classifier = nn.Linear(features.shape[1], 1).to(device)
    optimizer = torch.optim.AdamW(
        classifier.parameters(), lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    best_selection, best_state, best_epoch, stale = -np.inf, None, -1, 0
    history = []
    for epoch in range(args.epochs + 1):
        classifier.train()
        logits = classifier(x).squeeze(1)
        point_loss = F.binary_cross_entropy_with_logits(logits, y, reduction="none")
        if args.group_dro_temperature > 0:
            losses = torch.stack([
                0.5 * point_loss[positive].mean()
                + 0.5 * point_loss[negative].mean()
                for _, positive, negative in environment_indices
            ])
            temperature = args.group_dro_temperature
            base = temperature * (
                torch.logsumexp(losses / temperature, dim=0)
                - np.log(len(environment_indices))
            )
        else:
            base = (point_loss * sample_weight).mean()
        consistency = logits.new_zeros(())
        if len(pairs):
            consistency = F.smooth_l1_loss(
                logits[pair_index[:, 0]], logits[pair_index[:, 1]]
            )
        loss = base + args.consistency_weight * consistency
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if epoch % args.eval_every:
            continue
        classifier.eval()
        with torch.inference_mode():
            scores = classifier(evaluation_x).squeeze(1).sigmoid().cpu().numpy()
        dev, selection = evaluate(frame, scores, args.dev_datasets)
        history.append({
            "EPOCH": epoch, "LOSS": float(loss.detach()),
            "BASE_LOSS": float(base.detach()),
            "CONSISTENCY_LOSS": float(consistency.detach()),
            "SELECTION": selection, "MEAN_EER": dev.MUSIC_EER.mean(),
            "WORST_EER": dev.MUSIC_EER.max(),
        })
        if selection > best_selection + 1e-5:
            best_selection, best_epoch = selection, epoch
            best_state = copy.deepcopy(classifier.state_dict())
            stale = 0
        else:
            stale += 1
            if stale >= args.patience:
                break
        if epoch % 50 == 0:
            print(history[-1], flush=True)
    if best_state is None:
        raise RuntimeError("Training produced no checkpoint")
    classifier.load_state_dict(best_state)
    classifier.eval()
    with torch.inference_mode():
        scores = classifier(evaluation_x).squeeze(1).sigmoid().cpu().numpy()
    dev, selection = evaluate(frame, scores, args.dev_datasets)
    audit, _ = evaluate(frame, scores, args.audit_datasets)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.output_dir / "mert_temporal_music_head.npz",
        projection=projection.cpu().numpy(), mean=mean, std=std,
        weight=best_state["weight"].cpu().numpy().reshape(-1),
        bias=best_state["bias"].cpu().numpy().reshape(()),
        feature_mode=np.asarray(args.feature_mode),
        projection_seed=np.asarray(args.projection_seed),
        group_dro_temperature=np.asarray(args.group_dro_temperature),
        excluded_music_generators=np.asarray(sorted(excluded)),
    )
    pd.DataFrame(history).to_csv(args.output_dir / "history.csv", index=False)
    dev.to_csv(args.output_dir / "dev_metrics.csv", index=False)
    audit.to_csv(args.output_dir / "audit_metrics.csv", index=False)
    pd.DataFrame({
        "DATASET": frame.DATASET, "ID": frame.ID,
        "MERT_TEMPORAL_MUSIC_PROB": scores,
    }).to_csv(args.output_dir / "predictions.csv", index=False)
    summary = {
        "best_epoch": best_epoch, "selection": selection,
        "mean_dev_eer": float(dev.MUSIC_EER.mean()),
        "worst_dev_eer": float(dev.MUSIC_EER.max()),
        "feature_mode": args.feature_mode,
        "group_dro_temperature": args.group_dro_temperature,
        "excluded_music_generators": sorted(excluded),
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(dev.to_string(index=False))
    print(audit.to_string(index=False))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
