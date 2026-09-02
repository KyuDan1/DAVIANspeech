#!/usr/bin/env python3
"""Train a low-capacity component head on frozen original-mixture MERT embeddings."""

from __future__ import annotations

import argparse
import copy
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from evaluate_diagnostic import official_eer  # noqa: E402
from train_dual_domain_head import DEV_DEFAULT, TRAIN_DEFAULT, truth_path  # noqa: E402


def load_embeddings(root: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    shards = [np.load(path, allow_pickle=False) for path in sorted(root.glob("shard_*.npz"))]
    if not shards:
        raise FileNotFoundError(f"No shards in {root}")
    return (
        np.concatenate([item["datasets"].astype(str) for item in shards]),
        np.concatenate([item["ids"].astype(str) for item in shards]),
        np.concatenate([item["embeddings"] for item in shards]),
    )


def metadata(datasets: np.ndarray, ids: np.ndarray) -> pd.DataFrame:
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


def component_pairs(
    frame: pd.DataFrame, indices: np.ndarray, source: str, target: str, nuisance: str
) -> np.ndarray:
    groups: dict[tuple[str, int], dict[int, list[int]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for index in indices:
        row = frame.iloc[index]
        value = row.get(source)
        if pd.isna(value) or not str(value).strip():
            continue
        groups[(str(value), int(row[target]))][int(row[nuisance])].append(int(index))
    pairs = []
    for nuisances in groups.values():
        values = sorted(nuisances)
        if len(values) < 2:
            continue
        for position, first in enumerate(values):
            second = values[(position + 1) % len(values)]
            for offset, left in enumerate(nuisances[first]):
                pairs.append((left, nuisances[second][offset % len(nuisances[second])]))
    return np.asarray(pairs, dtype=np.int64).reshape(-1, 2)


def channel_pairs(frame: pd.DataFrame, indices: np.ndarray) -> np.ndarray:
    allowed = set(int(item) for item in indices)
    by_id = {str(frame.iloc[item].ID): int(item) for item in indices}
    pairs = []
    if "PARENT_ID" not in frame:
        return np.empty((0, 2), dtype=np.int64)
    for item in indices:
        parent = frame.iloc[item].get("PARENT_ID")
        if pd.isna(parent) or str(parent) not in by_id:
            continue
        other = by_id[str(parent)]
        if other in allowed:
            pairs.append((int(item), other))
    return np.asarray(pairs, dtype=np.int64).reshape(-1, 2)


def metrics(frame: pd.DataFrame, scores: np.ndarray) -> dict[str, float]:
    present_voice = frame.VOICE_PRESENT.eq(1).to_numpy()
    present_music = frame.MUSIC_PRESENT.eq(1).to_numpy()
    file_eer = official_eer(frame.FILE_FAKE, scores[:, 2])
    voice_eer = official_eer(frame.loc[present_voice, "VOICE_FAKE"], scores[present_voice, 0])
    music_eer = official_eer(frame.loc[present_music, "MUSIC_FAKE"], scores[present_music, 1])
    return {
        "FILE_EER": file_eer, "VOICE_EER": voice_eer, "MUSIC_EER": music_eer,
        "ADS": .5 * (1 - file_eer) + .2 * (1 - voice_eer) + .3 * (1 - music_eer),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--embeddings", type=Path,
                        default=ROOT / "output/mert_embeddings_v1")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--train-datasets", nargs="+", default=list(TRAIN_DEFAULT))
    parser.add_argument("--dev-datasets", nargs="+", default=list(DEV_DEFAULT))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--eval-every", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=0.03)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--component-consistency", type=float, default=0.0)
    parser.add_argument("--channel-consistency", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=20260902)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    datasets, ids, embeddings = load_embeddings(args.embeddings)
    frame = metadata(datasets, ids)
    train_mask = np.isin(datasets, args.train_datasets)
    train_indices = np.flatnonzero(train_mask)
    if not len(train_indices):
        raise ValueError("No training samples")
    mean = embeddings[train_mask].mean(axis=0, dtype=np.float64).astype(np.float32)
    std = embeddings[train_mask].std(axis=0, dtype=np.float64).clip(1e-4).astype(np.float32)
    features = np.clip((embeddings - mean) / std, -8, 8).astype(np.float32)
    targets = frame[["VOICE_FAKE", "MUSIC_FAKE", "FILE_FAKE"]].fillna(0).to_numpy(np.float32)

    # Equal total mass per corpus and per binary class/task.
    corpus_counts = frame.loc[train_mask, "DATASET"].value_counts()
    sample_weights = np.asarray([
        1 / corpus_counts[name] for name in frame.loc[train_mask, "DATASET"]
    ], dtype=np.float32)
    task_weights = np.empty((len(train_indices), 3), dtype=np.float32)
    for task in range(3):
        labels = targets[train_mask, task].astype(int)
        counts = np.bincount(labels, minlength=2)
        task_weights[:, task] = sample_weights / counts[labels].clip(1)
    task_weights /= task_weights.mean(axis=0, keepdims=True)

    music_pairs = component_pairs(
        frame, train_indices, "MUSIC_SOURCE_ID", "MUSIC_FAKE", "VOICE_FAKE"
    )
    voice_pairs = component_pairs(
        frame, train_indices, "VOICE_SOURCE_ID", "VOICE_FAKE", "MUSIC_FAKE"
    )
    phone_pairs = channel_pairs(frame, train_indices)
    print(json.dumps({
        "train": len(train_indices), "music_pairs": len(music_pairs),
        "voice_pairs": len(voice_pairs), "channel_pairs": len(phone_pairs),
    }))

    device = torch.device(args.device)
    x = torch.from_numpy(features).to(device)
    y = torch.from_numpy(targets).to(device)
    weight = torch.from_numpy(task_weights).to(device)
    train_tensor = torch.from_numpy(train_indices).to(device)
    pair_tensors = [torch.from_numpy(value).to(device) for value in (
        music_pairs, voice_pairs, phone_pairs
    )]
    model = nn.Linear(features.shape[1], 3).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    history, best_state, best_rows = [], None, None
    best_selection = -float("inf")
    for epoch in range(args.epochs + 1):
        model.train()
        logits = model(x[train_tensor])
        base = (
            F.binary_cross_entropy_with_logits(
                logits, y[train_tensor], reduction="none"
            ) * weight
        ).mean()
        pair_loss = torch.zeros((), device=device)
        if len(music_pairs):
            pair_loss = pair_loss + args.component_consistency * F.mse_loss(
                model(x[pair_tensors[0][:, 0]])[:, 1],
                model(x[pair_tensors[0][:, 1]])[:, 1],
            )
        if len(voice_pairs):
            pair_loss = pair_loss + args.component_consistency * F.mse_loss(
                model(x[pair_tensors[1][:, 0]])[:, 0],
                model(x[pair_tensors[1][:, 1]])[:, 0],
            )
        if len(phone_pairs):
            pair_loss = pair_loss + args.channel_consistency * F.mse_loss(
                model(x[pair_tensors[2][:, 0]]),
                model(x[pair_tensors[2][:, 1]]),
            )
        loss = base + pair_loss
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if epoch % args.eval_every:
            continue
        model.eval()
        with torch.inference_mode():
            scores = model(x).sigmoid().cpu().numpy()
        rows = []
        for name in args.dev_datasets:
            mask = datasets == name
            row = {"DATASET": name, **metrics(frame.loc[mask].reset_index(drop=True), scores[mask])}
            rows.append(row)
        result = pd.DataFrame(rows)
        selection = 0.5 * result.ADS.mean() + 0.5 * result.ADS.min()
        history.append({
            "EPOCH": epoch, "LOSS": float(loss.detach()),
            "BASE_LOSS": float(base.detach()), "PAIR_LOSS": float(pair_loss.detach()),
            "SELECTION": selection, "MEAN_ADS": result.ADS.mean(),
            "WORST_ADS": result.ADS.min(),
        })
        if selection > best_selection:
            best_selection = float(selection)
            best_state = copy.deepcopy(model.state_dict())
            best_rows = result.copy()
        if epoch % 100 == 0:
            print(history[-1], flush=True)

    if best_state is None:
        raise RuntimeError("No checkpoint selected")
    model.load_state_dict(best_state)
    model.eval()
    with torch.inference_mode():
        scores = model(x).sigmoid().cpu().numpy()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.output_dir / "mert_linear_head.npz",
        weight=best_state["weight"].cpu().numpy(),
        bias=best_state["bias"].cpu().numpy(), mean=mean, std=std,
    )
    pd.DataFrame(history).to_csv(args.output_dir / "history.csv", index=False)
    best_rows.to_csv(args.output_dir / "dev_metrics.csv", index=False)
    audit_names = [
        name for name in np.unique(datasets)
        if name not in set(args.train_datasets) | set(args.dev_datasets)
    ]
    audits = []
    for name in audit_names:
        mask = datasets == name
        audits.append({
            "DATASET": name,
            **metrics(frame.loc[mask].reset_index(drop=True), scores[mask]),
        })
    pd.DataFrame(audits).to_csv(args.output_dir / "audit_metrics.csv", index=False)
    summary = {
        "selection": best_selection,
        "mean_dev_ads": float(best_rows.ADS.mean()),
        "worst_dev_ads": float(best_rows.ADS.min()),
        "component_consistency": args.component_consistency,
        "channel_consistency": args.channel_consistency,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(best_rows.to_string(index=False))
    if audits:
        print(pd.DataFrame(audits).to_string(index=False))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
