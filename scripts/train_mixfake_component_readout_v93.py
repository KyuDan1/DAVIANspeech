#!/usr/bin/env python3
"""Train a channel-consistent component readout on reserved MixFake v93.

Only frozen WPT embeddings are consumed.  The prospective locked cache is not
accepted by this program, making it impossible to tune against the final bank.
"""
from __future__ import annotations

import argparse
from collections import Counter
import copy
import hashlib
import json
from pathlib import Path
import random
import sys
import time

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_curve
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from mixfake_component_readout_v93 import ComponentReadoutV93  # noqa: E402


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def eer(labels, scores) -> float:
    labels = np.asarray(labels, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    if len(np.unique(labels)) != 2 or not np.isfinite(scores).all():
        return float("nan")
    fpr, tpr, _ = roc_curve(labels, scores, pos_label=1, drop_intermediate=False)
    fnr = 1 - tpr
    index = int(np.argmin(np.abs(fpr - fnr)))
    return float((fpr[index] + fnr[index]) / 2)


def load_shards(paths: list[Path]) -> dict[str, np.ndarray]:
    if not paths:
        raise ValueError("no cache shards")
    pieces: dict[str, list[np.ndarray]] = {}
    scalar_keys = {"channel", "manifest_sha256", "checkpoint_sha256", "views", "window"}
    scalars: dict[str, object] = {}
    for path in sorted(paths):
        with np.load(path, allow_pickle=False) as archive:
            for key in archive.files:
                value = archive[key]
                if key in scalar_keys:
                    item = value.item()
                    if key in scalars and scalars[key] != item:
                        raise ValueError(f"inconsistent {key} across shards")
                    scalars[key] = item
                else:
                    pieces.setdefault(key, []).append(value)
    merged = {key: np.concatenate(values) for key, values in pieces.items()}
    order = np.argsort(merged["positions"])
    if len(np.unique(merged["positions"])) != len(order):
        raise ValueError("duplicate cache positions")
    merged = {key: value[order] for key, value in merged.items()}
    merged.update({key: np.asarray(value) for key, value in scalars.items()})
    if len(set(merged["ids"].astype(str))) != len(order):
        raise ValueError("duplicate cache IDs")
    return merged


def discover(root: Path, role: str, channel: str) -> list[Path]:
    paths = sorted(root.glob(f"{role}_{channel}_shard*.npz"))
    if not paths:
        raise FileNotFoundError(f"no {role}/{channel} caches under {root}")
    return paths


def aligned(clean: dict[str, np.ndarray], changed: dict[str, np.ndarray]) -> None:
    for key in (
        "positions", "ids", "file_fake", "voice_fake", "music_fake",
        "component_case", "voice_source_id", "music_group_id", "music_generator",
    ):
        if not np.array_equal(clean[key], changed[key]):
            raise ValueError(f"paired caches disagree on {key}")
    if clean["manifest_sha256"].item() != changed["manifest_sha256"].item():
        raise ValueError("paired caches use different manifests")


class PairedFeatures(Dataset):
    def __init__(self, clean: dict[str, np.ndarray], changed: dict[str, np.ndarray]):
        aligned(clean, changed)
        self.clean = clean
        self.changed = changed

    def __len__(self) -> int:
        return len(self.clean["ids"])

    def __getitem__(self, index: int):
        targets = np.asarray([
            self.clean["voice_fake"][index], self.clean["music_fake"][index],
            self.clean["file_fake"][index],
        ], dtype=np.float32)
        return (
            self.clean["embeddings"][index].astype(np.float32),
            self.clean["base_logits"][index].astype(np.float32),
            self.changed["embeddings"][index].astype(np.float32),
            self.changed["base_logits"][index].astype(np.float32),
            targets, index,
        )


def sample_weights(cache: dict[str, np.ndarray]) -> torch.Tensor:
    # Equal component-cell mass; within each cell, equal real/fake music-source
    # family mass. This blocks the much larger Suno/Udio subsets from defining
    # the decision boundary by themselves.
    keys = list(zip(
        cache["component_case"].astype(str),
        cache["music_generator"].astype(str),
    ))
    counts = Counter(keys)
    families = Counter(key[0] for key in counts)
    weights = np.asarray([
        1 / (counts[key] * families[key[0]]) for key in keys
    ], dtype=np.float64)
    return torch.from_numpy(weights / weights.mean())


def ranking_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    terms = []
    for task in range(3):
        positive = logits[targets[:, task].eq(1), task]
        negative = logits[targets[:, task].eq(0), task]
        if len(positive) and len(negative):
            terms.append(F.softplus(-(positive[:, None] - negative[None, :])).mean())
    return torch.stack(terms).mean() if terms else logits.sum() * 0


def music_voice_independence(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    terms = []
    for music_label in (0, 1):
        subset = targets[:, 1].eq(music_label)
        real_voice = logits[subset & targets[:, 0].eq(0), 1]
        fake_voice = logits[subset & targets[:, 0].eq(1), 1]
        if len(real_voice) and len(fake_voice):
            terms.append((real_voice.mean() - fake_voice.mean()).square())
    return torch.stack(terms).mean() if terms else logits.sum() * 0


@torch.inference_mode()
def predict(model: ComponentReadoutV93, cache: dict[str, np.ndarray], device: torch.device,
            batch_size: int) -> np.ndarray:
    model.eval()
    outputs = []
    for start in range(0, len(cache["ids"]), batch_size):
        stop = start + batch_size
        embeddings = torch.from_numpy(
            cache["embeddings"][start:stop].astype(np.float32)
        ).to(device)
        logits = torch.from_numpy(
            cache["base_logits"][start:stop].astype(np.float32)
        ).to(device)
        output, _ = model(embeddings, logits)
        outputs.append(output.sigmoid().cpu().numpy())
    return np.concatenate(outputs)


def metrics(cache: dict[str, np.ndarray], probabilities: np.ndarray) -> dict:
    targets = np.stack((
        cache["voice_fake"], cache["music_fake"], cache["file_fake"],
    ), axis=1)
    eers = [eer(targets[:, task], probabilities[:, task]) for task in range(3)]
    ads = .2 * (1 - eers[0]) + .3 * (1 - eers[1]) + .5 * (1 - eers[2])
    cells = {}
    for case in sorted(set(cache["component_case"].astype(str))):
        mask = cache["component_case"].astype(str) == case
        cells[case] = {
            "n": int(mask.sum()),
            "music_mean": float(probabilities[mask, 1].mean()),
            "file_mean": float(probabilities[mask, 2].mean()),
        }
    generators = {}
    names = cache["music_generator"].astype(str)
    real = cache["music_fake"].eq(0) if isinstance(cache["music_fake"], pd.Series) else cache["music_fake"] == 0
    for name in sorted(set(names[cache["music_fake"] == 1])):
        mask = real | (names == name)
        generators[name] = {
            "n_fake": int((names == name).sum()),
            "music_eer_vs_all_real": eer(cache["music_fake"][mask], probabilities[mask, 1]),
        }
    return {
        "voice_eer": eers[0], "music_eer": eers[1], "file_eer": eers[2],
        "ads": ads, "cells": cells, "generators": generators,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--train-channel", default="paired_fast")
    parser.add_argument("--dev-channels", nargs="+", default=["clean", "paired_fast"])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=20260924)
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=.15)
    parser.add_argument("--residual-limit", type=float, default=4.)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--eval-batch-size", type=int, default=1024)
    parser.add_argument("--samples-per-epoch", type=int, default=16_384)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--ranking-weight", type=float, default=.10)
    parser.add_argument("--consistency-weight", type=float, default=.10)
    parser.add_argument("--independence-weight", type=float, default=.05)
    parser.add_argument("--residual-weight", type=float, default=.002)
    parser.add_argument("--voice-weight", type=float, default=.20)
    parser.add_argument("--music-weight", type=float, default=.30)
    parser.add_argument("--file-weight", type=float, default=.50)
    parser.add_argument(
        "--selection-axis", choices=("overall", "music", "background"),
        default="overall",
    )
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if "locked" in " ".join(str(path) for path in [args.cache_root, args.output]).lower():
        raise ValueError("locked evaluation paths are forbidden during training")
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    device = torch.device(args.device)

    train_clean_paths = discover(args.cache_root, "train", "clean")
    train_changed_paths = discover(args.cache_root, "train", args.train_channel)
    train_clean = load_shards(train_clean_paths)
    train_changed = load_shards(train_changed_paths)
    development = {
        channel: load_shards(discover(args.cache_root, "development", channel))
        for channel in args.dev_channels
    }
    for cache in development.values():
        if set(cache["ids"].astype(str)) & set(train_clean["ids"].astype(str)):
            raise ValueError("train/development ID overlap")
    dataset = PairedFeatures(train_clean, train_changed)
    sampler = WeightedRandomSampler(
        sample_weights(train_clean), args.samples_per_epoch, replacement=True,
        generator=torch.Generator().manual_seed(args.seed),
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, sampler=sampler,
                        num_workers=0, pin_memory=True)

    embedding_dim = int(train_clean["embeddings"].shape[-1])
    model = ComponentReadoutV93(
        embedding_dim=embedding_dim, width=args.width, dropout=args.dropout,
        residual_limit=args.residual_limit,
    ).to(device)
    flattened = train_clean["embeddings"].astype(np.float32).reshape(-1, embedding_dim)
    model.set_normalization(
        torch.from_numpy(flattened.mean(axis=0)).to(device),
        torch.from_numpy(flattened.std(axis=0)).to(device),
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=args.weight_decay)
    task_weight = torch.tensor(
        [args.voice_weight, args.music_weight, args.file_weight], device=device
    )
    if (task_weight < 0).any() or float(task_weight.sum()) <= 0:
        raise ValueError("task weights must be non-negative with positive sum")
    task_weight = task_weight / task_weight.sum()

    def select(reports: dict[str, dict]) -> tuple[float, float, float]:
        if args.selection_axis == "music":
            values = [1 - report["music_eer"] for report in reports.values()]
        elif args.selection_axis == "background":
            values = [
                .375 * (1 - report["music_eer"])
                + .625 * (1 - report["file_eer"])
                for report in reports.values()
            ]
        else:
            values = [report["ads"] for report in reports.values()]
        mean_value, worst_value = float(np.mean(values)), float(np.min(values))
        return .65 * mean_value + .35 * worst_value, mean_value, worst_value
    initial_reports = {
        channel: metrics(cache, predict(model, cache, device, args.eval_batch_size))
        for channel, cache in development.items()
    }
    initial_selection, initial_mean, initial_worst = select(initial_reports)
    best = initial_selection
    best_state = copy.deepcopy(model.state_dict())
    best_report = copy.deepcopy(initial_reports)
    stale = 0
    history = [{
        "epoch": -1, "loss": float("nan"), "selection": initial_selection,
        "mean_objective": initial_mean,
        "worst_objective": initial_worst, "seconds": 0.0,
        **{
            f"{channel}_{key}": report[key]
            for channel, report in initial_reports.items()
            for key in ("voice_eer", "music_eer", "file_eer", "ads")
        },
    }]
    print(json.dumps(history[0]), flush=True)
    started = time.monotonic()
    for epoch in range(args.epochs):
        model.train()
        losses = []
        for clean_e, clean_l, changed_e, changed_l, targets, _ in loader:
            clean_e, clean_l = clean_e.to(device), clean_l.to(device)
            changed_e, changed_l = changed_e.to(device), changed_l.to(device)
            targets = targets.to(device)
            clean, clean_residual = model(clean_e, clean_l)
            changed, changed_residual = model(changed_e, changed_l)
            clean_bce = F.binary_cross_entropy_with_logits(
                clean, targets, reduction="none"
            ).mean(dim=0)
            changed_bce = F.binary_cross_entropy_with_logits(
                changed, targets, reduction="none"
            ).mean(dim=0)
            bce = (.5 * (clean_bce + changed_bce) * task_weight).sum()
            rank = .5 * (ranking_loss(clean, targets) + ranking_loss(changed, targets))
            consistency = F.smooth_l1_loss(changed, clean.detach())
            independence = .5 * (
                music_voice_independence(clean, targets)
                + music_voice_independence(changed, targets)
            )
            residual_penalty = .5 * (
                clean_residual.square().mean() + changed_residual.square().mean()
            )
            loss = (
                bce + args.ranking_weight * rank
                + args.consistency_weight * consistency
                + args.independence_weight * independence
                + args.residual_weight * residual_penalty
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 3., error_if_nonfinite=True)
            optimizer.step()
            losses.append(float(loss.detach()))

        reports = {}
        for channel, cache in development.items():
            probability = predict(model, cache, device, args.eval_batch_size)
            reports[channel] = metrics(cache, probability)
        selection, mean_objective, worst_objective = select(reports)
        record = {
            "epoch": epoch, "loss": float(np.mean(losses)),
            "selection": selection, "mean_objective": mean_objective,
            "worst_objective": worst_objective,
            "seconds": time.monotonic() - started,
        }
        for channel, report in reports.items():
            for key in ("voice_eer", "music_eer", "file_eer", "ads"):
                record[f"{channel}_{key}"] = report[key]
        history.append(record)
        print(json.dumps(record), flush=True)
        if selection > best + 1e-5:
            best = selection
            best_state = copy.deepcopy(model.state_dict())
            best_report = copy.deepcopy(reports)
            stale = 0
        else:
            stale += 1
            if stale >= args.patience:
                break

    args.output.mkdir(parents=True)
    provenance = {
        "schema": "mixfake_component_readout_v93",
        "args": vars(args) | {"cache_root": str(args.cache_root), "output": str(args.output)},
        "train_rows": len(train_clean["ids"]),
        "development_rows": {key: len(value["ids"]) for key, value in development.items()},
        "train_ids_sha256": hashlib.sha256("\n".join(train_clean["ids"].astype(str)).encode()).hexdigest(),
        "cache_files": [
            {"path": str(path), "sha256": sha256_file(path)}
            for path in train_clean_paths + train_changed_paths
            + sum((discover(args.cache_root, "development", channel)
                   for channel in args.dev_channels), [])
        ],
        "locked_scores_read": False,
        "initial_selection": initial_selection,
        "best_selection": best,
        "development_gain": best - initial_selection,
    }
    torch.save({
        "model_type": "mixfake_component_readout_v93",
        "state": best_state,
        "config": {
            "embedding_dim": embedding_dim, "width": args.width,
            "dropout": args.dropout, "residual_limit": args.residual_limit,
            "temperature": 5.0,
        },
        "selection": best, "seed": args.seed, "provenance": provenance,
    }, args.output / "component_readout.pt")
    pd.DataFrame(history).to_csv(args.output / "history.csv", index=False)
    (args.output / "development_report.json").write_text(
        json.dumps(best_report, indent=2) + "\n"
    )
    (args.output / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(json.dumps({"status": "complete", "best": best,
                      "epochs": len(history), "output": str(args.output)}), flush=True)


if __name__ == "__main__":
    main()
