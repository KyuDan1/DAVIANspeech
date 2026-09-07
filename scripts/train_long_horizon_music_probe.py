#!/usr/bin/env python3
"""Train a generator-balanced music probe on frozen multi-view SSL statistics.

The deployed invariant head already projects EAT and SPEAR statistics into a
compact shared space.  This experiment freezes those projections and gives a
linear music classifier two kinds of original-audio evidence:

* the mean representation across start/middle/end crops; and
* the representation dispersion across those crops.

The latter is deliberately a track-internal cue.  It cannot identify a sample
merely from one local Suno/codec fingerprint, and adds no new backbone pass at
deployment time.  Model selection uses only configured development banks.
"""

from __future__ import annotations

import argparse
import copy
import json
import random
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

from data_guard import assert_no_locked_eval_leakage  # noqa: E402
from evaluate_diagnostic import official_eer  # noqa: E402
from invariant_dual_domain_head import InvariantDualDomainHead  # noqa: E402
from train_dual_domain_head import Bank, load_bank, truth_path  # noqa: E402


TRAIN_DEFAULT = (
    "external_mixed_train_v1",
    "mixed_devvoice_train_v1",
    "mixed_fmc_music_train_v1",
    "mixfake_music_train_v1",
    "telephone_mixed_train_v1",
    "temporal_mixed_train_v2",
    "channel_invariant_factorial_train_v1",
)
DEV_DEFAULT = (
    "mixfake_music_dev_v1",
    "external_mixed_v1",
    "source_disjoint_mixed_v1",
    "source_disjoint_mixed_equal_v1",
    "source_disjoint_music_v1",
    "factorial_eval_1200_v2_dev",
    "telephone_mixed_dev_v1",
)


class RandomSslProjection(nn.Module):
    """Label-free dimensionality reduction of frozen SSL coordinates."""

    def __init__(self, width: int, seed: int, device: torch.device) -> None:
        super().__init__()
        generator = torch.Generator(device="cpu").manual_seed(seed)
        self.register_buffer(
            "eat_matrix",
            torch.randn(768, width, generator=generator) / np.sqrt(768),
        )
        self.register_buffer(
            "spear_matrix",
            torch.randn(1280, width, generator=generator) / np.sqrt(1280),
        )
        self.register_buffer("stat_embedding", torch.zeros(4, width))
        self.register_buffer("spear_layer_embedding", torch.zeros(13, width))
        self.register_buffer("stream_embedding", torch.zeros(2, width))
        self.to(device).eval()

    def eat_projection(self, values: torch.Tensor) -> torch.Tensor:
        values = F.layer_norm(values, (values.shape[-1],))
        return values @ self.eat_matrix

    def spear_projection(self, values: torch.Tensor) -> torch.Tensor:
        values = F.layer_norm(values, (values.shape[-1],))
        return values @ self.spear_matrix


def load_projection(checkpoint_path: Path, device: torch.device):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint.get("model_type") != "invariant":
        raise ValueError("projection checkpoint must contain an invariant head")
    model = InvariantDualDomainHead(**checkpoint["config"])
    model.load_state_dict(checkpoint["model"], strict=True)
    model = model.to(device).eval().requires_grad_(False)
    normalization = {
        name: torch.from_numpy(np.asarray(value, dtype=np.float32)).to(device)[None, None]
        for name, value in checkpoint["normalization"].items()
    }
    return model, normalization


def load_random_projection(
    device: torch.device, width: int, seed: int,
) -> tuple[RandomSslProjection, dict[str, torch.Tensor]]:
    model = RandomSslProjection(width, seed, device)
    # Random projection uses per-token LayerNorm.  Identity corpus statistics
    # keep a leave-generator-out audit free of target-generator exposure.
    normalization = {
        "eat_mean": torch.zeros(1, 1, 1, 768, device=device),
        "eat_std": torch.ones(1, 1, 1, 768, device=device),
        "spear_mean": torch.zeros(1, 1, 1, 1, 1280, device=device),
        "spear_std": torch.ones(1, 1, 1, 1, 1280, device=device),
    }
    return model, normalization


def selected_layers(specification: str) -> list[int]:
    if specification == "all":
        return list(range(13))
    if specification == "shallow":
        return list(range(7))
    if specification == "late":
        return list(range(6, 13))
    layers = sorted({int(value) for value in specification.split(",")})
    if not layers or layers[0] < 0 or layers[-1] >= 13:
        raise ValueError("SPEAR layers must be in [0, 12]")
    return layers


@torch.inference_mode()
def projected_features(
    bank: Bank,
    model: InvariantDualDomainHead,
    normalization: dict[str, torch.Tensor],
    device: torch.device,
    layers: list[int],
    feature_mode: str,
    batch_size: int,
) -> np.ndarray:
    batches = []
    for offset in range(0, len(bank.ids), batch_size):
        eat = torch.from_numpy(bank.eat[offset:offset + batch_size]).to(
            device=device, dtype=torch.float32
        )
        spear = torch.from_numpy(bank.spear[offset:offset + batch_size]).to(
            device=device, dtype=torch.float32
        )
        eat = ((eat - normalization["eat_mean"]) /
               normalization["eat_std"]).clamp_(-8, 8)
        spear = ((spear - normalization["spear_mean"]) /
                 normalization["spear_std"]).clamp_(-8, 8)
        eat = model.eat_projection(eat)
        spear = model.spear_projection(spear[:, :, layers])
        # Keep block identity, but omit learned view embeddings: dispersion
        # should describe the audio rather than fixed start/middle/end IDs.
        eat = (
            eat + model.stat_embedding[None, None, :, :]
            + model.stream_embedding[0]
        )
        spear = (
            spear
            + model.spear_layer_embedding[layers][None, None, :, None, :]
            + model.stat_embedding[None, None, None, :, :]
            + model.stream_embedding[1]
        )
        values = torch.cat((
            eat.flatten(2, 3),
            spear.flatten(2, 4),
        ), dim=2)
        mask = torch.from_numpy(
            bank.eat_mask[offset:offset + batch_size]
            & bank.spear_mask[offset:offset + batch_size]
        ).to(device)
        count = mask.sum(dim=1, keepdim=True).clamp_min(1).to(values.dtype)
        weights = mask[:, :, None].to(values.dtype)
        mean = (values * weights).sum(dim=1) / count
        second = (values.square() * weights).sum(dim=1) / count
        dispersion = (second - mean.square()).clamp_min(0).sqrt()
        if feature_mode == "mean":
            features = mean
        elif feature_mode == "dispersion":
            features = dispersion
        else:
            features = torch.cat((mean, dispersion), dim=1)
        # View count is a nuisance audit variable as well as a useful missing
        # value indicator.  It gets one bounded scalar, not a duration vector.
        features = torch.cat((features, count / 3.0), dim=1)
        batches.append(features.float().cpu().numpy())
    return np.concatenate(batches).astype(np.float32, copy=False)


def generator_group(row: pd.Series) -> str:
    if int(row.MUSIC_FAKE) == 1:
        for column in ("MUSIC_GENERATOR", "GENERATOR"):
            value = row.get(column)
            if pd.notna(value) and str(value).strip():
                return "fake:" + str(value).lower()
        return "fake:" + str(row.get("MUSIC_SOURCE_BANK", row.get("SOURCE", "unknown")))
    for column in ("MUSIC_SOURCE_BANK", "SOURCE"):
        value = row.get(column)
        if pd.notna(value) and str(value).strip():
            return "real:" + str(value).lower()
    return "real:unknown"


def normalized_generator(row: pd.Series) -> str:
    for column in ("MUSIC_GENERATOR", "GENERATOR"):
        value = row.get(column)
        if pd.notna(value) and str(value).strip():
            return str(value).strip().lower()
    return ""


def sample_weights(frame: pd.DataFrame) -> np.ndarray:
    keys = pd.Series([
        f"{row.DATASET}|{generator_group(row)}"
        for _, row in frame.iterrows()
    ])
    counts = keys.value_counts()
    weights = np.asarray([1.0 / counts[key] for key in keys], dtype=np.float32)
    # Equal total mass per dataset, label, and available generator/source group.
    dataset_counts = frame.DATASET.value_counts()
    weights *= np.asarray([1.0 / dataset_counts[name] for name in frame.DATASET],
                          dtype=np.float32)
    labels = frame.MUSIC_FAKE.astype(int).to_numpy()
    totals = np.bincount(labels, weights=weights, minlength=2)
    weights /= totals[labels].clip(1e-12)
    return weights / weights.mean()


def component_partners(frame: pd.DataFrame) -> np.ndarray:
    """Pair the same music source across changed voice/mix conditions."""
    groups: dict[tuple[str, str, int], dict[str, list[int]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for index, row in frame.iterrows():
        source = row.get("MUSIC_SOURCE_ID")
        if pd.isna(source) or not str(source).strip():
            continue
        nuisance = f"{row.get('VOICE_FAKE')}|{row.get('MIX_MODE', row.get('CONDITION'))}"
        groups[(str(row.DATASET), str(source), int(row.MUSIC_FAKE))][nuisance].append(index)
    pairs = []
    for nuisances in groups.values():
        keys = sorted(nuisances)
        if len(keys) < 2:
            continue
        for position, key in enumerate(keys):
            other = keys[(position + 1) % len(keys)]
            for offset, left in enumerate(nuisances[key]):
                pairs.append((left, nuisances[other][offset % len(nuisances[other])]))
    return np.asarray(pairs, dtype=np.int64).reshape(-1, 2)


def generator_environments(frame: pd.DataFrame) -> list[tuple[str, np.ndarray, np.ndarray]]:
    """Build balanced real-vs-one-generator environments per source bank."""
    labels = frame.MUSIC_FAKE.astype(int).to_numpy()
    environments = []
    for dataset, block in frame.groupby("DATASET"):
        negatives = block.index[block.MUSIC_FAKE.eq(0)].to_numpy(np.int64)
        if not len(negatives):
            continue
        fake = block[block.MUSIC_FAKE.eq(1)].copy()
        fake["_GENERATOR"] = fake.apply(generator_group, axis=1)
        for generator, group in fake.groupby("_GENERATOR"):
            positives = group.index.to_numpy(np.int64)
            if len(positives):
                environments.append((
                    f"{dataset}|{generator}", positives, negatives,
                ))
    if not environments or labels.min() == labels.max():
        raise ValueError("generator environments require both music labels")
    return environments


def music_eer(frame: pd.DataFrame, scores: np.ndarray) -> float:
    selected = frame.MUSIC_PRESENT.eq(1) & frame.MUSIC_FAKE.notna()
    labels = frame.loc[selected, "MUSIC_FAKE"].astype(int)
    if labels.nunique() < 2:
        return float("nan")
    return float(official_eer(labels, scores[selected.to_numpy()]))


def evaluate(
    frames: list[pd.DataFrame], scores: list[np.ndarray]
) -> tuple[pd.DataFrame, float]:
    rows = []
    for frame, probability in zip(frames, scores):
        rows.append({
            "DATASET": frame.DATASET.iloc[0],
            "N": len(frame),
            "MUSIC_EER": music_eer(frame, probability),
        })
    result = pd.DataFrame(rows)
    quality = 1 - result.MUSIC_EER
    selection = 0.5 * quality.mean() + 0.5 * quality.min()
    return result, float(selection)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stats-root", type=Path,
                        default=ROOT / "output/dual_domain_stats_v1")
    parser.add_argument("--projection-checkpoint", type=Path)
    parser.add_argument("--projection-kind", choices=("invariant", "random"),
                        default="invariant")
    parser.add_argument("--projection-width", type=int, default=128)
    parser.add_argument("--projection-seed", type=int, default=20260903)
    parser.add_argument("--train-datasets", nargs="+", default=list(TRAIN_DEFAULT))
    parser.add_argument("--dev-datasets", nargs="+", default=list(DEV_DEFAULT))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--layers", default="all")
    parser.add_argument("--feature-mode", choices=("mean", "dispersion", "both"),
                        default="both")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=600)
    parser.add_argument("--eval-every", type=int, default=5)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--learning-rate", type=float, default=3e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--consistency-weight", type=float, default=0.05)
    parser.add_argument(
        "--group-dro-temperature", type=float, default=0.0,
        help=(
            "Positive values replace average BCE with a smooth worst-generator "
            "objective; smaller values focus more strongly on the worst group."
        ),
    )
    parser.add_argument(
        "--exclude-music-generators", nargs="*", default=[],
        help="Remove named fake generators from training for leave-generator-out audits.",
    )
    parser.add_argument(
        "--select-last", action="store_true",
        help="Use the final epoch without dev-based checkpoint selection.",
    )
    parser.add_argument("--seed", type=int, default=20260903)
    args = parser.parse_args()
    if args.projection_kind == "invariant" and args.projection_checkpoint is None:
        parser.error("--projection-checkpoint is required for invariant projection")
    if args.group_dro_temperature < 0:
        parser.error("--group-dro-temperature must be non-negative")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    for name in args.train_datasets:
        assert_no_locked_eval_leakage(
            truth_path(name), ROOT / "configs/data_partitions.yaml"
        )

    layers = selected_layers(args.layers)
    device = torch.device(args.device)
    if args.projection_kind == "invariant":
        projection, projection_norm = load_projection(
            args.projection_checkpoint, device
        )
    else:
        projection, projection_norm = load_random_projection(
            device, args.projection_width, args.projection_seed
        )
    train_banks = [load_bank(args.stats_root, name, "clean")
                   for name in args.train_datasets]
    dev_banks = [load_bank(args.stats_root, name, "clean")
                 for name in args.dev_datasets]

    feature_blocks, frames = [], []
    for bank in [*train_banks, *dev_banks]:
        feature_blocks.append(projected_features(
            bank, projection, projection_norm, device, layers,
            args.feature_mode, args.batch_size,
        ))
        frame = bank.truth.copy()
        frame["DATASET"] = bank.name
        frames.append(frame)
        print(f"features {bank.name}: {feature_blocks[-1].shape}", flush=True)
    del projection, train_banks, dev_banks
    torch.cuda.empty_cache()

    train_count = len(args.train_datasets)
    train_frame = pd.concat(frames[:train_count], ignore_index=True)
    train_features = np.concatenate(feature_blocks[:train_count])
    music = train_frame.MUSIC_PRESENT.eq(1) & train_frame.MUSIC_FAKE.notna()
    excluded = {value.strip().lower() for value in args.exclude_music_generators}
    if excluded:
        generator = train_frame.apply(normalized_generator, axis=1)
        music &= ~(
            train_frame.MUSIC_FAKE.eq(1)
            & generator.isin(excluded)
        )
    train_frame = train_frame.loc[music].reset_index(drop=True)
    train_features = train_features[music.to_numpy()]
    dev_frames = frames[train_count:]
    dev_features = feature_blocks[train_count:]

    mean = train_features.mean(axis=0, dtype=np.float64).astype(np.float32)
    std = train_features.std(axis=0, dtype=np.float64).clip(1e-4).astype(np.float32)
    train_features = np.clip((train_features - mean) / std, -8, 8)
    dev_features = [np.clip((values - mean) / std, -8, 8)
                    for values in dev_features]
    weights = sample_weights(train_frame)
    pairs = component_partners(train_frame)
    environments = generator_environments(train_frame)
    print(json.dumps({
        "train": len(train_frame), "features": train_features.shape[1],
        "pairs": len(pairs), "environments": len(environments),
        "layers": layers,
    }), flush=True)

    x = torch.from_numpy(train_features).to(device)
    y = torch.from_numpy(
        train_frame.MUSIC_FAKE.astype(np.float32).to_numpy(copy=True)
    ).to(device)
    weight = torch.from_numpy(weights).to(device)
    pair_index = torch.from_numpy(pairs).to(device)
    environment_indices = [
        (name, torch.from_numpy(positive).to(device),
         torch.from_numpy(negative).to(device))
        for name, positive, negative in environments
    ]
    dev_x = [torch.from_numpy(values).to(device) for values in dev_features]
    classifier = nn.Linear(train_features.shape[1], 1).to(device)
    optimizer = torch.optim.AdamW(
        classifier.parameters(), lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    best_selection, best_epoch, best_state, stale = -float("inf"), -1, None, 0
    history = []
    for epoch in range(args.epochs + 1):
        classifier.train()
        logits = classifier(x).squeeze(1)
        point_loss = F.binary_cross_entropy_with_logits(
            logits, y, reduction="none"
        )
        if args.group_dro_temperature > 0:
            environment_loss = torch.stack([
                0.5 * point_loss[positive].mean()
                + 0.5 * point_loss[negative].mean()
                for _, positive, negative in environment_indices
            ])
            temperature = args.group_dro_temperature
            base = temperature * (
                torch.logsumexp(environment_loss / temperature, dim=0)
                - np.log(len(environment_indices))
            )
        else:
            base = (point_loss * weight).mean()
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
            dev_scores = [classifier(values).squeeze(1).sigmoid().cpu().numpy()
                          for values in dev_x]
        result, selection = evaluate(dev_frames, dev_scores)
        history.append({
            "EPOCH": epoch, "LOSS": float(loss.detach()),
            "BASE_LOSS": float(base.detach()),
            "CONSISTENCY_LOSS": float(consistency.detach()),
            "SELECTION": selection, "MEAN_MUSIC_EER": result.MUSIC_EER.mean(),
            "WORST_MUSIC_EER": result.MUSIC_EER.max(),
        })
        print(
            f"epoch={epoch:03d} loss={float(loss.detach()):.5f} "
            f"selection={selection:.5f} mean_eer={result.MUSIC_EER.mean():.5f} "
            f"worst_eer={result.MUSIC_EER.max():.5f}", flush=True,
        )
        if selection > best_selection + 1e-5:
            best_selection, best_epoch = selection, epoch
            best_state = copy.deepcopy(classifier.state_dict())
            stale = 0
        else:
            stale += 1
            if stale >= args.patience:
                break
        if args.select_last:
            best_selection, best_epoch = selection, epoch
            best_state = copy.deepcopy(classifier.state_dict())
            stale = 0
    if best_state is None:
        raise RuntimeError("training produced no checkpoint")
    classifier.load_state_dict(best_state)
    classifier.eval()
    with torch.inference_mode():
        dev_scores = [classifier(values).squeeze(1).sigmoid().cpu().numpy()
                      for values in dev_x]
    result, selection = evaluate(dev_frames, dev_scores)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.save({
        "classifier": {name: value.cpu() for name, value in best_state.items()},
        "mean": mean, "std": std, "layers": layers,
        "feature_mode": args.feature_mode,
        "projection_checkpoint": (
            str(args.projection_checkpoint)
            if args.projection_checkpoint is not None else None
        ),
        "projection_kind": args.projection_kind,
        "projection_width": args.projection_width,
        "projection_seed": args.projection_seed,
        "train_datasets": args.train_datasets,
        "dev_datasets": args.dev_datasets,
        "excluded_music_generators": sorted(excluded),
        "select_last": args.select_last,
        "group_dro_temperature": args.group_dro_temperature,
        "seed": args.seed, "best_epoch": best_epoch,
        "selection": selection,
    }, args.output_dir / "long_horizon_music_head.pt")
    pd.DataFrame(history).to_csv(args.output_dir / "history.csv", index=False)
    result.to_csv(args.output_dir / "dev_metrics.csv", index=False)
    rows = []
    for frame, scores in zip(dev_frames, dev_scores):
        rows.append(pd.DataFrame({
            "DATASET": frame.DATASET.to_numpy(), "ID": frame.ID.to_numpy(),
            "LONG_HORIZON_MUSIC_PROB": scores,
        }))
    pd.concat(rows, ignore_index=True).to_csv(
        args.output_dir / "dev_predictions.csv", index=False
    )
    summary = {
        "best_epoch": best_epoch, "selection": selection,
        "mean_music_eer": float(result.MUSIC_EER.mean()),
        "worst_music_eer": float(result.MUSIC_EER.max()),
        "feature_mode": args.feature_mode, "layers": layers,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(result.to_string(index=False))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
