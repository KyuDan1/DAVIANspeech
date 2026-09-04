#!/usr/bin/env python3
"""Train the separation-free patch-preserving EAT graph detector."""

from __future__ import annotations

import argparse
import copy
import json
import random
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from data_guard import assert_no_locked_eval_leakage  # noqa: E402
from eat_patch_graph import EatPatchGraphHead, patch_graph_loss  # noqa: E402
from evaluate_diagnostic import score_frame  # noqa: E402
from train_unified_dual_ssl_head import (  # noqa: E402
    DEV_DEFAULT, TRAIN_DEFAULT, align, targets, truth_for,
)


ARRAY_KEYS = ("temporal", "spectral", "view_mask")


def load_cache(
    root: Path, name: str, frame: pd.DataFrame,
) -> dict[str, np.ndarray]:
    paths = sorted((root / name).glob("shard_*.npz"))
    if not paths:
        single = root / name / "features.npz"
        paths = [single] if single.is_file() else []
    if not paths:
        raise FileNotFoundError(f"no patch-graph cache for {name}")
    ids, arrays = [], {key: [] for key in ARRAY_KEYS}
    projection = layers = None
    for path in paths:
        with np.load(path, allow_pickle=False) as archive:
            ids.append(archive["ids"].astype(str))
            for key in ARRAY_KEYS:
                arrays[key].append(archive[key])
            current_projection = archive["projection"].astype(np.float32)
            current_layers = archive["layers"].astype(np.int64)
            if projection is None:
                projection, layers = current_projection, current_layers
            elif (
                not np.array_equal(projection, current_projection)
                or not np.array_equal(layers, current_layers)
            ):
                raise ValueError(f"patch-graph metadata differs in {name}")
    ids = np.concatenate(ids)
    if len(set(ids)) != len(ids):
        raise ValueError(f"duplicate patch-graph IDs in {name}")
    order = align(ids, frame)
    return {
        **{
            key: np.concatenate(values)[order]
            for key, values in arrays.items()
        },
        "projection": projection,
        "layers": layers,
    }


def load_block(
    root: Path, name: str, role: str,
) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    frame = truth_for(name, role)
    frame = frame.copy()
    frame["DATASET"] = name
    return frame, load_cache(root, name, frame)


def concatenate(blocks: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    return {
        key: np.concatenate([block[key] for block in blocks])
        for key in ARRAY_KEYS
    }


def generator_name(row: pd.Series, component: str) -> str:
    def first_group(columns: tuple[str, ...]) -> str:
        for column in columns:
            value = row.get(column)
            if value is None or pd.isna(value):
                continue
            text = str(value).strip().lower()
            if text and text not in {"0", "nan", "none"}:
                return text
        return "unknown"

    if not int(row.get(f"{component}_PRESENT", 0)):
        return "absent"
    if int(row.get(f"{component}_FAKE", 0) or 0):
        return "fake:" + first_group((
            f"{component}_GENERATOR", "GENERATOR", "SOURCE_BANK", "SOURCE",
        ))
    return "real:" + first_group(
        (f"{component}_SOURCE_BANK", "SOURCE_BANK", "SOURCE")
    )


def sample_weights(frame: pd.DataFrame) -> np.ndarray:
    """Balance dataset, component cell, and available generator identity."""
    keys = []
    for _, row in frame.iterrows():
        keys.append((
            str(row.DATASET), int(row.FILE_FAKE),
            int(row.VOICE_PRESENT), int(row.MUSIC_PRESENT),
            int(row.VOICE_FAKE if pd.notna(row.VOICE_FAKE) else 0),
            int(row.MUSIC_FAKE if pd.notna(row.MUSIC_FAKE) else 0),
            generator_name(row, "VOICE"), generator_name(row, "MUSIC"),
        ))
    counts = Counter(keys)
    weights = np.asarray([1 / counts[key] for key in keys], dtype=np.float32)
    datasets = frame.DATASET.astype(str).to_numpy()
    # Give every corpus the same total mass, then balance generator/cell groups
    # inside it.  Otherwise a corpus with richer metadata creates more groups
    # and can dominate merely because it has more annotation categories.
    for dataset in np.unique(datasets):
        selected = datasets == dataset
        weights[selected] /= weights[selected].sum()
    return weights / weights.mean()


def move_to_device(
    block: dict[str, np.ndarray], device: torch.device,
) -> dict[str, torch.Tensor]:
    return {
        key: torch.from_numpy(block[key]).to(device)
        for key in ARRAY_KEYS
    }


def tensor_batch(
    block: dict[str, torch.Tensor], indices: np.ndarray, device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    index = torch.as_tensor(indices, dtype=torch.long, device=device)
    return tuple(block[key].index_select(0, index) for key in ARRAY_KEYS)


def drop_views(mask: torch.Tensor, probability: float) -> torch.Tensor:
    if probability <= 0:
        return mask
    keep = mask.bool() & (torch.rand(mask.shape, device=mask.device) >= probability)
    empty = ~keep.any(dim=1)
    if empty.any():
        first = mask.float().argmax(dim=1)
        keep[empty, first[empty]] = True
    return keep


def domain_randomize(
    temporal: torch.Tensor,
    spectral: torch.Tensor,
    probability: float,
    scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Randomize per-example feature moments without changing node ordering."""
    if probability <= 0 or not torch.is_grad_enabled():
        return temporal, spectral
    selected = torch.rand(len(temporal), device=temporal.device) < probability
    if not selected.any():
        return temporal, spectral
    temporal = temporal.float()
    spectral = spectral.float()
    shift = torch.randn(
        len(temporal), 1, 1, 1, 1, temporal.shape[-1],
        device=temporal.device,
    ) * scale
    gain = torch.exp(torch.randn_like(shift) * (.5 * scale))
    gate = selected[:, None, None, None, None, None]
    return (
        torch.where(gate, temporal * gain + shift, temporal),
        torch.where(gate, spectral * gain + shift, spectral),
    )


@torch.inference_mode()
def predict(
    model: EatPatchGraphHead,
    block: dict[str, torch.Tensor],
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    model.eval()
    outputs = []
    for offset in range(0, len(block["temporal"]), batch_size):
        indices = np.arange(
            offset, min(offset + batch_size, len(block["temporal"]))
        )
        logits = model(*tensor_batch(block, indices, device))
        outputs.append(model.probabilities(logits).cpu().numpy())
    return np.concatenate(outputs)


def evaluate(
    model: EatPatchGraphHead,
    frames: tuple[pd.DataFrame, ...],
    blocks: tuple[dict[str, torch.Tensor], ...],
    device: torch.device,
    batch_size: int,
    selection_mode: str = "ads",
) -> tuple[pd.DataFrame, pd.DataFrame, float]:
    metrics, predictions, ads = [], [], []
    for frame, block in zip(frames, blocks):
        probability = predict(model, block, device, batch_size)
        prediction = pd.DataFrame({
            "VOICE_FAKE_PROB": probability[:, 0],
            "MUSIC_FAKE_PROB": probability[:, 1],
            "FILE_FAKE_PROB": probability[:, 2],
        }, index=frame.ID)
        metric = score_frame(frame.set_index("ID").join(prediction))
        metrics.append({"DATASET": frame.DATASET.iloc[0], **metric})
        current = prediction.reset_index().rename(columns={"index": "ID"})
        current.insert(0, "DATASET", frame.DATASET.to_numpy())
        predictions.append(current)
        if selection_mode == "ads":
            quality = metric["ADS"]
        elif selection_mode == "file_music":
            quality = .5 * (1 - metric["FILE_EER"]) + .5 * (
                1 - metric["MUSIC_EER"]
            )
        else:
            raise ValueError(f"unknown selection mode: {selection_mode}")
        ads.append(quality)
    values = np.asarray(ads, dtype=np.float64)
    finite = values[np.isfinite(values)]
    selection = .5 * finite.mean() + .5 * finite.min()
    return pd.DataFrame(metrics), pd.concat(predictions), float(selection)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cache-root", type=Path, default=ROOT / "output/eat_patch_graph_v1"
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--train-datasets", nargs="+", default=list(TRAIN_DEFAULT))
    parser.add_argument("--dev-datasets", nargs="+", default=list(DEV_DEFAULT))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--width", type=int, default=96)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=.15)
    parser.add_argument("--temperature", type=float, default=5.)
    parser.add_argument("--file-component-weight", type=float, default=.10)
    parser.add_argument("--ranking-weight", type=float, default=.15)
    parser.add_argument(
        "--task-weights", type=float, nargs=3, default=[.20, .35, .45],
        metavar=("VOICE", "MUSIC", "FILE"),
    )
    parser.add_argument(
        "--balanced-sampling", action="store_true",
        help="Sample corpus/generator/cell groups uniformly with replacement.",
    )
    parser.add_argument(
        "--selection-mode", choices=("ads", "file_music"), default="ads",
    )
    parser.add_argument("--view-dropout", type=float, default=.15)
    parser.add_argument("--domain-randomization", type=float, default=.35)
    parser.add_argument("--domain-scale", type=float, default=.05)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--eval-batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=2e-2)
    parser.add_argument("--eval-every", type=int, default=2)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260904)
    args = parser.parse_args()
    if not 0 <= args.view_dropout < 1:
        parser.error("view dropout must lie in [0, 1)")
    if not 0 <= args.domain_randomization <= 1 or args.domain_scale < 0:
        parser.error("invalid domain randomization")
    for name in args.train_datasets:
        assert_no_locked_eval_leakage(
            ROOT / "data/eval" / name / "truth.csv",
            ROOT / "configs/data_partitions.yaml",
        )
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    train_pairs = [
        load_block(args.cache_root, name, "train")
        for name in args.train_datasets
    ]
    dev_pairs = [
        load_block(args.cache_root, name, "dev") for name in args.dev_datasets
    ]
    metadata = train_pairs[0][1]
    for _, block in train_pairs[1:] + dev_pairs:
        if (
            not np.array_equal(block["projection"], metadata["projection"])
            or not np.array_equal(block["layers"], metadata["layers"])
        ):
            raise ValueError("patch-graph metadata differs across datasets")
    train_frame = pd.concat([pair[0] for pair in train_pairs], ignore_index=True)
    train_numpy = concatenate([pair[1] for pair in train_pairs])
    dev_frames, dev_numpy = zip(*dev_pairs)
    fake_numpy, presence_numpy = targets(train_frame)
    weight_numpy = sample_weights(train_frame)

    device = torch.device(args.device)
    train = move_to_device(train_numpy, device)
    dev = tuple(move_to_device(block, device) for block in dev_numpy)
    fake = torch.from_numpy(fake_numpy.copy()).to(device)
    presence = torch.from_numpy(presence_numpy.copy()).to(device)
    sample_weight = torch.from_numpy(weight_numpy).to(device)
    temporal_shape = train["temporal"].shape
    spectral_shape = train["spectral"].shape
    model = EatPatchGraphHead(
        layers=temporal_shape[2], dimension=temporal_shape[-1],
        width=args.width, heads=args.heads, depth=args.depth,
        maximum_views=temporal_shape[1],
        maximum_time_nodes=temporal_shape[-2],
        maximum_frequency_nodes=spectral_shape[-2],
        dropout=args.dropout, temperature=args.temperature,
        file_component_weight=args.file_component_weight,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.learning_rate * .05
    )
    generator = np.random.default_rng(args.seed)
    sampling_probability = weight_numpy.astype(np.float64)
    sampling_probability /= sampling_probability.sum(dtype=np.float64)
    best_selection, best_epoch, best_state, stale = -np.inf, -1, None, 0
    history = []
    for epoch in range(args.epochs + 1):
        model.train()
        if args.balanced_sampling:
            order = generator.choice(
                len(train_frame), size=len(train_frame), replace=True,
                p=sampling_probability,
            )
        else:
            order = generator.permutation(len(train_frame))
        losses = []
        for offset in range(0, len(order), args.batch_size):
            indices = order[offset:offset + args.batch_size]
            temporal, spectral, mask = tensor_batch(train, indices, device)
            temporal, spectral = domain_randomize(
                temporal, spectral, args.domain_randomization, args.domain_scale
            )
            mask = drop_views(mask, args.view_dropout)
            logits = model(temporal, spectral, mask)
            index = torch.as_tensor(indices, dtype=torch.long, device=device)
            batch_weight = (
                torch.ones(len(indices), device=device)
                if args.balanced_sampling
                else sample_weight.index_select(0, index)
            )
            loss, _ = patch_graph_loss(
                model, logits, fake.index_select(0, index),
                presence.index_select(0, index),
                batch_weight, task_weights=tuple(args.task_weights),
                ranking_weight=args.ranking_weight,
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
            optimizer.step()
            losses.append(float(loss.detach()))
        scheduler.step()
        if epoch % args.eval_every:
            continue
        metrics, _, selection = evaluate(
            model, dev_frames, dev, device, args.eval_batch_size,
            args.selection_mode,
        )
        row = {
            "EPOCH": epoch, "LOSS": float(np.mean(losses)),
            "SELECTION": selection, "MEAN_ADS": float(metrics.ADS.mean()),
            "WORST_ADS": float(metrics.ADS.min()),
        }
        history.append(row)
        print(json.dumps(row), flush=True)
        if selection > best_selection + 1e-5:
            best_selection, best_epoch = selection, epoch
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
            if stale >= args.patience:
                break
    if best_state is None:
        raise RuntimeError("training produced no checkpoint")
    model.load_state_dict(best_state)
    metrics, predictions, selection = evaluate(
        model, dev_frames, dev, device, args.eval_batch_size,
        args.selection_mode,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "model_type": "eat_patch_graph_multitask",
        "model": {key: value.cpu() for key, value in best_state.items()},
        "config": {
            "layers": temporal_shape[2], "dimension": temporal_shape[-1],
            "width": args.width, "heads": args.heads, "depth": args.depth,
            "maximum_views": temporal_shape[1],
            "maximum_time_nodes": temporal_shape[-2],
            "maximum_frequency_nodes": spectral_shape[-2],
            "dropout": args.dropout, "temperature": args.temperature,
            "file_component_weight": args.file_component_weight,
        },
        "projection": metadata["projection"],
        "eat_layers": metadata["layers"],
        "train_datasets": args.train_datasets,
        "dev_datasets": args.dev_datasets,
        "best_epoch": best_epoch, "selection": selection, "seed": args.seed,
        "task_weights": list(args.task_weights),
        "balanced_sampling": bool(args.balanced_sampling),
        "selection_mode": args.selection_mode,
    }
    torch.save(checkpoint, args.output_dir / "eat_patch_graph_head.pt")
    pd.DataFrame(history).to_csv(args.output_dir / "history.csv", index=False)
    metrics.to_csv(args.output_dir / "dev_metrics.csv", index=False)
    predictions.to_csv(args.output_dir / "dev_predictions.csv", index=False)
    summary = {
        "best_epoch": best_epoch, "selection": selection,
        "mean_ads": float(metrics.ADS.mean()),
        "worst_ads": float(metrics.ADS.min()),
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "train_examples": len(train_frame),
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(metrics.to_string(index=False))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
