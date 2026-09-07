#!/usr/bin/env python3
"""Train a separation-free EAT/SPEAR component-query MHFA detector."""

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


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from component_query_mhfa import ComponentQueryMHFA, component_query_loss  # noqa: E402
from data_guard import assert_no_locked_eval_leakage  # noqa: E402
from evaluate_diagnostic import score_frame  # noqa: E402
from train_eat_patch_graph import load_cache, sample_weights  # noqa: E402
from train_spear_temporal_bin_mil import load_archive  # noqa: E402
from train_unified_dual_ssl_head import (  # noqa: E402
    DEV_DEFAULT, TRAIN_DEFAULT, align, spear_name, targets, truth_for,
)


KEYS = ("temporal", "spectral", "eat_mask", "spear", "spear_mask")


def load_block(
    eat_root: Path,
    spear_cache: dict[str, dict[str, np.ndarray]],
    name: str,
    role: str,
) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    frame = truth_for(name, role).copy()
    frame["DATASET"] = name
    eat = load_cache(eat_root, name, frame)
    spear = spear_cache[spear_name(name)]
    order = align(spear["ids"], frame)
    return frame, {
        "temporal": eat["temporal"], "spectral": eat["spectral"],
        "eat_mask": eat["view_mask"], "spear": spear["features"][order],
        "spear_mask": spear["mask"][order],
        "eat_projection": eat["projection"], "eat_layers": eat["layers"],
    }


def concatenate(blocks: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    return {key: np.concatenate([block[key] for block in blocks]) for key in KEYS}


def move_to_device(
    block: dict[str, np.ndarray], device: torch.device,
) -> dict[str, torch.Tensor]:
    return {key: torch.from_numpy(block[key]).to(device) for key in KEYS}


def batch(
    block: dict[str, torch.Tensor], indices: np.ndarray, device: torch.device,
) -> tuple[torch.Tensor, ...]:
    selected = torch.as_tensor(indices, dtype=torch.long, device=device)
    return tuple(block[key].index_select(0, selected) for key in KEYS)


def drop_views(
    eat_mask: torch.Tensor, spear_mask: torch.Tensor, probability: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    if probability <= 0:
        return eat_mask, spear_mask
    keep = eat_mask.bool() & (
        torch.rand(eat_mask.shape, device=eat_mask.device) >= probability
    )
    empty = ~keep.any(dim=1)
    if empty.any():
        first = eat_mask.float().argmax(dim=1)
        keep[empty, first[empty]] = True
    spear_keep = spear_mask.bool() & keep[:, :, None]
    return keep, spear_keep


@torch.inference_mode()
def predict(
    model: ComponentQueryMHFA,
    block: dict[str, torch.Tensor],
    device: torch.device,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    fake, presence = [], []
    for offset in range(0, len(block["temporal"]), batch_size):
        indices = np.arange(offset, min(offset + batch_size, len(block["temporal"])))
        fake_logits, presence_logits, _ = model(*batch(block, indices, device))
        fake.append(model.probabilities(fake_logits).cpu().numpy())
        presence.append(presence_logits.sigmoid().cpu().numpy())
    return np.concatenate(fake), np.concatenate(presence)


def evaluate(
    model: ComponentQueryMHFA,
    frames: tuple[pd.DataFrame, ...],
    blocks: tuple[dict[str, torch.Tensor], ...],
    device: torch.device,
    batch_size: int,
) -> tuple[pd.DataFrame, pd.DataFrame, float]:
    metrics, predictions, qualities = [], [], []
    for frame, block in zip(frames, blocks):
        fake, presence = predict(model, block, device, batch_size)
        prediction = pd.DataFrame({
            "FILE_FAKE_PROB": fake[:, 2],
            "VOICE_FAKE_PROB": fake[:, 0],
            "MUSIC_FAKE_PROB": fake[:, 1],
            "VOICE_PRESENT_PROB": presence[:, 0],
            "MUSIC_PRESENT_PROB": presence[:, 1],
        }, index=frame.ID)
        metric = score_frame(frame.set_index("ID").join(prediction))
        metrics.append({"DATASET": frame.DATASET.iloc[0], **metric})
        current = prediction.reset_index().rename(columns={"index": "ID"})
        current.insert(0, "DATASET", frame.DATASET.to_numpy())
        predictions.append(current)
        # Deployment initially changes File/Music only.  Match their official
        # 0.5:0.3 contribution instead of selecting on an unused Voice output.
        qualities.append(
            .625 * (1 - metric["FILE_EER"]) + .375 * (1 - metric["MUSIC_EER"])
        )
    values = np.asarray(qualities, dtype=np.float64)
    selection = .5 * values.mean() + .5 * values.min()
    return pd.DataFrame(metrics), pd.concat(predictions), float(selection)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--eat-cache-root", type=Path, default=ROOT / "output/eat_patch_graph_v1"
    )
    parser.add_argument(
        "--spear-cache-root", type=Path,
        default=ROOT / "reports/spear_temporal_attention_v1/cache",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--train-datasets", nargs="+", default=list(TRAIN_DEFAULT))
    parser.add_argument("--dev-datasets", nargs="+", default=list(DEV_DEFAULT))
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--width", type=int, default=96)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=.15)
    parser.add_argument("--file-component-weight", type=float, default=.20)
    parser.add_argument("--task-weights", type=float, nargs=3, default=[.20, .35, .45])
    parser.add_argument("--presence-weight", type=float, default=.05)
    parser.add_argument("--joint-weight", type=float, default=.15)
    parser.add_argument("--ranking-weight", type=float, default=.15)
    parser.add_argument("--view-dropout", type=float, default=.15)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=48)
    parser.add_argument("--eval-batch-size", type=int, default=96)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=2e-2)
    parser.add_argument("--eval-every", type=int, default=2)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260904)
    args = parser.parse_args()
    if not 0 <= args.view_dropout < 1:
        parser.error("view dropout must lie in [0,1)")
    for name in args.train_datasets:
        assert_no_locked_eval_leakage(
            ROOT / "data/eval" / name / "truth.csv",
            ROOT / "configs/data_partitions.yaml",
        )
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    spear_cache = load_archive(args.spear_cache_root)
    train_pairs = [
        load_block(args.eat_cache_root, spear_cache, name, "train")
        for name in args.train_datasets
    ]
    dev_pairs = [
        load_block(args.eat_cache_root, spear_cache, name, "dev")
        for name in args.dev_datasets
    ]
    metadata = train_pairs[0][1]
    spear_metadata = spear_cache["__metadata__"]
    for _, current in train_pairs[1:] + dev_pairs:
        if (
            not np.array_equal(current["eat_projection"], metadata["eat_projection"])
            or not np.array_equal(current["eat_layers"], metadata["eat_layers"])
        ):
            raise ValueError("EAT patch metadata differs across datasets")
    train_frame = pd.concat([item[0] for item in train_pairs], ignore_index=True)
    train_numpy = concatenate([item[1] for item in train_pairs])
    dev_frames, dev_numpy = zip(*dev_pairs)
    fake_numpy, presence_numpy = targets(train_frame)
    weight_numpy = sample_weights(train_frame)

    device = torch.device(args.device)
    train = move_to_device(train_numpy, device)
    dev = tuple(move_to_device(item, device) for item in dev_numpy)
    fake = torch.from_numpy(fake_numpy).to(device)
    presence = torch.from_numpy(presence_numpy).to(device)
    weights = torch.from_numpy(weight_numpy).to(device)
    shape = train["temporal"].shape
    spear_shape = train["spear"].shape
    model = ComponentQueryMHFA(
        eat_layers=shape[2], eat_dimension=shape[-1],
        spear_layers=spear_shape[3], spear_stats=spear_shape[4],
        spear_dimension=spear_shape[-1], width=args.width, heads=args.heads,
        depth=args.depth, maximum_views=shape[1],
        maximum_time_nodes=shape[-2],
        maximum_frequency_nodes=train["spectral"].shape[-2],
        maximum_bins=spear_shape[2], dropout=args.dropout,
        file_component_weight=args.file_component_weight,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.learning_rate * .05
    )
    generator = np.random.default_rng(args.seed)
    sampling = weight_numpy.astype(np.float64)
    sampling /= sampling.sum(dtype=np.float64)
    best_selection, best_epoch, best_state, stale = -np.inf, -1, None, 0
    history = []
    for epoch in range(args.epochs + 1):
        model.train()
        order = generator.choice(
            len(train_frame), size=len(train_frame), replace=True, p=sampling
        )
        losses = []
        for offset in range(0, len(order), args.batch_size):
            indices = order[offset:offset + args.batch_size]
            items = list(batch(train, indices, device))
            items[2], items[4] = drop_views(
                items[2], items[4], args.view_dropout
            )
            auth_logits, presence_logits, joint_logits = model(*items)
            selected = torch.as_tensor(indices, dtype=torch.long, device=device)
            loss, _ = component_query_loss(
                model, auth_logits, presence_logits, joint_logits,
                fake.index_select(0, selected), presence.index_select(0, selected),
                torch.ones(len(indices), device=device),
                task_weights=tuple(args.task_weights),
                presence_weight=args.presence_weight,
                joint_weight=args.joint_weight,
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
            model, dev_frames, dev, device, args.eval_batch_size
        )
        row = {
            "EPOCH": epoch, "LOSS": float(np.mean(losses)),
            "SELECTION": selection, "MEAN_ADS": float(metrics.ADS.mean()),
            "WORST_ADS": float(metrics.ADS.min()),
            "MEAN_FILE_EER": float(metrics.FILE_EER.mean()),
            "MEAN_MUSIC_EER": float(metrics.MUSIC_EER.mean()),
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
        model, dev_frames, dev, device, args.eval_batch_size
    )
    config = {
        "eat_layers": shape[2], "eat_dimension": shape[-1],
        "spear_layers": spear_shape[3], "spear_stats": spear_shape[4],
        "spear_dimension": spear_shape[-1], "width": args.width,
        "heads": args.heads, "depth": args.depth,
        "maximum_views": shape[1], "maximum_time_nodes": shape[-2],
        "maximum_frequency_nodes": train["spectral"].shape[-2],
        "maximum_bins": spear_shape[2], "dropout": args.dropout,
        "file_component_weight": args.file_component_weight,
    }
    checkpoint = {
        "model_type": "component_query_mhfa_v1",
        "model": {key: value.cpu() for key, value in best_state.items()},
        "config": config,
        "eat_projection": metadata["eat_projection"],
        "eat_layers": metadata["eat_layers"],
        "spear_projection": spear_metadata["projection"],
        "spear_layers": spear_metadata["layers"],
        "spear_bins": spear_metadata["bins"],
        "train_datasets": args.train_datasets, "dev_datasets": args.dev_datasets,
        "task_weights": list(args.task_weights), "seed": args.seed,
        "best_epoch": best_epoch, "selection": selection,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, args.output_dir / "component_query_mhfa.pt")
    pd.DataFrame(history).to_csv(args.output_dir / "history.csv", index=False)
    metrics.to_csv(args.output_dir / "dev_metrics.csv", index=False)
    predictions.to_csv(args.output_dir / "dev_predictions.csv", index=False)
    summary = {
        "best_epoch": best_epoch, "selection": selection,
        "mean_ads": float(metrics.ADS.mean()),
        "worst_ads": float(metrics.ADS.min()),
        "parameters": sum(value.numel() for value in model.parameters()),
        "train_examples": len(train_frame),
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(metrics.to_string(index=False))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
