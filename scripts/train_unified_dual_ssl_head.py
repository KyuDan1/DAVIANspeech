#!/usr/bin/env python3
"""Train a separation-free EAT/SPEAR token-fusion component detector."""

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
from evaluate_diagnostic import score_frame  # noqa: E402
from train_spear_temporal_bin_mil import load_archive  # noqa: E402
from unified_dual_ssl_head import (  # noqa: E402
    UnifiedDualSSLHead,
    unified_multitask_loss,
)


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


def truth_for(name: str, role: str) -> pd.DataFrame:
    if name == "factorial_eval_1200_v2_dev":
        path = ROOT / "data/eval/factorial_eval_1200_v2/truth_dev.csv"
    elif name == "factorial_eval_1200_v2_holdout":
        path = ROOT / "data/eval/factorial_eval_1200_v2/truth_holdout.csv"
    else:
        path = ROOT / "data/eval" / name / "truth.csv"
    frame = pd.read_csv(path, dtype={"ID": str})
    if name == "temporal_mixed_train_v2" and "SPLIT" in frame:
        frame = frame[frame.SPLIT.eq("train" if role == "train" else "dev")]
    return frame.reset_index(drop=True)


def spear_name(name: str) -> str:
    if name.startswith("factorial_eval_1200_v2_"):
        return "factorial_eval_1200_v2"
    return name


def align(ids: np.ndarray, frame: pd.DataFrame) -> np.ndarray:
    index = {str(item): offset for offset, item in enumerate(ids.astype(str))}
    missing = set(frame.ID).difference(index)
    if missing:
        raise FileNotFoundError(
            f"{len(missing)} IDs are absent from cache: {sorted(missing)[:5]}"
        )
    return np.asarray([index[item] for item in frame.ID], dtype=np.int64)


def load_eat(root: Path, name: str, frame: pd.DataFrame) -> dict[str, np.ndarray]:
    path = root / name / "shard_0.npz"
    if not path.is_file():
        raise FileNotFoundError(path)
    archive = np.load(path, allow_pickle=False)
    order = align(archive["ids"], frame)
    return {
        "eat": archive["statistics"][order],
        "eat_mask": archive["view_mask"][order],
        "eat_projection": archive["projection"],
    }


def load_block(
    eat_root: Path,
    spear_cache: dict[str, dict[str, np.ndarray]],
    name: str,
    role: str,
) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    frame = truth_for(name, role)
    eat = load_eat(eat_root, name, frame)
    spear = spear_cache[spear_name(name)]
    order = align(spear["ids"], frame)
    frame = frame.copy()
    frame["DATASET"] = name
    return frame, {
        **eat,
        "spear": spear["features"][order],
        "spear_mask": spear["mask"][order],
    }


def concatenate(blocks: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    keys = ("eat", "eat_mask", "spear", "spear_mask")
    return {key: np.concatenate([block[key] for block in blocks]) for key in keys}


def sample_weights(frame: pd.DataFrame) -> np.ndarray:
    keys = [
        (
            str(row.DATASET), int(row.FILE_FAKE), int(row.VOICE_PRESENT),
            int(row.MUSIC_PRESENT), int(row.VOICE_FAKE or 0),
            int(row.MUSIC_FAKE or 0),
        )
        for row in frame.fillna(0).itertuples(index=False)
    ]
    counts = Counter(keys)
    values = np.asarray([1 / counts[key] for key in keys], dtype=np.float32)
    return values / values.mean()


def targets(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    fake = np.stack((
        frame.VOICE_FAKE.fillna(0).to_numpy(np.float32),
        frame.MUSIC_FAKE.fillna(0).to_numpy(np.float32),
        frame.FILE_FAKE.to_numpy(np.float32),
    ), axis=1)
    presence = frame[["VOICE_PRESENT", "MUSIC_PRESENT"]].to_numpy(np.float32)
    return fake, presence


def tensor_batch(block: dict[str, np.ndarray], indices: np.ndarray, device):
    if torch.is_tensor(block["eat"]):
        index = torch.as_tensor(indices, dtype=torch.long, device=device)
        return tuple(
            block[key].index_select(0, index)
            for key in ("eat", "spear", "eat_mask", "spear_mask")
        )
    return (
        torch.from_numpy(block["eat"][indices]).to(device),
        torch.from_numpy(block["spear"][indices]).to(device),
        torch.from_numpy(block["eat_mask"][indices]).to(device),
        torch.from_numpy(block["spear_mask"][indices]).to(device),
    )


def move_block_to_device(
    block: dict[str, np.ndarray], device: torch.device
) -> dict[str, torch.Tensor]:
    """Preload compact caches once; repeated NumPy gathers dominate training."""
    return {
        key: torch.from_numpy(block[key]).to(device)
        for key in ("eat", "eat_mask", "spear", "spear_mask")
    }


@torch.inference_mode()
def predict(
    model: UnifiedDualSSLHead,
    block: dict[str, np.ndarray],
    device: torch.device,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    authenticity, presence = [], []
    for offset in range(0, len(block["eat"]), batch_size):
        indices = np.arange(offset, min(offset + batch_size, len(block["eat"])))
        task_logits, presence_logits, _ = model(*tensor_batch(block, indices, device))
        authenticity.append(model.probabilities(task_logits).cpu().numpy())
        presence.append(presence_logits.sigmoid().cpu().numpy())
    return np.concatenate(authenticity), np.concatenate(presence)


def evaluate(model, frames, blocks, device, batch_size):
    records, predictions, ads = [], [], []
    for frame, block in zip(frames, blocks):
        fake, presence = predict(model, block, device, batch_size)
        prediction = pd.DataFrame({
            "FILE_FAKE_PROB": fake[:, 2],
            "VOICE_FAKE_PROB": fake[:, 0],
            "MUSIC_FAKE_PROB": fake[:, 1],
            "VOICE_PRESENT_PROB": presence[:, 0],
            "MUSIC_PRESENT_PROB": presence[:, 1],
        }, index=frame.ID)
        metrics = score_frame(frame.set_index("ID").join(prediction))
        records.append({"DATASET": frame.DATASET.iloc[0], **metrics})
        current = prediction.reset_index().rename(columns={"index": "ID"})
        current.insert(0, "DATASET", frame.DATASET.to_numpy())
        predictions.append(current)
        ads.append(metrics["ADS"])
    values = np.asarray(ads, dtype=np.float64)
    finite = values[np.isfinite(values)]
    selection = 0.5 * finite.mean() + 0.5 * finite.min()
    return pd.DataFrame(records), pd.concat(predictions), float(selection)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--eat-cache-root", type=Path,
        default=ROOT / "output/eat_hierarchical_stats_v1",
    )
    parser.add_argument(
        "--spear-cache-root", type=Path,
        default=ROOT / "reports/spear_temporal_attention_v1/cache",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--train-datasets", nargs="+", default=list(TRAIN_DEFAULT))
    parser.add_argument("--dev-datasets", nargs="+", default=list(DEV_DEFAULT))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--width", type=int, default=96)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--context-layers", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.15)
    parser.add_argument("--file-component-weight", type=float, default=0.10)
    parser.add_argument("--presence-weight", type=float, default=0.15)
    parser.add_argument("--joint-weight", type=float, default=0.20)
    parser.add_argument("--consistency-weight", type=float, default=0.05)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--eval-batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=2e-3)
    parser.add_argument("--weight-decay", type=float, default=2e-2)
    parser.add_argument("--eval-every", type=int, default=2)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--seed", type=int, default=20260904)
    args = parser.parse_args()

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

    spear_cache = load_archive(args.spear_cache_root)
    train_pairs = [
        load_block(args.eat_cache_root, spear_cache, name, "train")
        for name in args.train_datasets
    ]
    dev_pairs = [
        load_block(args.eat_cache_root, spear_cache, name, "dev")
        for name in args.dev_datasets
    ]
    train_frame = pd.concat([item[0] for item in train_pairs], ignore_index=True)
    train = concatenate([item[1] for item in train_pairs])
    dev_frames, dev_blocks = zip(*dev_pairs)
    fake_numpy, presence_numpy = targets(train_frame)
    weight_numpy = sample_weights(train_frame)
    device = torch.device(args.device)
    train = move_block_to_device(train, device)
    dev_blocks = tuple(move_block_to_device(block, device) for block in dev_blocks)
    fake_all = torch.from_numpy(fake_numpy).to(device)
    presence_all = torch.from_numpy(presence_numpy).to(device)
    weight_all = torch.from_numpy(weight_numpy).to(device)
    model = UnifiedDualSSLHead(
        eat_layers=train["eat"].shape[2],
        eat_stats=train["eat"].shape[3],
        eat_dimension=train["eat"].shape[4],
        spear_layers=train["spear"].shape[3],
        spear_stats=train["spear"].shape[4],
        spear_dimension=train["spear"].shape[5],
        width=args.width,
        heads=args.heads,
        context_layers=args.context_layers,
        maximum_views=train["eat"].shape[1],
        maximum_bins=train["spear"].shape[2],
        dropout=args.dropout,
        file_component_weight=args.file_component_weight,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )

    best_selection, best_epoch, best_state, stale = -float("inf"), -1, None, 0
    history = []
    generator = np.random.default_rng(args.seed)
    for epoch in range(args.epochs + 1):
        model.train()
        order = generator.permutation(len(train_frame))
        epoch_losses = []
        for offset in range(0, len(order), args.batch_size):
            indices = order[offset:offset + args.batch_size]
            task_logits, presence_logits, joint_logits = model(
                *tensor_batch(train, indices, device)
            )
            index = torch.as_tensor(indices, dtype=torch.long, device=device)
            fake = fake_all.index_select(0, index)
            component_presence = presence_all.index_select(0, index)
            weights = weight_all.index_select(0, index)
            loss, terms = unified_multitask_loss(
                model, task_logits, presence_logits, joint_logits, fake,
                component_presence, weights,
                presence_weight=args.presence_weight,
                joint_weight=args.joint_weight,
                consistency_weight=args.consistency_weight,
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            epoch_losses.append(float(loss.detach()))
        if epoch % args.eval_every:
            continue
        metrics, _, selection = evaluate(
            model, dev_frames, dev_blocks, device, args.eval_batch_size
        )
        row = {
            "EPOCH": epoch,
            "LOSS": float(np.mean(epoch_losses)),
            "SELECTION": selection,
            "MEAN_ADS": float(metrics.ADS.mean()),
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
        model, dev_frames, dev_blocks, device, args.eval_batch_size
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "model": {key: value.cpu() for key, value in best_state.items()},
        "config": {
            "eat_layers": train["eat"].shape[2],
            "eat_stats": train["eat"].shape[3],
            "eat_dimension": train["eat"].shape[4],
            "spear_layers": train["spear"].shape[3],
            "spear_stats": train["spear"].shape[4],
            "spear_dimension": train["spear"].shape[5],
            "width": args.width,
            "heads": args.heads,
            "context_layers": args.context_layers,
            "maximum_views": train["eat"].shape[1],
            "maximum_bins": train["spear"].shape[2],
            "dropout": args.dropout,
            "file_component_weight": args.file_component_weight,
        },
        "eat_projection": train_pairs[0][1]["eat_projection"],
        "spear_projection": spear_cache["__metadata__"]["projection"],
        "spear_layers": spear_cache["__metadata__"]["layers"],
        "spear_bins": int(spear_cache["__metadata__"]["bins"]),
        "train_datasets": args.train_datasets,
        "dev_datasets": args.dev_datasets,
        "best_epoch": best_epoch,
        "selection": selection,
        "seed": args.seed,
    }
    torch.save(checkpoint, args.output_dir / "unified_dual_ssl_head.pt")
    pd.DataFrame(history).to_csv(args.output_dir / "history.csv", index=False)
    metrics.to_csv(args.output_dir / "dev_metrics.csv", index=False)
    predictions.to_csv(args.output_dir / "dev_predictions.csv", index=False)
    (args.output_dir / "summary.json").write_text(json.dumps({
        "best_epoch": best_epoch,
        "selection": selection,
        "mean_ads": float(metrics.ADS.mean()),
        "worst_ads": float(metrics.ADS.min()),
    }, indent=2), encoding="utf-8")
    print(metrics.to_string(index=False))
    print(f"best epoch {best_epoch}: {selection:.6f}")


if __name__ == "__main__":
    main()
