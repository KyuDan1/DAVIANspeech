#!/usr/bin/env python3
"""Train a compact music MIL expert on cached original-audio SPEAR bins."""

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
from evaluate_diagnostic import official_eer  # noqa: E402
from spear_temporal_bin_head import SpearTemporalBinHead, temporal_bin_loss  # noqa: E402


TRAIN_DEFAULT = (
    "external_mixed_train_v1", "mixed_devvoice_train_v1",
    "mixed_fmc_music_train_v1", "mixfake_music_train_v1",
    "telephone_mixed_train_v1", "temporal_mixed_train_v2",
    "channel_invariant_factorial_train_v1",
)
DEV_DEFAULT = (
    "mixfake_music_dev_v1", "external_mixed_v1", "source_disjoint_mixed_v1",
    "source_disjoint_mixed_equal_v1", "source_disjoint_music_v1",
    "telephone_mixed_dev_v1", "temporal_mixed_train_v2",
)


def load_archive(root: Path) -> dict[str, dict[str, np.ndarray]]:
    """Load sharded union caches and split them by declared dataset."""
    blocks: dict[str, dict[str, list[np.ndarray]]] = {}
    metadata = None
    paths = sorted(root.glob("shard_*.npz"))
    if not paths:
        raise FileNotFoundError(f"no temporal-bin shards in {root}")
    for path in paths:
        archive = np.load(path, allow_pickle=False)
        required = {"ids", "datasets", "features", "ranges", "mask", "projection", "layers"}
        missing = required.difference(archive.files)
        if missing:
            raise ValueError(f"{path} misses fields: {sorted(missing)}")
        current = (archive["projection"], archive["layers"], int(archive["bins"]))
        if metadata is None:
            metadata = current
        elif not (
            np.array_equal(metadata[0], current[0])
            and np.array_equal(metadata[1], current[1])
            and metadata[2] == current[2]
        ):
            raise ValueError("temporal-bin shard metadata differs")
        for name in np.unique(archive["datasets"].astype(str)):
            selected = archive["datasets"].astype(str) == name
            target = blocks.setdefault(name, {
                key: [] for key in ("ids", "features", "ranges", "mask")
            })
            for key in target:
                target[key].append(archive[key][selected])
    result = {}
    for name, values in blocks.items():
        result[name] = {key: np.concatenate(parts) for key, parts in values.items()}
        ids = result[name]["ids"].astype(str)
        if len(set(ids)) != len(ids):
            raise ValueError(f"duplicate IDs in cached dataset {name}")
    result["__metadata__"] = {
        "projection": metadata[0], "layers": metadata[1],
        "bins": np.asarray(metadata[2]),
    }
    return result


def truth(name: str) -> pd.DataFrame:
    return pd.read_csv(ROOT / "data/eval" / name / "truth.csv", dtype={"ID": str})


def align(cache: dict[str, np.ndarray], frame: pd.DataFrame):
    index = {sample_id: position for position, sample_id in enumerate(cache["ids"].astype(str))}
    missing = set(frame.ID) - set(index)
    if missing:
        raise FileNotFoundError(f"{len(missing)} truth IDs have no bin features")
    order = np.asarray([index[sample_id] for sample_id in frame.ID], dtype=np.int64)
    return {key: values[order] for key, values in cache.items() if key != "ids"}


def _overlaps(ranges: np.ndarray, intervals: list[tuple[int, int]]) -> np.ndarray:
    result = np.zeros(ranges.shape[:-1], dtype=bool)
    for start, end in intervals:
        overlap = np.minimum(ranges[..., 1], end) - np.maximum(ranges[..., 0], start)
        result |= overlap >= 1_600
    return result


def local_targets(frame: pd.DataFrame, ranges: np.ndarray, valid: np.ndarray):
    presence = np.zeros(valid.shape, dtype=np.float32)
    fake = np.zeros(valid.shape, dtype=np.float32)
    for index, row in frame.iterrows():
        if int(row.MUSIC_PRESENT) == 0:
            continue
        intervals = []
        if "MUSIC_START" in frame and pd.notna(row.get("MUSIC_START")):
            intervals = [(
                int(round(float(row.MUSIC_START) * 16_000)),
                int(round(float(row.MUSIC_END) * 16_000)),
            )]
        elif "MUSIC_RANGES" in frame and pd.notna(row.get("MUSIC_RANGES")):
            intervals = [
                (int(round(float(start) * 16_000)), int(round(float(end) * 16_000)))
                for start, end in json.loads(str(row.MUSIC_RANGES))
            ]
        if intervals:
            local = _overlaps(ranges[index], intervals) & valid[index]
        else:
            local = valid[index]
        presence[index, local] = 1.0
        if pd.notna(row.MUSIC_FAKE) and int(row.MUSIC_FAKE) == 1:
            fake[index, local] = 1.0
    return presence, fake


def subset(frame: pd.DataFrame, values: dict[str, np.ndarray], split: str | None):
    if split is None or "SPLIT" not in frame:
        return frame.reset_index(drop=True), values
    selected = frame.SPLIT.eq(split).to_numpy()
    return (
        frame.loc[selected].reset_index(drop=True),
        {key: value[selected] for key, value in values.items()},
    )


def prepare(cache, names: list[str], role: str):
    frames, values = [], []
    for name in names:
        base = truth(name)
        aligned = align(cache[name], base)
        split = None
        if name == "temporal_mixed_train_v2":
            split = "train" if role == "train" else "dev"
        base, aligned = subset(base, aligned, split)
        base["DATASET"] = name + (f"_{split}" if split else "")
        local_presence, local_fake = local_targets(
            base, aligned["ranges"], aligned["mask"]
        )
        aligned["local_presence"] = local_presence
        aligned["local_fake"] = local_fake
        frames.append(base)
        values.append(aligned)
        print(f"{role} {base.DATASET.iloc[0]}: {len(base)}", flush=True)
    return frames, values


def sample_weights(frame: pd.DataFrame) -> np.ndarray:
    labels = frame.MUSIC_FAKE.fillna(-1).astype(int)
    keys = list(zip(frame.DATASET, labels))
    pair_count = Counter(keys)
    weights = np.asarray([
        1.0 / pair_count[(name, int(label))]
        for name, label in zip(frame.DATASET, labels)
    ], dtype=np.float32)
    return weights / weights.mean()


def dataset_environments(frame: pd.DataFrame):
    environments = []
    for _, block in frame[frame.MUSIC_PRESENT.eq(1)].groupby("DATASET"):
        positive = block.index[block.MUSIC_FAKE.eq(1)].to_numpy(np.int64)
        negative = block.index[block.MUSIC_FAKE.eq(0)].to_numpy(np.int64)
        if len(positive) and len(negative):
            environments.append((positive, negative))
    return environments


def channel_pairs(frame: pd.DataFrame) -> np.ndarray:
    """Match clean/codec variants without using any evaluation identity."""
    groups = []
    for column in ("MIXTURE_ID", "PARENT_ID"):
        if column not in frame:
            continue
        available = frame[column].notna()
        for _, block in frame.loc[available].groupby(["DATASET", column]):
            indices = block.index.to_list()
            groups.extend((indices[0], index) for index in indices[1:])
    # Telephone training parents also exist as clean external training IDs.
    id_index = {str(sample_id): index for index, sample_id in enumerate(frame.ID)}
    if "PARENT_ID" in frame:
        for index, parent in frame.PARENT_ID.dropna().items():
            clean = id_index.get(str(parent))
            if clean is not None:
                groups.append((clean, int(index)))
    return np.asarray(sorted(set(groups)), dtype=np.int64).reshape(-1, 2)


def normalization(features: np.ndarray, mask: np.ndarray):
    flat = features.reshape(-1, features.shape[-1])
    selected = mask.reshape(-1)
    count = 0
    total = np.zeros(flat.shape[1], dtype=np.float64)
    square = np.zeros(flat.shape[1], dtype=np.float64)
    for offset in range(0, len(flat), 8192):
        block = flat[offset:offset + 8192][selected[offset:offset + 8192]].astype(np.float32)
        total += block.sum(axis=0, dtype=np.float64)
        square += np.square(block, dtype=np.float32).sum(axis=0, dtype=np.float64)
        count += len(block)
    mean = total / count
    variance = np.maximum(square / count - np.square(mean), 1e-8)
    return mean.astype(np.float32), np.sqrt(variance).clip(1e-4).astype(np.float32)


def concatenate(values: list[dict[str, np.ndarray]]):
    return {key: np.concatenate([block[key] for block in values]) for key in values[0]}


def music_eer(frame: pd.DataFrame, scores: np.ndarray) -> float:
    selected = frame.MUSIC_PRESENT.eq(1) & frame.MUSIC_FAKE.notna()
    labels = frame.loc[selected, "MUSIC_FAKE"].astype(int)
    if labels.nunique() < 2:
        return float("nan")
    return float(official_eer(labels, scores[selected.to_numpy()]))


@torch.inference_mode()
def predict(model, values: dict[str, np.ndarray], device: torch.device):
    features = torch.from_numpy(values["features"]).to(device)
    mask = torch.from_numpy(values["mask"]).to(device)
    model.eval()
    return model(features, mask)[1].float().cpu().numpy()


def evaluate(model, frames, values, device):
    rows, predictions, scores = [], [], []
    for frame, block in zip(frames, values):
        score = predict(model, block, device)
        eer = music_eer(frame, score)
        rows.append({"DATASET": frame.DATASET.iloc[0], "N": len(frame), "MUSIC_EER": eer})
        predictions.append(pd.DataFrame({
            "DATASET": frame.DATASET, "ID": frame.ID,
            "TEMPORAL_BIN_MUSIC_PROB": score,
        }))
        scores.append(eer)
    finite = np.asarray(scores)[np.isfinite(scores)]
    selection = 0.5 * (1 - finite.mean()) + 0.5 * (1 - finite.max())
    return pd.DataFrame(rows), pd.concat(predictions), float(selection)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--train-datasets", nargs="+", default=list(TRAIN_DEFAULT))
    parser.add_argument("--dev-datasets", nargs="+", default=list(DEV_DEFAULT))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--eval-every", type=int, default=5)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--learning-rate", type=float, default=3e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--hidden", type=int, default=0)
    parser.add_argument("--dropout", type=float, default=.10)
    parser.add_argument("--temperature", type=float, default=5.0)
    parser.add_argument("--presence-weight", type=float, default=.20)
    parser.add_argument("--local-fake-weight", type=float, default=.50)
    parser.add_argument("--group-dro-temperature", type=float, default=0.0)
    parser.add_argument("--consistency-weight", type=float, default=0.0)
    parser.add_argument("--balance-local-losses", action="store_true")
    parser.add_argument("--seed", type=int, default=20260903)
    args = parser.parse_args()
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(args.seed)
    for name in args.train_datasets:
        assert_no_locked_eval_leakage(
            ROOT / "data/eval" / name / "truth.csv",
            ROOT / "configs/data_partitions.yaml",
        )

    cache = load_archive(args.cache_root)
    train_frames, train_values = prepare(cache, args.train_datasets, "train")
    dev_frames, dev_values = prepare(cache, args.dev_datasets, "dev")
    train_frame = pd.concat(train_frames, ignore_index=True)
    train = concatenate(train_values)
    device = torch.device(args.device)
    feature_dimension = int(np.prod(train["features"].shape[3:]))
    # Layer/stat/width form one per-bin vector; keep view and bin as instances.
    train_features = train["features"].reshape(
        len(train_frame), train["features"].shape[1], train["features"].shape[2], -1
    )
    mean, std = normalization(train_features, train["mask"])
    dev_values = [{**block, "features": block["features"].reshape(
        len(block["features"]), block["features"].shape[1], block["features"].shape[2], -1
    )} for block in dev_values]
    model = SpearTemporalBinHead(
        feature_dimension, torch.from_numpy(mean), torch.from_numpy(std),
        hidden=args.hidden, dropout=args.dropout, temperature=args.temperature,
    ).to(device)
    x = torch.from_numpy(train_features).to(device)
    valid = torch.from_numpy(train["mask"]).to(device)
    local_presence = torch.from_numpy(train["local_presence"]).to(device)
    local_fake = torch.from_numpy(train["local_fake"]).to(device)
    file_present = torch.from_numpy(train_frame.MUSIC_PRESENT.to_numpy(np.float32)).to(device)
    file_fake = torch.from_numpy(
        train_frame.MUSIC_FAKE.fillna(0).to_numpy(np.float32)
    ).to(device)
    weight = torch.from_numpy(sample_weights(train_frame)).to(device)
    environments = [
        (torch.from_numpy(positive).to(device), torch.from_numpy(negative).to(device))
        for positive, negative in dataset_environments(train_frame)
    ]
    pairs = torch.from_numpy(channel_pairs(train_frame)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate,
                                  weight_decay=args.weight_decay)

    best_selection, best_epoch, best_state, stale = -float("inf"), -1, None, 0
    history = []
    for epoch in range(args.epochs + 1):
        model.train()
        logits, file_probability = model(x, valid)
        loss, terms = temporal_bin_loss(
            logits, file_probability, valid.reshape(len(valid), -1),
            local_presence.reshape(len(valid), -1), local_fake.reshape(len(valid), -1),
            file_fake, file_present, weight,
            presence_weight=args.presence_weight,
            local_fake_weight=args.local_fake_weight,
            environments=environments,
            group_dro_temperature=args.group_dro_temperature,
            consistency_pairs=pairs,
            consistency_weight=args.consistency_weight,
            balance_local_losses=args.balance_local_losses,
        )
        optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step()
        if epoch % args.eval_every:
            continue
        metrics, _, selection = evaluate(model, dev_frames, dev_values, device)
        row = {
            "EPOCH": epoch, "LOSS": float(loss.detach()), "SELECTION": selection,
            "MEAN_MUSIC_EER": metrics.MUSIC_EER.mean(),
            "WORST_MUSIC_EER": metrics.MUSIC_EER.max(),
            **{f"LOSS_{key.upper()}": float(value) for key, value in terms.items()},
        }
        history.append(row)
        print(json.dumps(row), flush=True)
        if selection > best_selection + 1e-5:
            best_selection, best_epoch = selection, epoch
            best_state = copy.deepcopy(model.state_dict()); stale = 0
        else:
            stale += 1
            if stale >= args.patience: break
    if best_state is None: raise RuntimeError("training produced no checkpoint")
    model.load_state_dict(best_state)
    metrics, predictions, selection = evaluate(model, dev_frames, dev_values, device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "model": {key: value.cpu() for key, value in best_state.items()},
        "config": {
            "feature_dimension": feature_dimension, "hidden": args.hidden,
            "dropout": args.dropout, "temperature": args.temperature,
            "minimum_presence_weight": model.minimum_presence_weight,
        },
        "projection": cache["__metadata__"]["projection"],
        "layers": cache["__metadata__"]["layers"],
        "bins": int(cache["__metadata__"]["bins"]),
        "train_datasets": args.train_datasets, "dev_datasets": args.dev_datasets,
        "best_epoch": best_epoch, "selection": selection, "seed": args.seed,
        "group_dro_temperature": args.group_dro_temperature,
        "consistency_weight": args.consistency_weight,
        "balance_local_losses": args.balance_local_losses,
    }
    torch.save(checkpoint, args.output_dir / "spear_temporal_bin_head.pt")
    pd.DataFrame(history).to_csv(args.output_dir / "history.csv", index=False)
    metrics.to_csv(args.output_dir / "dev_metrics.csv", index=False)
    predictions.to_csv(args.output_dir / "dev_predictions.csv", index=False)
    (args.output_dir / "summary.json").write_text(json.dumps({
        "best_epoch": best_epoch, "selection": selection,
        "mean_music_eer": float(metrics.MUSIC_EER.mean()),
        "worst_music_eer": float(metrics.MUSIC_EER.max()),
    }, indent=2), encoding="utf-8")
    print(metrics.to_string(index=False)); print(f"best epoch {best_epoch}: {selection:.6f}")


if __name__ == "__main__":
    main()
