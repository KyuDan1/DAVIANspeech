#!/usr/bin/env python3
"""Train a separation-free, codec-consistent Voice MIL head on cached SPEAR bins."""

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
from spear_voice_mil_head import SpearVoiceMILHead, spear_voice_mil_loss  # noqa: E402
from train_spear_temporal_bin_mil import (  # noqa: E402
    _overlaps, align, load_archive, normalization,
)


TRAIN_DEFAULT = (
    ("external_mixed_train_v1", "truth.csv"),
    ("mixed_devvoice_train_v1", "truth.csv"),
    ("mixed_fmc_music_train_v1", "truth.csv"),
    ("mixfake_music_train_v1", "truth.csv"),
    ("telephone_mixed_train_v1", "truth.csv"),
    ("channel_invariant_factorial_train_v1", "truth.csv"),
    ("temporal_mixed_train_v2", "truth_train.csv"),
)
DEV_DEFAULT = (
    ("mixfake_music_dev_v1", "truth.csv"),
    ("external_mixed_v1", "truth.csv"),
    ("external_mixed_v1_telephone_v1", "truth.csv"),
    ("source_disjoint_mixed_v1", "truth.csv"),
    ("source_disjoint_mixed_v1_telephone_v1", "truth.csv"),
    ("source_disjoint_mixed_equal_v1", "truth.csv"),
    ("telephone_mixed_dev_v1", "truth.csv"),
    ("factorial_eval_1200_v2", "truth_dev.csv"),
    ("multigen_voice_v2", "truth_dev.csv"),
    ("temporal_mixed_train_v2", "truth_dev.csv"),
)


def parse_spec(value: str) -> tuple[str, str]:
    parts = value.split(":", 1)
    return parts[0], parts[1] if len(parts) == 2 else "truth.csv"


def truth_path(spec: tuple[str, str]) -> Path:
    return ROOT / "data/eval" / spec[0] / spec[1]


def generator_group(row) -> str:
    """Return an explicit TTS group, with conservative source fallbacks."""
    value = row.get("VOICE_GENERATOR")
    if pd.isna(value):
        value = row.get("GENERATOR")
    if pd.notna(value):
        return str(value)
    source = str(row.get("VOICE_SOURCE_ID", "unknown")).lower()
    if "replay_fake" in source:
        return "echofake_replay_fake"
    if "replay_bonafide" in source:
        return "echofake_replay_bonafide"
    if "echofake_fake" in source:
        return "echofake_direct_fake"
    if "echofake_bonafide" in source:
        return "echofake_direct_bonafide"
    if source.startswith("la_"):
        return "asvspoof2019_la_fake" if int(row.VOICE_FAKE) else "bonafide"
    return "fake_other" if int(row.VOICE_FAKE) else "bonafide_other"


def local_targets(
    frame: pd.DataFrame, ranges: np.ndarray, valid: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    presence = np.zeros(valid.shape, dtype=np.float32)
    fake = np.zeros(valid.shape, dtype=np.float32)
    for index, row in frame.iterrows():
        intervals: list[tuple[int, int]] = []
        if "VOICE_START" in frame and pd.notna(row.get("VOICE_START")):
            intervals = [(
                int(round(float(row.VOICE_START) * 16_000)),
                int(round(float(row.VOICE_END) * 16_000)),
            )]
        elif "VOICE_RANGES" in frame and pd.notna(row.get("VOICE_RANGES")):
            intervals = [
                (int(round(float(start) * 16_000)), int(round(float(end) * 16_000)))
                for start, end in json.loads(str(row.VOICE_RANGES))
            ]
        local = _overlaps(ranges[index], intervals) & valid[index] if intervals else valid[index]
        presence[index, local] = 1.0
        fake_intervals: list[tuple[int, int]] = []
        if "VOICE_FAKE_RANGES" in frame and pd.notna(row.get("VOICE_FAKE_RANGES")):
            fake_intervals = [
                (int(round(float(start) * 16_000)), int(round(float(end) * 16_000)))
                for start, end in json.loads(str(row.VOICE_FAKE_RANGES))
            ]
        if fake_intervals:
            local_fake = _overlaps(ranges[index], fake_intervals) & local
            fake[index, local_fake] = 1.0
        elif int(row.VOICE_FAKE) == 1:
            fake[index, local] = 1.0
    return presence, fake


def load_fallback(cache_root: Path, dataset: str) -> dict[str, np.ndarray]:
    path = cache_root / "datasets" / dataset / "spear_component_bins.npz"
    archive = np.load(path, allow_pickle=False)
    required = {"ids", "features", "mask"}
    if missing := required.difference(archive.files):
        raise ValueError(f"fallback cache for {dataset} misses {sorted(missing)}")
    return {key: archive[key] for key in required}


def prepare_block(
    cache: dict[str, dict[str, np.ndarray]],
    cache_root: Path,
    spec: tuple[str, str],
    *,
    training: bool,
) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    dataset, _ = spec
    frame = pd.read_csv(truth_path(spec), dtype={"ID": str})
    raw = cache.get(dataset)
    if raw is None:
        raw = load_fallback(cache_root, dataset)
    block = align(raw, frame)
    selected = (
        frame.VOICE_PRESENT.eq(1) & frame.VOICE_FAKE.notna()
    ).to_numpy()
    frame = frame.loc[selected].reset_index(drop=True)
    block = {key: value[selected] for key, value in block.items()}
    frame["DATASET"] = dataset + (f":{spec[1]}" if spec[1] != "truth.csv" else "")
    frame["VOICE_GENERATOR_GROUP"] = [generator_group(row) for _, row in frame.iterrows()]
    if training:
        if "ranges" not in block:
            raise ValueError(f"training cache for {dataset} has no segment ranges")
        local_presence, local_fake = local_targets(
            frame, block["ranges"], block["mask"],
        )
        block["local_presence"] = local_presence
        block["local_fake"] = local_fake
    return frame, block


def concatenate(blocks: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    keys = set(blocks[0])
    if any(set(block) != keys for block in blocks):
        raise ValueError("Voice MIL blocks contain different fields")
    return {key: np.concatenate([block[key] for block in blocks]) for key in keys}


def sample_weights(frame: pd.DataFrame) -> np.ndarray:
    """Give every dataset equal mass, then every generator/label equal mass."""
    keys = list(zip(
        frame.DATASET.astype(str),
        frame.VOICE_GENERATOR_GROUP.astype(str),
        frame.VOICE_FAKE.astype(int),
    ))
    counts = Counter(keys)
    group_count = Counter(dataset for dataset, _, _ in set(keys))
    values = np.asarray([
        1.0 / (group_count[dataset] * counts[key])
        for key, (dataset, _, _) in zip(keys, keys)
    ], dtype=np.float64)
    return (values / values.mean()).astype(np.float32)


def channel_pairs(frame: pd.DataFrame) -> np.ndarray:
    pairs: set[tuple[int, int]] = set()
    if "MIXTURE_ID" in frame:
        available = frame.MIXTURE_ID.notna()
        for _, group in frame.loc[available].groupby(["DATASET", "MIXTURE_ID"]):
            indices = group.index.to_list()
            if len(indices) > 1:
                pairs.update((indices[0], item) for item in indices[1:])
    id_lookup = {str(value): index for index, value in enumerate(frame.ID)}
    if "PARENT_ID" in frame:
        for index, parent in frame.PARENT_ID.dropna().items():
            clean = id_lookup.get(str(parent))
            if clean is not None and int(frame.loc[index, "VOICE_FAKE"]) == int(
                frame.loc[clean, "VOICE_FAKE"]
            ):
                pairs.add((clean, int(index)))
    return np.asarray(sorted(pairs), dtype=np.int64).reshape(-1, 2)


@torch.inference_mode()
def predict(
    model: SpearVoiceMILHead,
    block: dict[str, np.ndarray],
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    model.eval()
    features = block["features"].reshape(
        len(block["features"]), block["features"].shape[1],
        block["features"].shape[2], -1,
    )
    result = []
    for offset in range(0, len(features), batch_size):
        probability, _, _ = model(
            torch.from_numpy(features[offset:offset + batch_size]).to(device),
            torch.from_numpy(block["mask"][offset:offset + batch_size]).to(device),
        )
        result.append(probability.float().cpu().numpy())
    return np.concatenate(result)


def contrast_metrics(frame: pd.DataFrame, probability: np.ndarray) -> list[dict]:
    rows = []
    real = frame.VOICE_FAKE.eq(0).to_numpy()
    for generator in sorted(frame.loc[frame.VOICE_FAKE.eq(1), "VOICE_GENERATOR_GROUP"].unique()):
        fake = (
            frame.VOICE_FAKE.eq(1)
            & frame.VOICE_GENERATOR_GROUP.eq(generator)
        ).to_numpy()
        chosen = real | fake
        if real.sum() and fake.sum():
            rows.append({
                "DATASET": frame.DATASET.iloc[0],
                "VOICE_GENERATOR": generator,
                "REAL_N": int(real.sum()), "FAKE_N": int(fake.sum()),
                "VOICE_EER": official_eer(
                    frame.loc[chosen, "VOICE_FAKE"].astype(int), probability[chosen],
                ),
            })
    return rows


def evaluate(model, frames, blocks, device, batch_size):
    metrics, generators, channels, predictions = [], [], [], []
    all_labels, all_probability = [], []
    for frame, block in zip(frames, blocks):
        probability = predict(model, block, device, batch_size)
        labels = frame.VOICE_FAKE.astype(int).to_numpy()
        eer = official_eer(labels, probability)
        metrics.append({"DATASET": frame.DATASET.iloc[0], "N": len(frame), "VOICE_EER": eer})
        generators.extend(contrast_metrics(frame, probability))
        if "CHANNEL" in frame:
            for channel, group in frame.groupby("CHANNEL"):
                if group.VOICE_FAKE.nunique() == 2:
                    index = group.index.to_numpy()
                    channels.append({
                        "DATASET": frame.DATASET.iloc[0], "CHANNEL": channel,
                        "N": len(group), "VOICE_EER": official_eer(
                            group.VOICE_FAKE.astype(int), probability[index],
                        ),
                    })
        predictions.append(pd.DataFrame({
            "DATASET": frame.DATASET, "ID": frame.ID,
            "SPEAR_VOICE_MIL_PROB": probability,
        }))
        all_labels.append(labels); all_probability.append(probability)
    metric_frame = pd.DataFrame(metrics)
    generator_frame = pd.DataFrame(generators)
    dataset_quality = 1 - metric_frame.VOICE_EER.to_numpy(np.float64)
    generator_quality = 1 - generator_frame.VOICE_EER.to_numpy(np.float64)
    pooled_quality = 1 - official_eer(
        np.concatenate(all_labels), np.concatenate(all_probability),
    )
    selection = float(
        .25 * pooled_quality + .25 * dataset_quality.mean()
        + .20 * dataset_quality.min() + .20 * generator_quality.mean()
        + .10 * generator_quality.min()
    )
    return (
        metric_frame, generator_frame, pd.DataFrame(channels),
        pd.concat(predictions), selection,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cache-root", type=Path,
        default=ROOT / "reports/v47_anchor_cache_train_dev_v2/cache",
    )
    parser.add_argument(
        "--union-cache", type=Path,
        default=ROOT / "reports/spear_temporal_attention_v1/cache",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--train", nargs="+", default=[f"{a}:{b}" for a, b in TRAIN_DEFAULT])
    parser.add_argument("--dev", nargs="+", default=[f"{a}:{b}" for a, b in DEV_DEFAULT])
    parser.add_argument("--exclude-train-generators", nargs="*", default=[])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--hidden", type=int, default=96)
    parser.add_argument("--dropout", type=float, default=.15)
    parser.add_argument("--temperature", type=float, default=2.0)
    parser.add_argument("--minimum-presence-weight", type=float, default=.05)
    parser.add_argument("--local-presence-weight", type=float, default=.20)
    parser.add_argument("--local-fake-weight", type=float, default=.40)
    parser.add_argument("--channel-consistency-weight", type=float, default=.01)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--eval-batch-size", type=int, default=512)
    parser.add_argument("--eval-every", type=int, default=2)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=8e-4)
    parser.add_argument("--weight-decay", type=float, default=2e-2)
    parser.add_argument("--seed", type=int, default=20260921)
    args = parser.parse_args()
    if args.channel_consistency_weight < 0:
        parser.error("channel consistency weight must be non-negative")
    train_specs = [parse_spec(value) for value in args.train]
    dev_specs = [parse_spec(value) for value in args.dev]
    for spec in train_specs:
        assert_no_locked_eval_leakage(
            truth_path(spec), ROOT / "configs/data_partitions.yaml",
        )
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    cache = load_archive(args.union_cache)
    train_frames, train_blocks = zip(*[
        prepare_block(cache, args.cache_root, spec, training=True)
        for spec in train_specs
    ])
    dev_frames, dev_blocks = zip(*[
        prepare_block(cache, args.cache_root, spec, training=False)
        for spec in dev_specs
    ])
    train_frame = pd.concat(train_frames, ignore_index=True)
    train = concatenate(list(train_blocks))
    if args.exclude_train_generators:
        keep = ~train_frame.VOICE_GENERATOR_GROUP.isin(args.exclude_train_generators)
        train_frame = train_frame.loc[keep].reset_index(drop=True)
        train = {key: value[keep.to_numpy()] for key, value in train.items()}
    shape = train["features"].shape
    train["features"] = train["features"].reshape(
        len(train_frame), shape[1], shape[2], -1,
    )
    mean, std = normalization(train["features"], train["mask"])
    device = torch.device(args.device)
    model = SpearVoiceMILHead(
        train["features"].shape[-1], torch.from_numpy(mean), torch.from_numpy(std),
        hidden=args.hidden, dropout=args.dropout, temperature=args.temperature,
        minimum_presence_weight=args.minimum_presence_weight,
    ).to(device)
    labels = torch.from_numpy(
        train_frame.VOICE_FAKE.to_numpy(np.float32, copy=True)
    ).to(device)
    weights_numpy = sample_weights(train_frame)
    pairs = channel_pairs(train_frame)
    print({
        "train_examples": len(train_frame), "dev_examples": sum(map(len, dev_frames)),
        "channel_pairs": len(pairs),
        "generator_counts": train_frame.VOICE_GENERATOR_GROUP.value_counts().to_dict(),
    }, flush=True)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay,
    )
    generator = np.random.default_rng(args.seed)
    sampling = weights_numpy / weights_numpy.sum()
    best_selection, best_epoch, best_state, stale = -np.inf, -1, None, 0
    history = []
    for epoch in range(args.epochs + 1):
        model.train()
        order = generator.choice(
            len(train_frame), size=len(train_frame), replace=True, p=sampling,
        )
        losses = []
        for batch_index, offset in enumerate(range(0, len(order), args.batch_size)):
            index = order[offset:offset + args.batch_size]
            probability, logits, _ = model(
                torch.from_numpy(train["features"][index]).to(device),
                torch.from_numpy(train["mask"][index]).to(device),
            )
            loss, _ = spear_voice_mil_loss(
                probability, logits, torch.from_numpy(train["mask"][index]).to(device).reshape(len(index), -1),
                torch.from_numpy(train["local_presence"][index]).to(device).reshape(len(index), -1),
                torch.from_numpy(train["local_fake"][index]).to(device).reshape(len(index), -1),
                labels[index], torch.from_numpy(weights_numpy[index]).to(device),
                local_presence_weight=args.local_presence_weight,
                local_fake_weight=args.local_fake_weight,
            )
            if args.channel_consistency_weight and len(pairs) and not batch_index % 4:
                chosen = pairs[generator.integers(0, len(pairs), size=min(128, len(pairs)))]
                pair_index = chosen.reshape(-1)
                pair_probability, _, _ = model(
                    torch.from_numpy(train["features"][pair_index]).to(device),
                    torch.from_numpy(train["mask"][pair_index]).to(device),
                )
                pair_logit = torch.logit(pair_probability.clamp(1e-5, 1 - 1e-5)).reshape(-1, 2)
                loss = loss + args.channel_consistency_weight * torch.nn.functional.smooth_l1_loss(
                    pair_logit[:, 0], pair_logit[:, 1],
                )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step(); losses.append(float(loss.detach()))
        if epoch % args.eval_every:
            continue
        metrics, generators, channels, _, selection = evaluate(
            model, dev_frames, dev_blocks, device, args.eval_batch_size,
        )
        record = {
            "EPOCH": epoch, "LOSS": float(np.mean(losses)),
            "SELECTION": selection, "MEAN_VOICE_EER": float(metrics.VOICE_EER.mean()),
            "WORST_VOICE_EER": float(metrics.VOICE_EER.max()),
            "WORST_GENERATOR_EER": float(generators.VOICE_EER.max()),
        }
        history.append(record)
        print(record, flush=True)
        if selection > best_selection + 1e-5:
            best_selection, best_epoch = selection, epoch
            best_state = copy.deepcopy(model.state_dict()); stale = 0
        else:
            stale += 1
            if stale >= args.patience:
                break
    if best_state is None:
        raise RuntimeError("Voice MIL training produced no checkpoint")
    model.load_state_dict(best_state)
    metrics, generators, channels, predictions, selection = evaluate(
        model, dev_frames, dev_blocks, device, args.eval_batch_size,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "model_type": "spear_voice_mil_v1",
        "model": {name: value.cpu() for name, value in best_state.items()},
        "config": {
            "feature_dimension": train["features"].shape[-1],
            "hidden": args.hidden, "dropout": args.dropout,
            "temperature": args.temperature,
            "minimum_presence_weight": args.minimum_presence_weight,
        },
        "projection": cache["__metadata__"]["projection"],
        "layers": cache["__metadata__"]["layers"],
        "bins": cache["__metadata__"]["bins"],
        "seed": args.seed, "best_epoch": best_epoch, "selection": selection,
        "train_specs": train_specs, "dev_specs": dev_specs,
        "exclude_train_generators": args.exclude_train_generators,
        "channel_consistency_weight": args.channel_consistency_weight,
    }
    torch.save(checkpoint, args.output_dir / "spear_voice_mil_head.pt")
    pd.DataFrame(history).to_csv(args.output_dir / "history.csv", index=False)
    metrics.to_csv(args.output_dir / "dev_metrics.csv", index=False)
    generators.to_csv(args.output_dir / "dev_generators.csv", index=False)
    channels.to_csv(args.output_dir / "dev_channels.csv", index=False)
    predictions.to_csv(args.output_dir / "dev_predictions.csv", index=False)
    (args.output_dir / "summary.json").write_text(json.dumps({
        "best_epoch": best_epoch, "selection": selection,
        "mean_voice_eer": float(metrics.VOICE_EER.mean()),
        "worst_voice_eer": float(metrics.VOICE_EER.max()),
        "worst_generator_eer": float(generators.VOICE_EER.max()),
        "train_examples": len(train_frame),
    }, indent=2), encoding="utf-8")
    print(metrics.to_string(index=False)); print(f"selection={selection:.6f}")


if __name__ == "__main__":
    main()
