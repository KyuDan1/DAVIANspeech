#!/usr/bin/env python3
"""Train a separation-free joint voice/music head on SPEAR temporal bins."""

from __future__ import annotations

import argparse
import copy
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from data_guard import assert_no_locked_eval_leakage  # noqa: E402
from evaluate_diagnostic import score_frame  # noqa: E402
from spear_temporal_joint_head import (  # noqa: E402
    SpearTemporalJointAttentionHead, SpearTemporalJointHead,
    joint_temporal_loss,
)
from train_spear_temporal_bin_mil import (  # noqa: E402
    DEV_DEFAULT, TRAIN_DEFAULT, _overlaps, align, concatenate, load_archive,
    normalization, subset, truth,
)


def local_component_targets(
    frame: pd.DataFrame, ranges: np.ndarray, valid: np.ndarray, component: str,
) -> tuple[np.ndarray, np.ndarray]:
    presence = np.zeros(valid.shape, dtype=np.float32)
    fake = np.zeros(valid.shape, dtype=np.float32)
    start_column, end_column = f"{component}_START", f"{component}_END"
    ranges_column = f"{component}_RANGES"
    for index, row in frame.iterrows():
        if int(row[f"{component}_PRESENT"]) == 0:
            continue
        intervals: list[tuple[int, int]] = []
        if start_column in frame and pd.notna(row.get(start_column)):
            intervals = [(
                int(round(float(row[start_column]) * 16_000)),
                int(round(float(row[end_column]) * 16_000)),
            )]
        elif ranges_column in frame and pd.notna(row.get(ranges_column)):
            intervals = [
                (int(round(float(start) * 16_000)), int(round(float(end) * 16_000)))
                for start, end in json.loads(str(row[ranges_column]))
            ]
        local = (
            _overlaps(ranges[index], intervals) & valid[index]
            if intervals else valid[index]
        )
        presence[index, local] = 1
        if pd.notna(row.get(f"{component}_FAKE")) and int(row[f"{component}_FAKE"]) == 1:
            fake[index, local] = 1
    return presence, fake


def prepare_joint(cache, names: list[str], role: str):
    frames, values = [], []
    for name in names:
        frame = truth(name)
        block = align(cache[name], frame)
        split = None
        if name == "temporal_mixed_train_v2":
            split = "train" if role == "train" else "dev"
        frame, block = subset(frame, block, split)
        frame["DATASET"] = name + (f"_{split}" if split else "")
        for component in ("VOICE", "MUSIC"):
            presence, fake = local_component_targets(
                frame, block["ranges"], block["mask"], component
            )
            block[f"local_{component.lower()}_presence"] = presence
            block[f"local_{component.lower()}_fake"] = fake
        frames.append(frame); values.append(block)
        print(f"{role} {frame.DATASET.iloc[0]}: {len(frame)}", flush=True)
    return frames, values


def joint_sample_weights(frame: pd.DataFrame) -> np.ndarray:
    """Balance dataset, file label, and component-layout cells."""
    keys = [
        (
            str(row.DATASET), int(row.FILE_FAKE), int(row.VOICE_PRESENT),
            int(row.MUSIC_PRESENT),
        )
        for row in frame.itertuples(index=False)
    ]
    counts = Counter(keys)
    weights = np.asarray([1 / counts[key] for key in keys], dtype=np.float32)
    labels = frame.FILE_FAKE.astype(int).to_numpy()
    totals = np.bincount(labels, weights=weights, minlength=2)
    weights /= totals[labels].clip(1e-12)
    return weights / weights.mean()


def component_pairs(
    frame: pd.DataFrame, source_column: str, target_column: str,
    nuisance_column: str,
) -> np.ndarray:
    """Pair the same labelled component while the other component changes."""
    groups: dict[tuple[str, int], dict[int, list[int]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for index, row in frame.iterrows():
        source = row.get(source_column)
        target, nuisance = row.get(target_column), row.get(nuisance_column)
        if pd.isna(source) or pd.isna(target) or pd.isna(nuisance):
            continue
        groups[(str(source), int(target))][int(nuisance)].append(index)
    pairs: list[tuple[int, int]] = []
    for nuisance_groups in groups.values():
        if len(nuisance_groups) < 2:
            continue
        keys = sorted(nuisance_groups)
        for nuisance in keys:
            alternatives = [
                item for other in keys if other != nuisance
                for item in nuisance_groups[other]
            ]
            for offset, item in enumerate(nuisance_groups[nuisance]):
                pairs.append((item, alternatives[offset % len(alternatives)]))
    return np.asarray(pairs, dtype=np.int64).reshape(-1, 2)


def channel_pairs(frame: pd.DataFrame) -> np.ndarray:
    """Pair telephone/codec children with the exact clean parent if present."""
    by_id = {str(item): index for index, item in enumerate(frame.ID)}
    pairs = []
    for index, parent in enumerate(frame.get("PARENT_ID", pd.Series(index=frame.index))):
        if pd.notna(parent) and str(parent) in by_id:
            candidate = by_id[str(parent)]
            targets = ("FILE_FAKE", "VOICE_FAKE", "MUSIC_FAKE")
            if all(
                pd.isna(frame.iloc[index][name]) or pd.isna(frame.iloc[candidate][name])
                or int(frame.iloc[index][name]) == int(frame.iloc[candidate][name])
                for name in targets
            ):
                pairs.append((index, candidate))
    return np.asarray(pairs, dtype=np.int64).reshape(-1, 2)


def pair_loss(first: torch.Tensor, pairs: torch.Tensor) -> torch.Tensor:
    if not len(pairs):
        return first.new_zeros(())
    return torch.nn.functional.smooth_l1_loss(
        first[pairs[:, 0]], first[pairs[:, 1]]
    )


@torch.inference_mode()
def predict(model, block, device):
    features = block["features"].reshape(
        len(block["features"]), block["features"].shape[1],
        block["features"].shape[2], -1,
    )
    _, outputs = model(
        torch.from_numpy(features.copy()).to(device),
        torch.from_numpy(block["mask"].copy()).to(device),
    )
    return [value.float().cpu().numpy() for value in outputs]


def evaluate(model, frames, blocks, device):
    rows, predictions, ads_values = [], [], []
    for frame, block in zip(frames, blocks):
        values = predict(model, block, device)
        prediction = pd.DataFrame({
            "FILE_FAKE_PROB": values[0], "VOICE_FAKE_PROB": values[1],
            "MUSIC_FAKE_PROB": values[2], "VOICE_PRESENT_PROB": values[3],
            "MUSIC_PRESENT_PROB": values[4],
        }, index=frame.ID)
        truth_frame = frame.set_index("ID")
        metrics = score_frame(truth_frame.join(prediction))
        rows.append({"DATASET": frame.DATASET.iloc[0], **metrics})
        block_prediction = prediction.reset_index().rename(columns={"index": "ID"})
        block_prediction.insert(0, "DATASET", frame.DATASET.to_numpy())
        predictions.append(block_prediction)
        ads_values.append(metrics["ADS"])
    finite = np.asarray(ads_values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if not len(finite):
        raise ValueError("no development dataset has a finite ADS")
    selection = .5 * finite.mean() + .5 * finite.min()
    return pd.DataFrame(rows), pd.concat(predictions), float(selection)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--train-datasets", nargs="+", default=list(TRAIN_DEFAULT))
    parser.add_argument("--dev-datasets", nargs="+", default=list(DEV_DEFAULT))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--hidden", type=int, default=96)
    parser.add_argument("--architecture", choices=("mlp", "attention"), default="mlp")
    parser.add_argument("--attention-layers", type=int, default=2)
    parser.add_argument("--attention-heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=.15)
    parser.add_argument("--temperature", type=float, default=5.0)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--eval-every", type=int, default=5)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--learning-rate", type=float, default=2e-3)
    parser.add_argument("--weight-decay", type=float, default=2e-2)
    parser.add_argument("--local-weight", type=float, default=.25)
    parser.add_argument("--presence-weight", type=float, default=.15)
    parser.add_argument("--file-weight", type=float, default=.50)
    parser.add_argument("--component-consistency", type=float, default=0.0)
    parser.add_argument("--channel-consistency", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=20260909)
    args = parser.parse_args()
    for name in args.train_datasets:
        assert_no_locked_eval_leakage(
            ROOT / "data/eval" / name / "truth.csv",
            ROOT / "configs/data_partitions.yaml",
        )
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(args.seed)

    cache = load_archive(args.cache_root)
    train_frames, train_blocks = prepare_joint(cache, args.train_datasets, "train")
    dev_frames, dev_blocks = prepare_joint(cache, args.dev_datasets, "dev")
    train_frame = pd.concat(train_frames, ignore_index=True)
    train = concatenate(train_blocks)
    shape = train["features"].shape
    features = train["features"].reshape(shape[0], shape[1], shape[2], -1)
    dimension = features.shape[-1]
    mean, std = normalization(features, train["mask"])
    device = torch.device(args.device)
    model_class = (
        SpearTemporalJointAttentionHead
        if args.architecture == "attention" else SpearTemporalJointHead
    )
    extra = (
        {"layers": args.attention_layers, "heads": args.attention_heads}
        if args.architecture == "attention" else {}
    )
    model = model_class(
        dimension, torch.from_numpy(mean), torch.from_numpy(std),
        hidden=args.hidden, dropout=args.dropout, temperature=args.temperature,
        **extra,
    ).to(device)
    x = torch.from_numpy(features.copy()).to(device)
    mask = torch.from_numpy(train["mask"].copy()).to(device)
    local = {
        key: torch.from_numpy(train[key].copy()).to(device).reshape(len(train_frame), -1)
        for key in (
            "local_voice_presence", "local_music_presence",
            "local_voice_fake", "local_music_fake",
        )
    }
    label = {
        key: torch.from_numpy(
            train_frame[key.upper()].fillna(0).to_numpy(np.float32, copy=True)
        ).to(device)
        for key in ("file_fake", "voice_fake", "music_fake",
                    "voice_present", "music_present")
    }
    weights = torch.from_numpy(joint_sample_weights(train_frame)).to(device)
    voice_pairs = torch.from_numpy(component_pairs(
        train_frame, "VOICE_SOURCE_ID", "VOICE_FAKE", "MUSIC_FAKE"
    )).to(device)
    music_pairs = torch.from_numpy(component_pairs(
        train_frame, "MUSIC_SOURCE_ID", "MUSIC_FAKE", "VOICE_FAKE"
    )).to(device)
    codec_pairs = torch.from_numpy(channel_pairs(train_frame)).to(device)
    print({
        "voice_pairs": len(voice_pairs), "music_pairs": len(music_pairs),
        "channel_pairs": len(codec_pairs),
    }, flush=True)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )

    best_selection, best_epoch, best_state, stale = -np.inf, -1, None, 0
    history = []
    for epoch in range(args.epochs + 1):
        model.train(); logits, outputs = model(x, mask)
        loss, terms = joint_temporal_loss(
            logits, outputs, mask.reshape(len(mask), -1),
            local["local_voice_presence"], local["local_music_presence"],
            local["local_voice_fake"], local["local_music_fake"],
            label["file_fake"], label["voice_fake"], label["music_fake"],
            label["voice_present"], label["music_present"], weights,
            local_weight=args.local_weight, presence_weight=args.presence_weight,
            file_weight=args.file_weight,
        )
        component_consistency = .5 * (
            pair_loss(outputs[1], voice_pairs) + pair_loss(outputs[2], music_pairs)
        )
        channel_consistency = (
            pair_loss(outputs[0], codec_pairs)
            + pair_loss(outputs[1], codec_pairs)
            + pair_loss(outputs[2], codec_pairs)
        ) / 3
        loss = (
            loss + args.component_consistency * component_consistency
            + args.channel_consistency * channel_consistency
        )
        terms["component_consistency"] = component_consistency.detach()
        terms["channel_consistency"] = channel_consistency.detach()
        optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step()
        if epoch % args.eval_every:
            continue
        metrics, _, selection = evaluate(model, dev_frames, dev_blocks, device)
        record = {
            "EPOCH": epoch, "LOSS": float(loss.detach()), "SELECTION": selection,
            "MEAN_ADS": float(metrics.ADS.mean()), "WORST_ADS": float(metrics.ADS.min()),
            **{f"LOSS_{name.upper()}": float(value) for name, value in terms.items()},
        }
        history.append(record)
        if selection > best_selection + 1e-5:
            best_selection, best_epoch = selection, epoch
            best_state = copy.deepcopy(model.state_dict()); stale = 0
        else:
            stale += 1
            if stale >= args.patience:
                break
        if epoch % 50 == 0:
            print(record, flush=True)
    if best_state is None:
        raise RuntimeError("training produced no checkpoint")
    model.load_state_dict(best_state)
    metrics, predictions, selection = evaluate(model, dev_frames, dev_blocks, device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model": {name: value.cpu() for name, value in best_state.items()},
        "config": {
            "feature_dimension": dimension, "hidden": args.hidden,
            "dropout": args.dropout, "temperature": args.temperature,
            "minimum_presence_weight": model.minimum_presence_weight,
            "architecture": args.architecture,
            "attention_layers": args.attention_layers,
            "attention_heads": args.attention_heads,
        },
        "projection": cache["__metadata__"]["projection"],
        "layers": cache["__metadata__"]["layers"],
        "bins": int(cache["__metadata__"]["bins"]),
        "best_epoch": best_epoch, "selection": selection, "seed": args.seed,
    }, args.output_dir / "spear_temporal_joint_head.pt")
    pd.DataFrame(history).to_csv(args.output_dir / "history.csv", index=False)
    metrics.to_csv(args.output_dir / "dev_metrics.csv", index=False)
    predictions.to_csv(args.output_dir / "dev_predictions.csv", index=False)
    (args.output_dir / "summary.json").write_text(json.dumps({
        "best_epoch": best_epoch, "selection": selection,
        "mean_ads": float(metrics.ADS.mean()),
        "worst_ads": float(metrics.ADS.min()),
    }, indent=2), encoding="utf-8")
    print(metrics.to_string(index=False)); print(f"selection={selection:.6f}")


if __name__ == "__main__":
    main()
