#!/usr/bin/env python3
"""Train a source-separation-free EAT/SPEAR multi-task authenticity head."""

from __future__ import annotations

import argparse
import copy
import json
import random
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from data_guard import assert_no_locked_eval_leakage  # noqa: E402
from dual_domain_head import DualDomainHead, multitask_loss  # noqa: E402
from evaluate_diagnostic import official_eer  # noqa: E402


TRAIN_DEFAULT = (
    "external_mixed_train_v1",
    "mixed_devvoice_train_v1",
    "mixed_fmc_music_train_v1",
    "mixfake_music_train_v1",
    "telephone_mixed_train_v1",
)
DEV_DEFAULT = (
    "mixfake_music_dev_v1",
    "external_mixed_v1",
    "source_disjoint_mixed_v1",
    "source_disjoint_mixed_equal_v1",
    "factorial_eval_1200_v2_dev",
    "telephone_mixed_dev_v1",
)
TRUTH_OVERRIDES = {
    "factorial_eval_1200_v2_dev":
        ROOT / "data/eval/factorial_eval_1200_v2/truth_dev.csv",
    "factorial_eval_1200_v2_holdout":
        ROOT / "data/eval/factorial_eval_1200_v2/truth_holdout.csv",
    "factorial_eval_1200_v2_locked":
        ROOT / "data/eval/factorial_eval_1200_v2/truth_locked.csv",
}


@dataclass
class Bank:
    name: str
    channel: str
    ids: np.ndarray
    eat: np.ndarray
    spear: np.ndarray
    eat_mask: np.ndarray
    spear_mask: np.ndarray
    targets: np.ndarray
    joint: np.ndarray
    truth: pd.DataFrame


def truth_path(name: str) -> Path:
    return TRUTH_OVERRIDES.get(name, ROOT / "data/eval" / name / "truth.csv")


def load_stream(directory: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ids, statistics, masks = [], [], []
    paths = sorted(directory.glob("shard_*.npz"))
    if not paths:
        raise FileNotFoundError(f"No statistic shards in {directory}")
    for path in paths:
        shard = np.load(path, allow_pickle=False)
        ids.append(shard["ids"].astype(str))
        statistics.append(shard["statistics"])
        masks.append(shard["view_mask"])
    return np.concatenate(ids), np.concatenate(statistics), np.concatenate(masks)


def load_bank(stats_root: Path, name: str, channel: str) -> Bank:
    eat_ids, eat, eat_mask = load_stream(stats_root / name / "eat" / channel)
    spear_ids, spear, spear_mask = load_stream(stats_root / name / "spear" / channel)
    spear_index = {item: index for index, item in enumerate(spear_ids)}
    if set(eat_ids) != set(spear_ids):
        raise ValueError(f"EAT/SPEAR ID mismatch for {name}/{channel}")
    order = np.asarray([spear_index[item] for item in eat_ids])
    spear, spear_mask = spear[order], spear_mask[order]

    truth = pd.read_csv(truth_path(name), dtype={"ID": str}).set_index("ID")
    missing = set(eat_ids) - set(truth.index)
    if missing:
        raise ValueError(f"No truth for {len(missing)} IDs in {name}: {sorted(missing)[:5]}")
    truth = truth.loc[eat_ids].reset_index()
    voice = truth.VOICE_FAKE.fillna(0).to_numpy(np.float32)
    music = truth.MUSIC_FAKE.fillna(0).to_numpy(np.float32)
    file_fake = truth.FILE_FAKE.to_numpy(np.float32)
    targets = np.column_stack((voice, music, file_fake)).astype(np.float32)
    joint = (2 * voice + music).astype(np.int64)
    return Bank(
        name=name, channel=channel, ids=eat_ids, eat=eat, spear=spear,
        eat_mask=eat_mask, spear_mask=spear_mask, targets=targets,
        joint=joint, truth=truth,
    )


class BanksDataset(Dataset):
    def __init__(self, banks: list[Bank]) -> None:
        self.banks = banks
        self.offsets = np.cumsum([0] + [len(bank.ids) for bank in banks])

    def __len__(self) -> int:
        return int(self.offsets[-1])

    def __getitem__(self, index: int):
        bank_index = int(np.searchsorted(self.offsets, index, side="right") - 1)
        bank = self.banks[bank_index]
        local = index - self.offsets[bank_index]
        return (
            bank.eat[local], bank.spear[local], bank.eat_mask[local],
            bank.spear_mask[local], bank.targets[local], bank.joint[local],
        )


def normalization(banks: list[Bank], field: str, mask_field: str) -> tuple[np.ndarray, np.ndarray]:
    total = total_square = None
    count = 0
    for bank in banks:
        values = getattr(bank, field)
        masks = getattr(bank, mask_field)
        for offset in range(0, len(values), 32):
            chunk = values[offset:offset + 32].astype(np.float32)
            selected = chunk[masks[offset:offset + 32]]
            current = selected.sum(axis=0, dtype=np.float64)
            current_square = np.square(selected, dtype=np.float32).sum(
                axis=0, dtype=np.float64
            )
            total = current if total is None else total + current
            total_square = current_square if total_square is None else total_square + current_square
            count += len(selected)
    mean = total / count
    variance = np.maximum(total_square / count - mean ** 2, 1e-6)
    return mean.astype(np.float32), np.sqrt(variance).astype(np.float32)


def metrics(truth: pd.DataFrame, probabilities: np.ndarray) -> dict[str, float]:
    voice_present = truth.VOICE_PRESENT.eq(1).to_numpy()
    music_present = truth.MUSIC_PRESENT.eq(1).to_numpy()
    file_eer = official_eer(truth.FILE_FAKE, probabilities[:, 2])
    voice_eer = official_eer(
        truth.loc[voice_present, "VOICE_FAKE"], probabilities[voice_present, 0]
    )
    music_eer = official_eer(
        truth.loc[music_present, "MUSIC_FAKE"], probabilities[music_present, 1]
    )
    ads = 0.5 * (1 - file_eer) + 0.2 * (1 - voice_eer) + 0.3 * (1 - music_eer)
    return {
        "FILE_EER": file_eer, "VOICE_EER": voice_eer,
        "MUSIC_EER": music_eer, "ADS": ads,
    }


@torch.inference_mode()
def predict(
    model: DualDomainHead,
    banks: list[Bank],
    norm: dict[str, torch.Tensor],
    device: torch.device,
    batch_size: int,
) -> dict[tuple[str, str], np.ndarray]:
    model.eval()
    result = {}
    for bank in banks:
        batches = []
        for offset in range(0, len(bank.ids), batch_size):
            eat = torch.from_numpy(bank.eat[offset:offset + batch_size]).to(
                device=device, dtype=torch.float32
            )
            spear = torch.from_numpy(bank.spear[offset:offset + batch_size]).to(
                device=device, dtype=torch.float32
            )
            eat = ((eat - norm["eat_mean"]) / norm["eat_std"]).clamp_(-8, 8)
            spear = ((spear - norm["spear_mean"]) / norm["spear_std"]).clamp_(-8, 8)
            eat_mask = torch.from_numpy(bank.eat_mask[offset:offset + batch_size]).to(device)
            spear_mask = torch.from_numpy(bank.spear_mask[offset:offset + batch_size]).to(device)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                                enabled=device.type == "cuda"):
                task_logits, joint_logits = model(eat, spear, eat_mask, spear_mask)
            batches.append(model.probabilities(task_logits.float(), joint_logits.float()).cpu())
        result[(bank.name, bank.channel)] = torch.cat(batches).numpy()
    return result


def evaluate_banks(
    banks: list[Bank], predictions: dict[tuple[str, str], np.ndarray]
) -> tuple[pd.DataFrame, float]:
    rows = []
    for bank in banks:
        row = {"DATASET": bank.name, "CHANNEL": bank.channel}
        row.update(metrics(bank.truth, predictions[(bank.name, bank.channel)]))
        rows.append(row)
    frame = pd.DataFrame(rows)
    # Reward average performance but require source/channel robustness as well.
    selection = 0.5 * frame.ADS.mean() + 0.5 * frame.ADS.min()
    return frame, float(selection)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stats-root", type=Path,
                        default=ROOT / "output/dual_domain_stats_v1")
    parser.add_argument("--train-datasets", nargs="+", default=list(TRAIN_DEFAULT))
    parser.add_argument("--dev-datasets", nargs="+", default=list(DEV_DEFAULT))
    parser.add_argument("--train-channels", nargs="+", default=["clean"])
    parser.add_argument("--dev-channels", nargs="+", default=["clean"])
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=20260902)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.25)
    parser.add_argument("--stream-dropout", type=float, default=0.10)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    for name in args.train_datasets:
        assert_no_locked_eval_leakage(
            truth_path(name), ROOT / "configs/data_partitions.yaml"
        )

    train_banks = [
        load_bank(args.stats_root, name, channel)
        for name in args.train_datasets for channel in args.train_channels
    ]
    dev_banks = [
        load_bank(args.stats_root, name, channel)
        for name in args.dev_datasets for channel in args.dev_channels
    ]
    eat_mean, eat_std = normalization(train_banks, "eat", "eat_mask")
    spear_mean, spear_std = normalization(train_banks, "spear", "spear_mask")
    device = torch.device(args.device)
    norm = {
        "eat_mean": torch.from_numpy(eat_mean).to(device)[None, None],
        "eat_std": torch.from_numpy(eat_std).to(device)[None, None],
        "spear_mean": torch.from_numpy(spear_mean).to(device)[None, None],
        "spear_std": torch.from_numpy(spear_std).to(device)[None, None],
    }

    model = DualDomainHead(
        width=args.width, heads=args.heads, dropout=args.dropout,
        stream_dropout=args.stream_dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    loader_generator = torch.Generator().manual_seed(args.seed)
    loader = DataLoader(
        BanksDataset(train_banks), batch_size=args.batch_size, shuffle=True,
        num_workers=0, generator=loader_generator,
    )

    history, best_state, best_metrics = [], None, None
    best_selection, best_epoch, stale = -float("inf"), -1, 0
    for epoch in range(args.epochs):
        model.train()
        losses = []
        for eat, spear, eat_mask, spear_mask, targets, joint in loader:
            eat = eat.to(device=device, dtype=torch.float32)
            spear = spear.to(device=device, dtype=torch.float32)
            eat = ((eat - norm["eat_mean"]) / norm["eat_std"]).clamp_(-8, 8)
            spear = ((spear - norm["spear_mean"]) / norm["spear_std"]).clamp_(-8, 8)
            eat_mask = eat_mask.to(device)
            spear_mask = spear_mask.to(device)
            targets = targets.to(device)
            joint = joint.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                                enabled=device.type == "cuda"):
                task_logits, joint_logits = model(eat, spear, eat_mask, spear_mask)
                loss = multitask_loss(task_logits, joint_logits, targets, joint)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            losses.append(float(loss.detach()))

        predictions = predict(model, dev_banks, norm, device, args.batch_size)
        dev_metrics, selection = evaluate_banks(dev_banks, predictions)
        history.append({
            "EPOCH": epoch, "TRAIN_LOSS": np.mean(losses),
            "SELECTION": selection, "MEAN_ADS": dev_metrics.ADS.mean(),
            "WORST_ADS": dev_metrics.ADS.min(),
        })
        print(
            f"epoch={epoch:03d} loss={np.mean(losses):.5f} "
            f"selection={selection:.5f} mean={dev_metrics.ADS.mean():.5f} "
            f"worst={dev_metrics.ADS.min():.5f}", flush=True,
        )
        if selection > best_selection + 1e-5:
            best_selection = selection
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            best_metrics = dev_metrics.copy()
            stale = 0
        else:
            stale += 1
            if stale >= args.patience:
                break

    if best_state is None:
        raise RuntimeError("Training did not produce a checkpoint")
    model.load_state_dict(best_state)
    predictions = predict(model, dev_banks, norm, device, args.batch_size)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model": {name: value.cpu() for name, value in best_state.items()},
        "normalization": {
            "eat_mean": eat_mean, "eat_std": eat_std,
            "spear_mean": spear_mean, "spear_std": spear_std,
        },
        "config": {
            "width": args.width, "heads": args.heads, "dropout": args.dropout,
            "stream_dropout": args.stream_dropout,
        },
        "seed": args.seed, "best_epoch": best_epoch,
        "selection": best_selection,
        "train_datasets": args.train_datasets,
        "train_channels": args.train_channels,
    }, args.output_dir / "dual_domain_head.pt")
    pd.DataFrame(history).to_csv(args.output_dir / "history.csv", index=False)
    best_metrics.to_csv(args.output_dir / "dev_metrics.csv", index=False)
    prediction_rows = []
    for bank in dev_banks:
        values = predictions[(bank.name, bank.channel)]
        for item, probabilities in zip(bank.ids, values):
            prediction_rows.append({
                "DATASET": bank.name, "CHANNEL": bank.channel, "ID": item,
                "VOICE_FAKE_PROB": probabilities[0],
                "MUSIC_FAKE_PROB": probabilities[1],
                "FILE_FAKE_PROB": probabilities[2],
            })
    pd.DataFrame(prediction_rows).to_csv(
        args.output_dir / "dev_predictions.csv", index=False
    )
    summary = {
        "best_epoch": best_epoch, "selection": best_selection,
        "mean_ads": float(best_metrics.ADS.mean()),
        "worst_ads": float(best_metrics.ADS.min()),
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(best_metrics.to_string(index=False))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
