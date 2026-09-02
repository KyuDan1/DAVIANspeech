#!/usr/bin/env python3
"""Train a temporal MIL head on frozen EAT/SPEAR mixture statistics."""

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
sys.path.insert(0, str(ROOT / "scripts"))

from data_guard import assert_no_locked_eval_leakage  # noqa: E402
from dual_domain_stats import interval_view_targets  # noqa: E402
from eat_detector import EatMusicDetector  # noqa: E402
from temporal_dual_domain_head import (  # noqa: E402
    TemporalDualDomainHead,
    temporal_multitask_loss,
)
from train_dual_domain_head import (  # noqa: E402
    Bank,
    evaluate_banks,
    load_bank,
    normalization,
)


SPEAR_SAMPLES = 160_000
MAX_VIEWS = 3
TRAIN_DEFAULT = (
    "temporal_mixed_train_v2",
    "external_mixed_train_v1",
    "mixed_devvoice_train_v1",
    "mixed_fmc_music_train_v1",
    "mixfake_music_train_v1",
    "telephone_mixed_train_v1",
)
DEV_DEFAULT = (
    "temporal_mixed_train_v2",
    "mixfake_music_dev_v1",
    "external_mixed_v1",
    "source_disjoint_mixed_v1",
    "source_disjoint_mixed_equal_v1",
    "factorial_eval_1200_v2_dev",
    "telephone_mixed_dev_v1",
)


@dataclass
class TemporalBank:
    bank: Bank
    eat_view_targets: np.ndarray
    spear_view_targets: np.ndarray


def subset_bank(bank: Bank, selected: np.ndarray, suffix: str) -> Bank:
    return Bank(
        name=f"{bank.name}_{suffix}", channel=bank.channel,
        ids=bank.ids[selected], eat=bank.eat[selected], spear=bank.spear[selected],
        eat_mask=bank.eat_mask[selected], spear_mask=bank.spear_mask[selected],
        targets=bank.targets[selected], joint=bank.joint[selected],
        truth=bank.truth.iloc[np.flatnonzero(selected)].reset_index(drop=True),
    )


def view_targets(bank: Bank, crop_samples: int) -> np.ndarray:
    result = np.zeros((len(bank.ids), MAX_VIEWS, 3), dtype=np.float32)
    has_intervals = {
        "VOICE_START", "VOICE_END", "MUSIC_START", "MUSIC_END", "DURATION",
    }.issubset(bank.truth.columns)
    if not has_intervals:
        result[:] = bank.targets[:, None, :]
        return result
    for index, row in bank.truth.iterrows():
        samples = int(round(float(row.DURATION) * 16_000))
        local, _ = interval_view_targets(
            samples, crop_samples,
            (int(round(float(row.VOICE_START) * 16_000)),
             int(round(float(row.VOICE_END) * 16_000))),
            (int(round(float(row.MUSIC_START) * 16_000)),
             int(round(float(row.MUSIC_END) * 16_000))),
            int(float(row.VOICE_FAKE)), int(float(row.MUSIC_FAKE)),
            max_views=MAX_VIEWS,
        )
        result[index] = local
    return result


def load_temporal_bank(
    stats_root: Path, name: str, channel: str, split: str | None = None,
) -> TemporalBank:
    bank = load_bank(stats_root, name, channel)
    if split is not None and "SPLIT" in bank.truth:
        selected = bank.truth.SPLIT.eq(split).to_numpy()
        bank = subset_bank(bank, selected, split)
    return TemporalBank(
        bank=bank,
        eat_view_targets=view_targets(bank, EatMusicDetector.SAMPLES),
        spear_view_targets=view_targets(bank, SPEAR_SAMPLES),
    )


class TemporalBanksDataset(Dataset):
    def __init__(self, banks: list[TemporalBank]) -> None:
        self.banks = banks
        self.offsets = np.cumsum([0] + [len(item.bank.ids) for item in banks])

    def __len__(self) -> int:
        return int(self.offsets[-1])

    def __getitem__(self, index: int):
        bank_index = int(np.searchsorted(self.offsets, index, side="right") - 1)
        item = self.banks[bank_index]
        local = index - self.offsets[bank_index]
        bank = item.bank
        return (
            bank.eat[local], bank.spear[local], bank.eat_mask[local],
            bank.spear_mask[local], bank.targets[local], bank.joint[local],
            item.eat_view_targets[local], item.spear_view_targets[local],
        )


@torch.inference_mode()
def predict(
    model: TemporalDualDomainHead,
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
            with torch.autocast(
                device_type=device.type, dtype=torch.bfloat16,
                enabled=device.type == "cuda",
            ):
                task, joint, _, _ = model(eat, spear, eat_mask, spear_mask)
            batches.append(model.probabilities(task.float(), joint.float()).cpu())
        result[(bank.name, bank.channel)] = torch.cat(batches).numpy()
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stats-root", type=Path, default=ROOT / "output/dual_domain_stats_v1")
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
    parser.add_argument("--auxiliary-weight", type=float, default=0.5)
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
            ROOT / "data/eval" / name / "truth.csv",
            ROOT / "configs/data_partitions.yaml",
        )

    train_items, dev_items = [], []
    for name in args.train_datasets:
        for channel in args.train_channels:
            split = "train" if name.startswith("temporal_mixed_train_v") else None
            train_items.append(load_temporal_bank(args.stats_root, name, channel, split))
    for name in args.dev_datasets:
        for channel in args.dev_channels:
            split = "dev" if name.startswith("temporal_mixed_train_v") else None
            dev_items.append(load_temporal_bank(args.stats_root, name, channel, split))
    train_banks = [item.bank for item in train_items]
    dev_banks = [item.bank for item in dev_items]
    eat_mean, eat_std = normalization(train_banks, "eat", "eat_mask")
    spear_mean, spear_std = normalization(train_banks, "spear", "spear_mask")
    device = torch.device(args.device)
    norm = {
        "eat_mean": torch.from_numpy(eat_mean).to(device)[None, None],
        "eat_std": torch.from_numpy(eat_std).to(device)[None, None],
        "spear_mean": torch.from_numpy(spear_mean).to(device)[None, None],
        "spear_std": torch.from_numpy(spear_std).to(device)[None, None],
    }
    model = TemporalDualDomainHead(
        width=args.width, heads=args.heads, dropout=args.dropout,
        stream_dropout=args.stream_dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    loader = DataLoader(
        TemporalBanksDataset(train_items), batch_size=args.batch_size,
        shuffle=True, num_workers=0,
        generator=torch.Generator().manual_seed(args.seed),
    )

    history, best_state, best_metrics = [], None, None
    best_selection, best_epoch, stale = -float("inf"), -1, 0
    for epoch in range(args.epochs):
        model.train()
        losses = []
        for batch in loader:
            (eat, spear, eat_mask, spear_mask, targets, joint,
             eat_local, spear_local) = batch
            eat = eat.to(device=device, dtype=torch.float32)
            spear = spear.to(device=device, dtype=torch.float32)
            eat = ((eat - norm["eat_mean"]) / norm["eat_std"]).clamp_(-8, 8)
            spear = ((spear - norm["spear_mean"]) / norm["spear_std"]).clamp_(-8, 8)
            eat_mask = eat_mask.to(device)
            spear_mask = spear_mask.to(device)
            targets = targets.to(device)
            joint = joint.to(device)
            eat_local = eat_local.to(device)
            spear_local = spear_local.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=device.type, dtype=torch.bfloat16,
                enabled=device.type == "cuda",
            ):
                outputs = model(eat, spear, eat_mask, spear_mask)
                loss = temporal_multitask_loss(
                    *outputs, targets, joint, eat_local, spear_local,
                    eat_mask, spear_mask, args.auxiliary_weight,
                )
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
            best_selection, best_epoch, stale = selection, epoch, 0
            best_state = copy.deepcopy(model.state_dict())
            best_metrics = dev_metrics.copy()
        else:
            stale += 1
            if stale >= args.patience:
                break

    if best_state is None or best_metrics is None:
        raise RuntimeError("Training did not produce a checkpoint")
    model.load_state_dict(best_state)
    predictions = predict(model, dev_banks, norm, device, args.batch_size)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_type": "temporal",
        "model": {name: value.cpu() for name, value in best_state.items()},
        "normalization": {
            "eat_mean": eat_mean, "eat_std": eat_std,
            "spear_mean": spear_mean, "spear_std": spear_std,
        },
        "config": {
            "width": args.width, "heads": args.heads, "dropout": args.dropout,
            "stream_dropout": args.stream_dropout,
            "auxiliary_weight": args.auxiliary_weight,
        },
        "seed": args.seed, "best_epoch": best_epoch,
        "selection": best_selection, "train_datasets": args.train_datasets,
        "train_channels": args.train_channels,
    }, args.output_dir / "temporal_dual_domain_head.pt")
    pd.DataFrame(history).to_csv(args.output_dir / "history.csv", index=False)
    best_metrics.to_csv(args.output_dir / "dev_metrics.csv", index=False)
    prediction_rows = []
    for bank in dev_banks:
        values = predictions[(bank.name, bank.channel)]
        for sample_id, probabilities in zip(bank.ids, values):
            prediction_rows.append({
                "DATASET": bank.name, "CHANNEL": bank.channel, "ID": sample_id,
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
