#!/usr/bin/env python3
"""Train a source/generator-balanced channel-invariant Music-only head."""

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
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from channel_invariant_music_head import (  # noqa: E402
    ChannelInvariantMusicHead,
    asymmetric_bernoulli_consistency,
    dedicated_music_loss,
)
from data_guard import assert_no_locked_eval_leakage  # noqa: E402
from evaluate_diagnostic import official_eer  # noqa: E402
from train_dual_domain_head import (  # noqa: E402
    Bank, load_bank, normalization, truth_path,
)


TRAIN_DEFAULT = (
    "channel_invariant_factorial_train_v1",
    "external_mixed_train_v1",
    "mixed_devvoice_train_v1",
    "mixed_fmc_music_train_v1",
    "mixfake_music_train_v1",
    "telephone_mixed_train_v1",
    "temporal_mixed_train_v2",
)
DEV_DEFAULT = (
    "mixfake_music_dev_v1",
    "factorial_eval_1200_v2_dev",
    "external_mixed_v1",
    "source_disjoint_mixed_v1",
    "source_disjoint_mixed_equal_v1",
    "source_disjoint_music_v1",
    "telephone_mixed_dev_v1",
    "codec_mixed_dev_v4",
)
TRAIN_TRUTH_OVERRIDES = {
    "temporal_mixed_train_v1":
        ROOT / "data/eval/temporal_mixed_train_v1/truth_train.csv",
    "temporal_mixed_train_v2":
        ROOT / "data/eval/temporal_mixed_train_v2/truth_train.csv",
    "forensic_call_train_v1":
        ROOT / "data/eval/forensic_call_train_v1/truth_train.csv",
}


def _valid(value: object) -> bool:
    return pd.notna(value) and bool(str(value).strip())


def _first(row: pd.Series, columns: tuple[str, ...], fallback: str) -> str:
    for column in columns:
        if column in row and _valid(row[column]):
            return str(row[column])
    return fallback


class MusicInvariantDataset(Dataset):
    """Music-present examples plus a clean teacher for each codec child."""

    def __init__(self, banks: list[Bank]) -> None:
        self.banks = banks
        bank_indices, local_indices, metadata = [], [], []
        for bank_index, bank in enumerate(banks):
            for local_index, row in bank.truth.iterrows():
                if int(row.get("MUSIC_PRESENT", 0)) != 1 or pd.isna(row.MUSIC_FAKE):
                    continue
                bank_indices.append(bank_index)
                local_indices.append(local_index)
                record = row.to_dict()
                record["BANK"] = bank.name
                record["ROW_INDEX"] = len(metadata)
                metadata.append(record)
        self.bank_indices = np.asarray(bank_indices, dtype=np.int32)
        self.local_indices = np.asarray(local_indices, dtype=np.int32)
        self.metadata = pd.DataFrame(metadata)
        self.channel_partner, self.channel_mask = self._channel_partners()

    def __len__(self) -> int:
        return len(self.bank_indices)

    def _channel_partners(self) -> tuple[np.ndarray, np.ndarray]:
        partner = np.arange(len(self), dtype=np.int64)
        mask = np.zeros(len(self), dtype=np.float32)
        by_id = {
            (str(row.BANK), str(row.ID)): int(row.ROW_INDEX)
            for row in self.metadata.itertuples(index=False)
        }
        grouped_clean = {}
        for row in self.metadata.itertuples(index=False):
            mixture = getattr(row, "MIXTURE_ID", None)
            channel = str(getattr(row, "CHANNEL", ""))
            if _valid(mixture) and channel == "clean":
                grouped_clean[(str(row.BANK), str(mixture))] = int(row.ROW_INDEX)
        for row in self.metadata.itertuples(index=False):
            channel = str(getattr(row, "CHANNEL", "clean"))
            if channel == "clean":
                continue
            candidate = None
            parent = getattr(row, "PARENT_ID", None)
            if _valid(parent):
                candidate = by_id.get((str(row.BANK), str(parent)))
            if candidate is None:
                mixture = getattr(row, "MIXTURE_ID", None)
                if _valid(mixture):
                    candidate = grouped_clean.get((str(row.BANK), str(mixture)))
            if candidate is None:
                continue
            teacher = self.metadata.iloc[candidate]
            if (
                int(teacher.MUSIC_FAKE) != int(row.MUSIC_FAKE)
                or str(teacher.get("MUSIC_SOURCE_ID", ""))
                != str(getattr(row, "MUSIC_SOURCE_ID", ""))
            ):
                raise ValueError("clean/codec Music partner changed source or label")
            partner[int(row.ROW_INDEX)] = candidate
            mask[int(row.ROW_INDEX)] = 1.0
        return partner, mask

    def _sample(self, index: int):
        bank = self.banks[int(self.bank_indices[index])]
        local = int(self.local_indices[index])
        row = bank.truth.iloc[local]
        targets = np.asarray([
            0.0 if pd.isna(row.VOICE_FAKE) else float(row.VOICE_FAKE),
            float(row.MUSIC_FAKE), float(row.FILE_FAKE),
            float(row.get("VOICE_PRESENT", 0)),
        ], dtype=np.float32)
        return (
            bank.eat[local], bank.spear[local], bank.eat_mask[local],
            bank.spear_mask[local], targets,
        )

    def __getitem__(self, index: int):
        return (
            *self._sample(index),
            *self._sample(int(self.channel_partner[index]))[:4],
            self.channel_mask[index],
        )


def balanced_sampling_weights(metadata: pd.DataFrame) -> np.ndarray:
    """Equal bank→label→generator→source mass, independent of row repeats."""
    hierarchy = []
    for _, row in metadata.iterrows():
        bank = str(row.BANK)
        label = int(row.MUSIC_FAKE)
        generator = _first(
            row,
            ("MUSIC_GENERATOR", "MUSIC_SOURCE_BANK", "SOURCE_BANK", "SOURCE_DATASET"),
            f"{bank}:unknown_generator",
        )
        source = _first(
            row, ("MUSIC_GROUP", "MUSIC_GROUP_ID", "MUSIC_SOURCE_ID"),
            str(row.ID),
        )
        hierarchy.append((bank, label, generator, source))
    keys = pd.DataFrame(hierarchy, columns=["BANK", "LABEL", "GENERATOR", "SOURCE"])
    bank_count = keys.BANK.nunique()
    weights = np.zeros(len(keys), dtype=np.float64)
    for bank, bank_rows in keys.groupby("BANK"):
        labels = bank_rows.LABEL.unique()
        for label, label_rows in bank_rows.groupby("LABEL"):
            generators = label_rows.GENERATOR.unique()
            for generator, generator_rows in label_rows.groupby("GENERATOR"):
                sources = generator_rows.SOURCE.unique()
                for source, source_rows in generator_rows.groupby("SOURCE"):
                    mass = (
                        1.0 / bank_count / len(labels) / len(generators)
                        / len(sources) / len(source_rows)
                    )
                    weights[source_rows.index.to_numpy()] = mass
    if not np.isfinite(weights).all() or np.any(weights <= 0):
        raise ValueError("invalid source/generator-balanced sampling weights")
    return weights / weights.mean()


def load_training_bank(stats_root: Path, name: str) -> Bank:
    """Load cached statistics and restrict split datasets to authorized train IDs."""
    bank = load_bank(stats_root, name, "clean")
    manifest = TRAIN_TRUTH_OVERRIDES.get(name)
    if manifest is None:
        return bank
    allowed = set(pd.read_csv(manifest, dtype={"ID": str}).ID)
    keep = np.asarray([item in allowed for item in bank.ids], dtype=bool)
    if int(keep.sum()) != len(allowed):
        missing = allowed - set(bank.ids)
        raise FileNotFoundError(
            f"authorized train split has {len(missing)} IDs without statistics"
        )
    return Bank(
        name=bank.name, channel=bank.channel, ids=bank.ids[keep],
        eat=bank.eat[keep], spear=bank.spear[keep],
        eat_mask=bank.eat_mask[keep], spear_mask=bank.spear_mask[keep],
        targets=bank.targets[keep], joint=bank.joint[keep],
        truth=bank.truth.loc[keep].reset_index(drop=True),
    )


def normalized_batch(
    batch: tuple[torch.Tensor, ...],
    norm: dict[str, torch.Tensor],
    device: torch.device,
) -> tuple[torch.Tensor, ...]:
    eat, spear, eat_mask, spear_mask = batch
    eat = eat.to(device=device, dtype=torch.float32)
    spear = spear.to(device=device, dtype=torch.float32)
    eat = ((eat - norm["eat_mean"]) / norm["eat_std"]).clamp_(-8, 8)
    spear = ((spear - norm["spear_mean"]) / norm["spear_std"]).clamp_(-8, 8)
    return eat, spear, eat_mask.to(device), spear_mask.to(device)


@torch.inference_mode()
def predict(
    model: ChannelInvariantMusicHead,
    banks: list[Bank],
    norm: dict[str, torch.Tensor],
    device: torch.device,
    batch_size: int,
) -> dict[tuple[str, str], np.ndarray]:
    model.eval()
    predictions = {}
    for bank in banks:
        scores = []
        for offset in range(0, len(bank.ids), batch_size):
            inputs = normalized_batch((
                torch.from_numpy(bank.eat[offset:offset + batch_size]),
                torch.from_numpy(bank.spear[offset:offset + batch_size]),
                torch.from_numpy(bank.eat_mask[offset:offset + batch_size]),
                torch.from_numpy(bank.spear_mask[offset:offset + batch_size]),
            ), norm, device)
            with torch.autocast(
                device_type=device.type, dtype=torch.bfloat16,
                enabled=device.type == "cuda",
            ):
                music, _ = model(*inputs)
            scores.append(music.float().sigmoid().cpu())
        predictions[(bank.name, bank.channel)] = torch.cat(scores).numpy()
    return predictions


def evaluate(
    banks: list[Bank], predictions: dict[tuple[str, str], np.ndarray],
) -> tuple[pd.DataFrame, float]:
    rows = []
    for bank in banks:
        present = bank.truth.MUSIC_PRESENT.eq(1).to_numpy()
        labels = bank.truth.loc[present, "MUSIC_FAKE"].astype(int)
        eer = official_eer(labels, predictions[(bank.name, bank.channel)][present])
        rows.append({
            "DATASET": bank.name, "CHANNEL": bank.channel,
            "N": int(present.sum()), "MUSIC_EER": eer,
            "MUSIC_SCORE": 1 - eer,
        })
    metrics = pd.DataFrame(rows)
    selection = .5 * metrics.MUSIC_SCORE.mean() + .5 * metrics.MUSIC_SCORE.min()
    return metrics, float(selection)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stats-root", type=Path, default=ROOT / "output/dual_domain_stats_v1",
    )
    parser.add_argument("--train-datasets", nargs="+", default=list(TRAIN_DEFAULT))
    parser.add_argument("--dev-datasets", nargs="+", default=list(DEV_DEFAULT))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=20260904)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--samples-per-epoch", type=int, default=8000)
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=.25)
    parser.add_argument("--stream-dropout", type=float, default=.05)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--rank-weight", type=float, default=.20)
    parser.add_argument("--auxiliary-weight", type=float, default=.05)
    parser.add_argument("--channel-consistency", type=float, default=.25)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    for name in args.train_datasets:
        assert_no_locked_eval_leakage(
            TRAIN_TRUTH_OVERRIDES.get(name, truth_path(name)),
            ROOT / "configs/data_partitions.yaml",
        )
    train_banks = [
        load_training_bank(args.stats_root, name) for name in args.train_datasets
    ]
    dev_banks = [
        load_bank(args.stats_root, name, "clean") for name in args.dev_datasets
    ]
    dataset = MusicInvariantDataset(train_banks)
    weights = balanced_sampling_weights(dataset.metadata)
    print(json.dumps({
        "samples": len(dataset),
        "banks": dataset.metadata.BANK.nunique(),
        "music_sources": dataset.metadata.MUSIC_SOURCE_ID.nunique(),
        "channel_pairs": int(dataset.channel_mask.sum()),
    }), flush=True)

    eat_mean, eat_std = normalization(train_banks, "eat", "eat_mask")
    spear_mean, spear_std = normalization(train_banks, "spear", "spear_mask")
    device = torch.device(args.device)
    norm = {
        "eat_mean": torch.from_numpy(eat_mean).to(device)[None, None],
        "eat_std": torch.from_numpy(eat_std).to(device)[None, None],
        "spear_mean": torch.from_numpy(spear_mean).to(device)[None, None],
        "spear_std": torch.from_numpy(spear_std).to(device)[None, None],
    }
    config = {
        "width": args.width, "heads": args.heads, "dropout": args.dropout,
        "stream_dropout": args.stream_dropout,
    }
    model = ChannelInvariantMusicHead(**config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay,
    )
    sampler = WeightedRandomSampler(
        torch.from_numpy(weights), num_samples=args.samples_per_epoch,
        replacement=True, generator=torch.Generator().manual_seed(args.seed),
    )
    loader = DataLoader(
        dataset, batch_size=args.batch_size, sampler=sampler, num_workers=0,
    )

    history, best_state, best_metrics = [], None, None
    best_selection, best_epoch, stale = -float("inf"), -1, 0
    for epoch in range(args.epochs):
        model.train()
        tracked = {key: [] for key in ("total", "music_bce", "rank", "auxiliary", "channel")}
        for batch in loader:
            original = normalized_batch(batch[:4], norm, device)
            targets = batch[4].to(device=device, dtype=torch.float32)
            teacher = normalized_batch(batch[5:9], norm, device)
            channel_mask = batch[9].to(device=device, dtype=torch.float32)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=device.type, dtype=torch.bfloat16,
                enabled=device.type == "cuda",
            ):
                music_logits, auxiliary_logits = model(*original)
                base, pieces = dedicated_music_loss(
                    music_logits, auxiliary_logits,
                    targets[:, 1], targets[:, 0], targets[:, 2], targets[:, 3],
                    rank_weight=args.rank_weight,
                    auxiliary_weight=args.auxiliary_weight,
                )
            # Keep the asymmetric clean teacher deterministic: independent
            # dropout in the target branch otherwise masquerades as a codec
            # difference and weakens invariance.
            model.eval()
            with torch.no_grad(), torch.autocast(
                device_type=device.type, dtype=torch.bfloat16,
                enabled=device.type == "cuda",
            ):
                teacher_music, _ = model(*teacher)
            model.train()
            with torch.autocast(
                device_type=device.type, dtype=torch.bfloat16,
                enabled=device.type == "cuda",
            ):
                channel = asymmetric_bernoulli_consistency(
                    music_logits, teacher_music, channel_mask,
                )
                loss = base + args.channel_consistency * channel
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            tracked["total"].append(float(loss.detach()))
            tracked["channel"].append(float(channel.detach()))
            for key, value in pieces.items():
                tracked[key].append(float(value.detach()))

        predictions = predict(model, dev_banks, norm, device, args.batch_size * 4)
        dev_metrics, selection = evaluate(dev_banks, predictions)
        record = {
            "EPOCH": epoch, "SELECTION": selection,
            "MEAN_MUSIC_SCORE": dev_metrics.MUSIC_SCORE.mean(),
            "WORST_MUSIC_SCORE": dev_metrics.MUSIC_SCORE.min(),
        }
        record.update({f"LOSS_{key.upper()}": np.mean(value) for key, value in tracked.items()})
        history.append(record)
        print(
            f"epoch={epoch:03d} loss={record['LOSS_TOTAL']:.5f} "
            f"selection={selection:.5f} mean={record['MEAN_MUSIC_SCORE']:.5f} "
            f"worst={record['WORST_MUSIC_SCORE']:.5f}",
            flush=True,
        )
        if selection > best_selection + 1e-5:
            best_selection, best_epoch = selection, epoch
            best_state = copy.deepcopy(model.state_dict())
            best_metrics = dev_metrics.copy()
            stale = 0
        else:
            stale += 1
            if stale >= args.patience:
                break
    if best_state is None or best_metrics is None:
        raise RuntimeError("training did not produce a checkpoint")

    model.load_state_dict(best_state)
    predictions = predict(model, dev_banks, norm, device, args.batch_size * 4)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "model_type": "channel_invariant_music",
        "model": {name: value.cpu() for name, value in best_state.items()},
        "normalization": {
            "eat_mean": eat_mean, "eat_std": eat_std,
            "spear_mean": spear_mean, "spear_std": spear_std,
        },
        "config": config,
        "seed": args.seed,
        "best_epoch": best_epoch,
        "selection": best_selection,
        "train_datasets": args.train_datasets,
        "dev_datasets": args.dev_datasets,
        "rank_weight": args.rank_weight,
        "auxiliary_weight": args.auxiliary_weight,
        "channel_consistency": args.channel_consistency,
        "sampling": "equal bank-label-generator-source mass",
    }
    torch.save(checkpoint, args.output_dir / "music_head.pt")
    pd.DataFrame(history).to_csv(args.output_dir / "history.csv", index=False)
    best_metrics.to_csv(args.output_dir / "dev_metrics.csv", index=False)
    rows = []
    for bank in dev_banks:
        for item, score in zip(bank.ids, predictions[(bank.name, bank.channel)]):
            rows.append({
                "DATASET": bank.name, "CHANNEL": bank.channel,
                "ID": item, "MUSIC_FAKE_PROB": score,
            })
    pd.DataFrame(rows).to_csv(args.output_dir / "dev_predictions.csv", index=False)
    summary = {
        "best_epoch": best_epoch, "selection": best_selection,
        "mean_music_score": float(best_metrics.MUSIC_SCORE.mean()),
        "worst_music_score": float(best_metrics.MUSIC_SCORE.min()),
        "channel_pairs": int(dataset.channel_mask.sum()),
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8",
    )
    print(best_metrics.to_string(index=False))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
