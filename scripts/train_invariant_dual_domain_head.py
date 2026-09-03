#!/usr/bin/env python3
"""Train a source-balanced EAT/SPEAR head with counterfactual invariance.

The paired synthetic banks contain repeated voice and music sources.  This
trainer uses those repetitions as counterfactual supervision: changing only
the voice must not change the music score, changing only the music must not
change the voice score, and telephone encoding must not change authenticity.
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
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from data_guard import assert_no_locked_eval_leakage  # noqa: E402
from dual_domain_head import DualDomainHead, multitask_loss  # noqa: E402
from invariant_dual_domain_head import (  # noqa: E402
    InvariantDualDomainHead,
    invariant_multitask_loss,
)
from train_dual_domain_head import (  # noqa: E402
    DEV_DEFAULT,
    TRAIN_DEFAULT,
    Bank,
    evaluate_banks,
    load_bank,
    normalization,
    predict,
    truth_path,
)


class CounterfactualDataset(Dataset):
    """Return an item plus music-, voice-, and channel-invariant partners."""

    def __init__(
        self, banks: list[Bank], asymmetric_channel: bool = False
    ) -> None:
        self.banks = banks
        self.asymmetric_channel = asymmetric_channel
        self.offsets = np.cumsum([0] + [len(bank.ids) for bank in banks])
        self.bank_index = np.concatenate([
            np.full(len(bank.ids), index, dtype=np.int32)
            for index, bank in enumerate(banks)
        ])
        self.local_index = np.concatenate([
            np.arange(len(bank.ids), dtype=np.int32) for bank in banks
        ])
        self.music_partner, self.music_mask = self._component_partners(
            source_column="MUSIC_SOURCE_ID", target_column="MUSIC_FAKE",
            nuisance_column="VOICE_FAKE",
        )
        self.voice_partner, self.voice_mask = self._component_partners(
            source_column="VOICE_SOURCE_ID", target_column="VOICE_FAKE",
            nuisance_column="MUSIC_FAKE",
        )
        self.channel_partner, self.channel_mask = self._channel_partners()

    def __len__(self) -> int:
        return len(self.bank_index)

    def _global_rows(self) -> pd.DataFrame:
        rows = []
        for bank_number, bank in enumerate(self.banks):
            frame = bank.truth.copy()
            frame["GLOBAL_INDEX"] = np.arange(
                self.offsets[bank_number], self.offsets[bank_number + 1]
            )
            rows.append(frame)
        return pd.concat(rows, ignore_index=True)

    def _component_partners(
        self, source_column: str, target_column: str, nuisance_column: str
    ) -> tuple[np.ndarray, np.ndarray]:
        frame = self._global_rows()
        partner = np.arange(len(self), dtype=np.int64)
        mask = np.zeros(len(self), dtype=np.float32)
        groups: dict[tuple[str, int], dict[int, list[int]]] = defaultdict(
            lambda: defaultdict(list)
        )
        for row in frame.itertuples(index=False):
            source = getattr(row, source_column, None)
            if pd.isna(source) or not str(source).strip():
                continue
            target_value = getattr(row, target_column)
            nuisance_value = getattr(row, nuisance_column)
            # A single-component file has no label for the absent nuisance
            # component.  It remains a valid direct training example, but it
            # cannot define a counterfactual pair for that component.
            if pd.isna(target_value) or pd.isna(nuisance_value):
                continue
            target = int(target_value)
            nuisance = int(nuisance_value)
            groups[(str(source), target)][nuisance].append(int(row.GLOBAL_INDEX))
        for nuisance_groups in groups.values():
            if len(nuisance_groups) < 2:
                continue
            values = sorted(nuisance_groups)
            for nuisance in values:
                alternatives = [
                    item for other in values if other != nuisance
                    for item in nuisance_groups[other]
                ]
                if not alternatives:
                    continue
                for offset, item in enumerate(nuisance_groups[nuisance]):
                    partner[item] = alternatives[offset % len(alternatives)]
                    mask[item] = 1.0
        return partner, mask

    def _channel_partners(self) -> tuple[np.ndarray, np.ndarray]:
        frame = self._global_rows()
        partner = np.arange(len(self), dtype=np.int64)
        mask = np.zeros(len(self), dtype=np.float32)
        by_id = {
            str(row.ID): (int(row.GLOBAL_INDEX), tuple(
                0 if pd.isna(getattr(row, column))
                else int(getattr(row, column))
                for column in ("VOICE_FAKE", "MUSIC_FAKE", "FILE_FAKE")
            ))
            for row in frame.itertuples(index=False)
        }
        for row in frame.itertuples(index=False):
            parent = getattr(row, "PARENT_ID", None)
            if pd.isna(parent) or not str(parent).strip() or str(parent) not in by_id:
                continue
            candidate, target = by_id[str(parent)]
            current_target = tuple(
                0 if pd.isna(getattr(row, column))
                else int(getattr(row, column))
                for column in ("VOICE_FAKE", "MUSIC_FAKE", "FILE_FAKE")
            )
            if target == current_target:
                partner[int(row.GLOBAL_INDEX)] = candidate
                mask[int(row.GLOBAL_INDEX)] = 1.0
                # In the asymmetric mode, the clean parent is a fixed teacher:
                # only a codec child is pulled toward its clean counterpart.
                # The symmetric mode remains available for exact reproduction
                # of earlier experiments.
                if not self.asymmetric_channel:
                    partner[candidate] = int(row.GLOBAL_INDEX)
                    mask[candidate] = 1.0
        return partner, mask

    def _sample(self, index: int):
        bank = self.banks[int(self.bank_index[index])]
        local = int(self.local_index[index])
        return (
            bank.eat[local], bank.spear[local], bank.eat_mask[local],
            bank.spear_mask[local], bank.targets[local], bank.joint[local],
        )

    def __getitem__(self, index: int):
        return (
            *self._sample(index),
            *self._sample(int(self.music_partner[index]))[:4],
            *self._sample(int(self.voice_partner[index]))[:4],
            *self._sample(int(self.channel_partner[index]))[:4],
            self.music_mask[index], self.voice_mask[index], self.channel_mask[index],
        )


def masked_consistency(
    first: torch.Tensor, second: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    difference = F.smooth_l1_loss(first, second, reduction="none")
    return (difference * mask).sum() / mask.sum().clamp_min(1.0)


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
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--samples-per-epoch", type=int, default=12000)
    parser.add_argument(
        "--natural-sampling", action="store_true",
        help="Sample rows uniformly instead of giving every source bank equal mass.",
    )
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.25)
    parser.add_argument("--stream-dropout", type=float, default=0.10)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument(
        "--init-checkpoint", type=Path,
        help=(
            "Warm-start from a compatible head and preserve its normalization. "
            "The unmodified checkpoint is included in model selection."
        ),
    )
    parser.add_argument("--component-consistency", type=float, default=0.5)
    parser.add_argument("--channel-consistency", type=float, default=0.25)
    parser.add_argument(
        "--asymmetric-channel-consistency", action="store_true",
        help=(
            "Treat clean parents as stop-gradient teachers and align only "
            "their codec children."
        ),
    )
    parser.add_argument(
        "--fixed-channel-teacher", action="store_true",
        help=(
            "Use the frozen initialization checkpoint, rather than the moving "
            "student, as the clean-side target for channel consistency."
        ),
    )
    parser.add_argument(
        "--retention-consistency", type=float, default=0.0,
        help="Weight for retaining the frozen teacher output on every input.",
    )
    parser.add_argument("--decoupled", action="store_true")
    parser.add_argument("--file-component-weight", type=float, default=0.10)
    args = parser.parse_args()
    if (args.fixed_channel_teacher or args.retention_consistency > 0) \
            and args.init_checkpoint is None:
        parser.error(
            "--fixed-channel-teacher and --retention-consistency require "
            "--init-checkpoint"
        )
    if args.retention_consistency < 0:
        parser.error("--retention-consistency must be non-negative")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    for name in args.train_datasets:
        assert_no_locked_eval_leakage(truth_path(name), ROOT / "configs/data_partitions.yaml")

    train_banks = [
        load_bank(args.stats_root, name, channel)
        for name in args.train_datasets for channel in args.train_channels
    ]
    dev_banks = [
        load_bank(args.stats_root, name, channel)
        for name in args.dev_datasets for channel in args.dev_channels
    ]
    dataset = CounterfactualDataset(
        train_banks, asymmetric_channel=args.asymmetric_channel_consistency
    )
    print(json.dumps({
        "samples": len(dataset),
        "music_pairs": int(dataset.music_mask.sum()),
        "voice_pairs": int(dataset.voice_mask.sum()),
        "channel_pairs": int(dataset.channel_mask.sum()),
    }), flush=True)
    initial_checkpoint = None
    if args.init_checkpoint is not None:
        initial_checkpoint = torch.load(
            args.init_checkpoint, map_location="cpu", weights_only=False
        )
        expected_type = "invariant" if args.decoupled else "dual_domain"
        if initial_checkpoint.get("model_type", "dual_domain") != expected_type:
            raise ValueError(
                f"Initial checkpoint type does not match {expected_type!r}"
            )
        initial_normalization = initial_checkpoint["normalization"]
        eat_mean = np.asarray(initial_normalization["eat_mean"], dtype=np.float32)
        eat_std = np.asarray(initial_normalization["eat_std"], dtype=np.float32)
        spear_mean = np.asarray(initial_normalization["spear_mean"], dtype=np.float32)
        spear_std = np.asarray(initial_normalization["spear_std"], dtype=np.float32)
    else:
        eat_mean, eat_std = normalization(train_banks, "eat", "eat_mask")
        spear_mean, spear_std = normalization(train_banks, "spear", "spear_mask")
    device = torch.device(args.device)
    norm = {
        "eat_mean": torch.from_numpy(eat_mean).to(device)[None, None],
        "eat_std": torch.from_numpy(eat_std).to(device)[None, None],
        "spear_mean": torch.from_numpy(spear_mean).to(device)[None, None],
        "spear_std": torch.from_numpy(spear_std).to(device)[None, None],
    }
    model_class = InvariantDualDomainHead if args.decoupled else DualDomainHead
    model_kwargs = {
        "width": args.width, "heads": args.heads, "dropout": args.dropout,
        "stream_dropout": args.stream_dropout,
    }
    if args.decoupled:
        model_kwargs["file_component_weight"] = args.file_component_weight
    model = model_class(**model_kwargs).to(device)
    if initial_checkpoint is not None:
        if initial_checkpoint["config"] != model_kwargs:
            raise ValueError(
                "Initial checkpoint config differs from requested model config: "
                f"{initial_checkpoint['config']} != {model_kwargs}"
            )
        model.load_state_dict(initial_checkpoint["model"])
    fixed_teacher = None
    if args.fixed_channel_teacher or args.retention_consistency > 0:
        fixed_teacher = copy.deepcopy(model).eval()
        fixed_teacher.requires_grad_(False)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    # Equal expected contribution per source bank; MixFake must not dominate merely
    # because it has ten times more rows than each synthetic factorial bank.
    weights = np.concatenate([
        np.ones(len(bank.ids), dtype=np.float64) if args.natural_sampling
        else np.full(len(bank.ids), 1.0 / len(bank.ids), dtype=np.float64)
        for bank in train_banks
    ])
    sampler = WeightedRandomSampler(
        torch.from_numpy(weights), num_samples=args.samples_per_epoch,
        replacement=True, generator=torch.Generator().manual_seed(args.seed),
    )
    loader = DataLoader(
        dataset, batch_size=args.batch_size, sampler=sampler, num_workers=0,
        pin_memory=False,
    )

    history, best_state, best_metrics = [], None, None
    best_selection, best_epoch, stale = -float("inf"), -1, 0
    if initial_checkpoint is not None:
        initial_predictions = predict(
            model, dev_banks, norm, device, args.batch_size * 4
        )
        best_metrics, best_selection = evaluate_banks(
            dev_banks, initial_predictions
        )
        best_state = copy.deepcopy(model.state_dict())
        history.append({
            "EPOCH": -1, "TRAIN_LOSS": np.nan, "BASE_LOSS": np.nan,
            "INVARIANCE_LOSS": np.nan, "SELECTION": best_selection,
            "MEAN_ADS": best_metrics.ADS.mean(),
            "WORST_ADS": best_metrics.ADS.min(),
        })
        print(
            f"epoch=-01 warm_start selection={best_selection:.5f} "
            f"mean={best_metrics.ADS.mean():.5f} "
            f"worst={best_metrics.ADS.min():.5f}",
            flush=True,
        )
    for epoch in range(args.epochs):
        model.train()
        losses, base_losses, invariance_losses = [], [], []
        for batch in loader:
            original = batch[:6]
            partner_blocks = (batch[6:10], batch[10:14], batch[14:18])
            pair_masks = batch[18:21]
            eat_parts = [original[0], *(block[0] for block in partner_blocks)]
            spear_parts = [original[1], *(block[1] for block in partner_blocks)]
            eat_mask_parts = [original[2], *(block[2] for block in partner_blocks)]
            spear_mask_parts = [original[3], *(block[3] for block in partner_blocks)]
            eat = torch.cat(eat_parts).to(device=device, dtype=torch.float32)
            spear = torch.cat(spear_parts).to(device=device, dtype=torch.float32)
            eat = ((eat - norm["eat_mean"]) / norm["eat_std"]).clamp_(-8, 8)
            spear = ((spear - norm["spear_mean"]) / norm["spear_std"]).clamp_(-8, 8)
            eat_mask = torch.cat(eat_mask_parts).to(device)
            spear_mask = torch.cat(spear_mask_parts).to(device)
            targets = original[4].to(device)
            joint = original[5].to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=device.type, dtype=torch.bfloat16,
                enabled=device.type == "cuda",
            ):
                task_logits, joint_logits = model(eat, spear, eat_mask, spear_mask)
                chunks_task = task_logits.chunk(4)
                chunks_joint = joint_logits.chunk(4)
                if args.decoupled:
                    base = invariant_multitask_loss(
                        model, chunks_task[0], chunks_joint[0], targets, joint
                    )
                else:
                    base = multitask_loss(
                        chunks_task[0], chunks_joint[0], targets, joint
                    )
                probabilities = [
                    model.probabilities(task, joint_logits_part)
                    for task, joint_logits_part in zip(chunks_task, chunks_joint)
                ]
                teacher_probabilities = None
                if fixed_teacher is not None:
                    with torch.no_grad():
                        teacher_task, teacher_joint = fixed_teacher(
                            eat, spear, eat_mask, spear_mask
                        )
                        teacher_probabilities = [
                            fixed_teacher.probabilities(task, joint_part)
                            for task, joint_part in zip(
                                teacher_task.chunk(4), teacher_joint.chunk(4)
                            )
                        ]
                music_invariance = masked_consistency(
                    probabilities[0][:, 1], probabilities[1][:, 1],
                    pair_masks[0].to(device),
                )
                voice_invariance = masked_consistency(
                    probabilities[0][:, 0], probabilities[2][:, 0],
                    pair_masks[1].to(device),
                )
                channel_mask = pair_masks[2].to(device)[:, None]
                channel_teacher = probabilities[3]
                if args.fixed_channel_teacher:
                    assert teacher_probabilities is not None
                    channel_teacher = teacher_probabilities[3]
                if args.asymmetric_channel_consistency:
                    channel_teacher = channel_teacher.detach()
                channel_invariance = masked_consistency(
                    probabilities[0], channel_teacher, channel_mask,
                )
                retention = probabilities[0].new_zeros(())
                if args.retention_consistency > 0:
                    assert teacher_probabilities is not None
                    retention = F.smooth_l1_loss(
                        probabilities[0], teacher_probabilities[0]
                    )
                invariance = (
                    args.component_consistency * (music_invariance + voice_invariance)
                    + args.channel_consistency * channel_invariance
                    + args.retention_consistency * retention
                )
                loss = base + invariance
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            losses.append(float(loss.detach()))
            base_losses.append(float(base.detach()))
            invariance_losses.append(float(invariance.detach()))

        predictions = predict(model, dev_banks, norm, device, args.batch_size * 4)
        dev_metrics, selection = evaluate_banks(dev_banks, predictions)
        history.append({
            "EPOCH": epoch, "TRAIN_LOSS": np.mean(losses),
            "BASE_LOSS": np.mean(base_losses),
            "INVARIANCE_LOSS": np.mean(invariance_losses),
            "SELECTION": selection, "MEAN_ADS": dev_metrics.ADS.mean(),
            "WORST_ADS": dev_metrics.ADS.min(),
        })
        print(
            f"epoch={epoch:03d} loss={np.mean(losses):.5f} "
            f"inv={np.mean(invariance_losses):.5f} selection={selection:.5f} "
            f"mean={dev_metrics.ADS.mean():.5f} worst={dev_metrics.ADS.min():.5f}",
            flush=True,
        )
        if selection > best_selection + 1e-5:
            best_selection, best_epoch = selection, epoch
            best_state, best_metrics = copy.deepcopy(model.state_dict()), dev_metrics.copy()
            stale = 0
        else:
            stale += 1
            if stale >= args.patience:
                break

    if best_state is None:
        raise RuntimeError("Training did not produce a checkpoint")
    model.load_state_dict(best_state)
    predictions = predict(model, dev_banks, norm, device, args.batch_size * 4)
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
            **({"file_component_weight": args.file_component_weight}
               if args.decoupled else {}),
        },
        "model_type": "invariant" if args.decoupled else "dual_domain",
        "seed": args.seed, "best_epoch": best_epoch,
        "selection": best_selection, "train_datasets": args.train_datasets,
        "train_channels": args.train_channels,
        "component_consistency": args.component_consistency,
        "channel_consistency": args.channel_consistency,
        "asymmetric_channel_consistency": args.asymmetric_channel_consistency,
        "fixed_channel_teacher": args.fixed_channel_teacher,
        "retention_consistency": args.retention_consistency,
        "init_checkpoint": (
            str(args.init_checkpoint) if args.init_checkpoint is not None else None
        ),
    }, args.output_dir / "dual_domain_head.pt")
    pd.DataFrame(history).to_csv(args.output_dir / "history.csv", index=False)
    best_metrics.to_csv(args.output_dir / "dev_metrics.csv", index=False)
    rows = []
    for bank in dev_banks:
        for item, score in zip(bank.ids, predictions[(bank.name, bank.channel)]):
            rows.append({
                "DATASET": bank.name, "CHANNEL": bank.channel, "ID": item,
                "VOICE_FAKE_PROB": score[0], "MUSIC_FAKE_PROB": score[1],
                "FILE_FAKE_PROB": score[2],
            })
    pd.DataFrame(rows).to_csv(args.output_dir / "dev_predictions.csv", index=False)
    summary = {
        "best_epoch": best_epoch, "selection": best_selection,
        "mean_ads": float(best_metrics.ADS.mean()),
        "worst_ads": float(best_metrics.ADS.min()),
        "music_pairs": int(dataset.music_mask.sum()),
        "voice_pairs": int(dataset.voice_mask.sum()),
        "channel_pairs": int(dataset.channel_mask.sum()),
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(best_metrics.to_string(index=False))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
