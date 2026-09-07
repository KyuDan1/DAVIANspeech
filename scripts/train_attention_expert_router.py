#!/usr/bin/env python3
"""Train and audit a bounded latent-attention gate over four invariant heads.

Only designated training banks update the router.  Six development banks are
used for early stopping/configuration selection, after which the chosen router
is opened once on factorial holdout, telephone factorial, and YuE audits.  The
gate operates on each expert's task-specific hidden representation and is
bounded around uniform voting, so a routing error cannot remove an expert.
"""

from __future__ import annotations

import argparse
import copy
import gc
import hashlib
import json
import random
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from attention_expert_router import BoundedAttentionRouter  # noqa: E402
from data_guard import assert_no_locked_eval_leakage  # noqa: E402
from dual_domain_head import DualDomainHead  # noqa: E402
from invariant_dual_domain_head import InvariantDualDomainHead  # noqa: E402
from train_dual_domain_head import (  # noqa: E402
    DEV_DEFAULT, Bank, load_bank, metrics, truth_path,
)


DEFAULT_MEMBERS = (
    ROOT / "reports/invariant_dual_domain_v4_paired/seed00_fixedteacher_lr1e4/dual_domain_head.pt",
    ROOT / "reports/invariant_dual_domain_v4_paired/seed00_ch01/dual_domain_head.pt",
    ROOT / "reports/invariant_dual_domain_v4_paired/seed01_ch01/dual_domain_head.pt",
    ROOT / "reports/invariant_dual_domain_v4_paired/seed02_ch01/dual_domain_head.pt",
)
AUDIT_DATASETS = (
    "factorial_eval_1200_v2_holdout",
    "phone_factorial_1200_v1",
    "yue_cross_component_audit_v1",
)
TASK_NAMES = ("VOICE", "MUSIC", "FILE")


@dataclass
class RoutedBank:
    name: str
    channel: str
    ids: np.ndarray
    hidden: np.ndarray  # [N,E,T,H]
    probabilities: np.ndarray  # [N,E,T]
    targets: np.ndarray
    task_weights: np.ndarray
    truth: pd.DataFrame


class RoutedDataset(Dataset):
    def __init__(self, banks: list[RoutedBank]) -> None:
        self.hidden = np.concatenate([bank.hidden for bank in banks])
        self.probabilities = np.concatenate([bank.probabilities for bank in banks])
        self.targets = np.concatenate([bank.targets for bank in banks])
        self.task_weights = np.concatenate([bank.task_weights for bank in banks])
        self.bank_id = np.concatenate([
            np.full(len(bank.ids), index, dtype=np.int64)
            for index, bank in enumerate(banks)
        ])

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, index: int):
        return (
            self.hidden[index], self.probabilities[index],
            self.targets[index], self.task_weights[index], self.bank_id[index],
        )


def checkpoint_digest(paths: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.read_bytes())
    return digest.hexdigest()[:16]


def load_members(paths: list[Path], device: torch.device):
    members = []
    for path in paths:
        state = torch.load(path, map_location="cpu", weights_only=False)
        model_class = (
            InvariantDualDomainHead
            if state.get("model_type") == "invariant" else DualDomainHead
        )
        model = model_class(**state["config"]).to(device)
        model.load_state_dict(state["model"])
        model.eval().requires_grad_(False)
        normal = {
            key: torch.from_numpy(np.asarray(state["normalization"][key]))
            .to(device)[None, None]
            for key in ("eat_mean", "eat_std", "spear_mean", "spear_std")
        }
        members.append((model, normal))
    return members


def task_weights(bank: Bank) -> np.ndarray:
    masks = np.column_stack((
        bank.truth.VOICE_PRESENT.eq(1).to_numpy(),
        bank.truth.MUSIC_PRESENT.eq(1).to_numpy(),
        np.ones(len(bank.ids), dtype=bool),
    ))
    result = np.zeros_like(bank.targets, dtype=np.float32)
    for task in range(3):
        selected = masks[:, task]
        labels = bank.targets[selected, task].astype(np.int64)
        counts = np.bincount(labels, minlength=2).clip(min=1)
        result[selected, task] = 1.0 / counts[labels]
        positive = result[:, task] > 0
        result[positive, task] /= result[positive, task].mean()
    return result


@torch.inference_mode()
def extract_bank(
    bank: Bank,
    members,
    device: torch.device,
    batch_size: int,
) -> RoutedBank:
    hidden_batches = []
    probability_batches = []
    for offset in range(0, len(bank.ids), batch_size):
        eat = torch.from_numpy(bank.eat[offset:offset + batch_size]).to(
            device=device, dtype=torch.float32
        )
        spear = torch.from_numpy(bank.spear[offset:offset + batch_size]).to(
            device=device, dtype=torch.float32
        )
        eat_mask = torch.from_numpy(
            bank.eat_mask[offset:offset + batch_size]
        ).to(device)
        spear_mask = torch.from_numpy(
            bank.spear_mask[offset:offset + batch_size]
        ).to(device)
        member_hidden, member_probabilities = [], []
        for model, normal in members:
            normalized_eat = (
                (eat - normal["eat_mean"]) / normal["eat_std"]
            ).clamp(-8, 8)
            normalized_spear = (
                (spear - normal["spear_mean"]) / normal["spear_std"]
            ).clamp(-8, 8)
            with torch.autocast(
                device_type=device.type, dtype=torch.bfloat16,
                enabled=device.type == "cuda",
            ):
                task_logits, joint_logits, hidden = model.forward_with_hidden(
                    normalized_eat, normalized_spear, eat_mask, spear_mask
                )
            probability = model.probabilities(
                task_logits.float(), joint_logits.float()
            )
            member_hidden.append(hidden.float().cpu())
            member_probabilities.append(probability.cpu())
        hidden_batches.append(torch.stack(member_hidden, dim=1))
        probability_batches.append(torch.stack(member_probabilities, dim=1))
    return RoutedBank(
        name=bank.name,
        channel=bank.channel,
        ids=bank.ids,
        hidden=torch.cat(hidden_batches).numpy().astype(np.float16),
        probabilities=torch.cat(probability_batches).numpy().astype(np.float32),
        targets=bank.targets.astype(np.float32),
        task_weights=task_weights(bank),
        truth=bank.truth,
    )


def cache_path(directory: Path, name: str, channel: str) -> Path:
    return directory / f"{name}__{channel}.npz"


def save_cache(path: Path, bank: RoutedBank, digest: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        digest=np.asarray(digest), ids=bank.ids,
        hidden=bank.hidden, probabilities=bank.probabilities,
        targets=bank.targets, task_weights=bank.task_weights,
    )


def load_cached_bank(
    stats_root: Path,
    cache_dir: Path,
    name: str,
    channel: str,
    digest: str,
    members,
    device: torch.device,
    batch_size: int,
) -> RoutedBank:
    path = cache_path(cache_dir, name, channel)
    raw_bank = None
    if path.is_file():
        archive = np.load(path, allow_pickle=False)
        if str(archive["digest"]) == digest:
            ids = archive["ids"].astype(str)
            truth = pd.read_csv(truth_path(name), dtype={"ID": str}).set_index("ID")
            truth = truth.loc[ids].reset_index()
            return RoutedBank(
                name=name, channel=channel, ids=ids,
                hidden=archive["hidden"],
                probabilities=archive["probabilities"],
                targets=archive["targets"],
                task_weights=archive["task_weights"], truth=truth,
            )
    raw_bank = load_bank(stats_root, name, channel)
    result = extract_bank(raw_bank, members, device, batch_size)
    save_cache(path, result, digest)
    del raw_bank
    gc.collect()
    return result


@torch.inference_mode()
def router_probabilities(
    model: BoundedAttentionRouter,
    bank: RoutedBank,
    device: torch.device,
    batch_size: int = 1024,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    probabilities, weights = [], []
    for offset in range(0, len(bank.ids), batch_size):
        hidden = torch.from_numpy(
            bank.hidden[offset:offset + batch_size]
        ).to(device=device, dtype=torch.float32)
        member_probability = torch.from_numpy(
            bank.probabilities[offset:offset + batch_size]
        ).to(device=device, dtype=torch.float32)
        logits, batch_weights = model(hidden, member_probability)
        probabilities.append(logits.sigmoid().cpu())
        weights.append(batch_weights.cpu())
    return torch.cat(probabilities).numpy(), torch.cat(weights).numpy()


def uniform_probabilities(bank: RoutedBank) -> np.ndarray:
    return bank.probabilities.mean(axis=1)


def ensemble_router_probabilities(
    models: list[BoundedAttentionRouter],
    bank: RoutedBank,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    outputs, weights = zip(*[
        router_probabilities(model, bank, device) for model in models
    ])
    return np.mean(outputs, axis=0), np.mean(weights, axis=0)


def evaluate(
    model: BoundedAttentionRouter,
    banks: list[RoutedBank],
    device: torch.device,
) -> tuple[pd.DataFrame, float]:
    rows = []
    for bank in banks:
        probability, _ = router_probabilities(model, bank, device)
        rows.append({
            "DATASET": bank.name, "CHANNEL": bank.channel,
            **metrics(bank.truth, probability),
        })
    frame = pd.DataFrame(rows)
    selection = 0.5 * frame.ADS.mean() + 0.5 * frame.ADS.min()
    return frame, float(selection)


def train_one(
    dataset: RoutedDataset,
    dev_banks: list[RoutedBank],
    device: torch.device,
    strength: float,
    regularization: float,
    seed: int,
    epochs: int,
    patience: int,
    samples_per_epoch: int,
    batch_size: int,
) -> tuple[BoundedAttentionRouter, dict, pd.DataFrame]:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    bank_counts = np.bincount(dataset.bank_id)
    sampling = 1.0 / bank_counts[dataset.bank_id]
    sampler = WeightedRandomSampler(
        torch.from_numpy(sampling), num_samples=samples_per_epoch,
        replacement=True, generator=torch.Generator().manual_seed(seed),
    )
    loader = DataLoader(
        dataset, batch_size=batch_size, sampler=sampler,
        num_workers=0, pin_memory=False,
    )
    model = BoundedAttentionRouter(
        experts=dataset.hidden.shape[1], tasks=dataset.hidden.shape[2],
        expert_width=dataset.hidden.shape[3], model_width=32,
        heads=4, layers=1, dropout=.20, strength=strength,
        temperature=1.5,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=1e-3, weight_decay=2e-2
    )
    best_state, best_selection, best_epoch, best_metrics = None, -np.inf, -1, None
    stale = 0
    history = []
    for epoch in range(epochs):
        model.train()
        losses = []
        for hidden, probability, target, weight, _bank in loader:
            hidden = hidden.to(device=device, dtype=torch.float32)
            probability = probability.to(device=device, dtype=torch.float32)
            target = target.to(device=device, dtype=torch.float32)
            weight = weight.to(device=device, dtype=torch.float32)
            logits, gate = model(hidden, probability)
            base = (
                F.binary_cross_entropy_with_logits(
                    logits, target, reduction="none"
                ) * weight
            ).sum() / weight.sum().clamp_min(1)
            uniform = 1.0 / gate.shape[-1]
            penalty = (gate - uniform).square().mean()
            loss = base + regularization * penalty
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 3.0)
            optimizer.step()
            losses.append(float(loss.detach()))
        dev_metrics, selection = evaluate(model, dev_banks, device)
        history.append({
            "EPOCH": epoch, "LOSS": np.mean(losses),
            "SELECTION": selection, "MEAN_ADS": dev_metrics.ADS.mean(),
            "WORST_ADS": dev_metrics.ADS.min(),
        })
        if selection > best_selection + 1e-5:
            best_state = copy.deepcopy(model.state_dict())
            best_selection, best_epoch = selection, epoch
            best_metrics = dev_metrics.copy()
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                break
    if best_state is None or best_metrics is None:
        raise RuntimeError("router training did not produce a checkpoint")
    model.load_state_dict(best_state)
    model.eval()
    record = {
        "STRENGTH": strength, "REGULARIZATION": regularization,
        "SEED": seed, "BEST_EPOCH": best_epoch,
        "SELECTION": best_selection,
        "MEAN_ADS": float(best_metrics.ADS.mean()),
        "WORST_ADS": float(best_metrics.ADS.min()),
    }
    for row in best_metrics.itertuples():
        record[f"{row.DATASET}_ADS"] = row.ADS
    return model, record, pd.DataFrame(history)


def audit_rows(
    method: str,
    banks: list[RoutedBank],
    probabilities: dict[str, np.ndarray],
) -> list[dict]:
    rows = []
    for bank in banks:
        rows.append({
            "METHOD": method, "DATASET": bank.name,
            **metrics(bank.truth, probabilities[bank.name]),
        })
    return rows


def fusion_audit(
    selected: list[BoundedAttentionRouter],
    banks: dict[str, RoutedBank],
    device: torch.device,
) -> pd.DataFrame:
    # Importing here keeps feature extraction independent of historical v18
    # reconstruction helpers.
    from evaluate_codec_invariant_fusion import AUDITS, fuse, reconstruct_v18
    from evaluate_presence_weighted_file_fusion import reconstruct_dev_v18

    temporal_dev = pd.read_csv(
        ROOT / "reports/temporal_dual_domain_hybrid/ensemble_dev/predictions.csv",
        dtype={"ID": str},
    )
    temporal_audit = pd.read_csv(
        ROOT / "reports/temporal_dual_domain_hybrid/ensemble_audit/predictions.csv",
        dtype={"ID": str},
    )
    fakeprint = pd.read_csv(
        ROOT / "reports/modern_fakeprint_v1/independent_margin_scores.csv",
        dtype={"ID": str},
    )
    invariant_dev = pd.read_csv(
        ROOT / "reports/invariant_dual_domain_v2/dec_n1/dev_predictions.csv",
        dtype={"ID": str},
    )
    invariant_audit = pd.read_csv(
        ROOT / "reports/invariant_dual_domain_v2/dec_n1_audit/predictions.csv",
        dtype={"ID": str},
    )
    specifications = {
        "dev": "factorial_eval_1200_v2_dev",
        "factorial": "factorial_eval_1200_v2_holdout",
        "phone": "phone_factorial_1200_v1",
        "yue": "yue_cross_component_audit_v1",
    }
    rows = []
    for label, name in specifications.items():
        bank = banks[name]
        if label == "dev":
            truth, anchor = reconstruct_dev_v18(
                temporal_dev, fakeprint, invariant_dev
            )
        else:
            truth, anchor = reconstruct_v18(
                AUDITS[label], temporal_audit, fakeprint, invariant_audit
            )
        index = {item: offset for offset, item in enumerate(bank.ids)}
        order = np.asarray([index[item] for item in anchor.index])
        uniform = uniform_probabilities(bank)[order]
        routed, _ = ensemble_router_probabilities(selected, bank, device)
        routed = routed[order]
        for method, expert in (("uniform", uniform), ("attention_router", routed)):
            candidate = anchor.copy()
            for task, column in enumerate((
                "VOICE_FAKE_PROB", "MUSIC_FAKE_PROB", "FILE_FAKE_PROB",
            )):
                candidate[column] = fuse(candidate[column], expert[:, task], .30)
            rows.append({
                "METHOD": method, "DATASET": label,
                **metrics(truth.reset_index(), candidate[
                    ["VOICE_FAKE_PROB", "MUSIC_FAKE_PROB", "FILE_FAKE_PROB"]
                ].to_numpy()),
            })
    result = pd.DataFrame(rows)
    baseline = result.loc[result.METHOD.eq("uniform")].set_index("DATASET").ADS
    result["DELTA_VS_UNIFORM"] = [
        row.ADS - baseline[row.DATASET] for row in result.itertuples()
    ]
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stats-root", type=Path,
        default=ROOT / "output/dual_domain_stats_v1",
    )
    parser.add_argument("--members", type=Path, nargs="+", default=list(DEFAULT_MEMBERS))
    parser.add_argument(
        "--output-dir", type=Path,
        default=ROOT / "reports/attention_expert_router_v1",
    )
    parser.add_argument(
        "--cache-dir", type=Path,
        help="Shared frozen-feature cache (defaults to OUTPUT_DIR/cache).",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--router-batch-size", type=int, default=256)
    parser.add_argument("--samples-per-epoch", type=int, default=12000)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--strengths", type=float, nargs="+", default=[.25, .50])
    parser.add_argument("--regularizations", type=float, nargs="+", default=[.05, .20])
    parser.add_argument("--seeds", type=int, nargs="+", default=[20260903, 20260904, 20260905])
    args = parser.parse_args()
    if len(args.members) != 4:
        parser.error("this controlled comparison expects exactly four experts")
    if any(not path.is_file() for path in args.members):
        parser.error("one or more expert checkpoints are missing")

    first = torch.load(args.members[0], map_location="cpu", weights_only=False)
    train_datasets = list(first["train_datasets"])
    for name in train_datasets:
        assert_no_locked_eval_leakage(
            truth_path(name), ROOT / "configs/data_partitions.yaml"
        )
    device = torch.device(args.device)
    members = load_members(args.members, device)
    digest = checkpoint_digest(args.members)
    cache_dir = args.cache_dir or args.output_dir / "cache"
    train_banks = [
        load_cached_bank(
            args.stats_root, cache_dir, name, "clean", digest,
            members, device, args.batch_size,
        )
        for name in train_datasets
    ]
    dev_banks = [
        load_cached_bank(
            args.stats_root, cache_dir, name, "clean", digest,
            members, device, args.batch_size,
        )
        for name in DEV_DEFAULT
    ]
    audit_banks = [
        load_cached_bank(
            args.stats_root, cache_dir, name, "clean", digest,
            members, device, args.batch_size,
        )
        for name in AUDIT_DATASETS
    ]
    del members
    gc.collect()
    torch.cuda.empty_cache()

    train_dataset = RoutedDataset(train_banks)
    uniform_dev = pd.DataFrame([
        {
            "DATASET": bank.name, "CHANNEL": bank.channel,
            **metrics(bank.truth, uniform_probabilities(bank)),
        }
        for bank in dev_banks
    ])
    uniform_selection = float(
        .5 * uniform_dev.ADS.mean() + .5 * uniform_dev.ADS.min()
    )
    runs = []
    candidates = []
    histories = {}
    for strength in args.strengths:
        for regularization in args.regularizations:
            for seed in args.seeds:
                model, record, history = train_one(
                    train_dataset, dev_banks, device,
                    strength, regularization, seed,
                    args.epochs, args.patience, args.samples_per_epoch,
                    args.router_batch_size,
                )
                record["DELTA_SELECTION_VS_UNIFORM"] = (
                    record["SELECTION"] - uniform_selection
                )
                print(json.dumps(record), flush=True)
                runs.append(record)
                candidates.append((record["SELECTION"], model, record))
                histories[(strength, regularization, seed)] = history

    args.output_dir.mkdir(parents=True, exist_ok=True)
    sweep = pd.DataFrame(runs).sort_values("SELECTION", ascending=False)
    sweep.to_csv(args.output_dir / "dev_sweep.csv", index=False)
    groups: dict[tuple[float, float], list[tuple[BoundedAttentionRouter, dict]]] = {}
    for _selection, model, record in candidates:
        key = (record["STRENGTH"], record["REGULARIZATION"])
        groups.setdefault(key, []).append((model, record))
    group_records = []
    for (strength, regularization), items in groups.items():
        rows = []
        models = [item[0] for item in items]
        for bank in dev_banks:
            probability, _ = ensemble_router_probabilities(models, bank, device)
            rows.append({
                "DATASET": bank.name,
                **metrics(bank.truth, probability),
            })
        frame = pd.DataFrame(rows)
        selection = .5 * frame.ADS.mean() + .5 * frame.ADS.min()
        group_records.append({
            "STRENGTH": strength, "REGULARIZATION": regularization,
            "SEEDS": len(items), "SELECTION": selection,
            "MEAN_ADS": frame.ADS.mean(), "WORST_ADS": frame.ADS.min(),
        })
    group_frame = pd.DataFrame(group_records).sort_values(
        "SELECTION", ascending=False
    )
    group_frame.to_csv(args.output_dir / "ensemble_dev_sweep.csv", index=False)
    selected_group_record = group_frame.iloc[0].to_dict()
    group_key = (
        selected_group_record["STRENGTH"],
        selected_group_record["REGULARIZATION"],
    )
    selected_items = groups[group_key]
    selected_models = [item[0] for item in selected_items]
    selected_records = [item[1] for item in selected_items]
    for model_index, (_model, record) in enumerate(selected_items):
        history_key = (
            record["STRENGTH"], record["REGULARIZATION"], record["SEED"]
        )
        histories[history_key].to_csv(
            args.output_dir / f"selected_history_{model_index:02d}.csv",
            index=False,
        )
    torch.save({
        "models": [
            {name: value.cpu() for name, value in model.state_dict().items()}
            for model in selected_models
        ],
        "config": {
            "experts": 4, "tasks": 3,
            "expert_width": train_dataset.hidden.shape[-1],
            "model_width": 32, "heads": 4, "layers": 1,
            "dropout": .20, "strength": selected_group_record["STRENGTH"],
            "temperature": 1.5,
        },
        "member_sha256_prefix": digest,
        "selection": selected_group_record,
        "member_runs": selected_records,
        "train_datasets": train_datasets,
        "dev_datasets": list(DEV_DEFAULT),
    }, args.output_dir / "attention_router.pt")

    routed_audit, weight_rows = {}, []
    for bank in audit_banks:
        probability, weight = ensemble_router_probabilities(
            selected_models, bank, device
        )
        routed_audit[bank.name] = probability
        for task, task_name in enumerate(TASK_NAMES):
            for member in range(weight.shape[-1]):
                values = weight[:, task, member]
                weight_rows.append({
                    "DATASET": bank.name, "TASK": task_name,
                    "MEMBER": member, "MEAN": values.mean(),
                    "STD": values.std(), "MIN": values.min(),
                    "MAX": values.max(),
                })
    uniform_audit = {
        bank.name: uniform_probabilities(bank) for bank in audit_banks
    }
    audit = pd.DataFrame(
        audit_rows("uniform", audit_banks, uniform_audit)
        + audit_rows("attention_router", audit_banks, routed_audit)
    )
    baseline = audit.loc[audit.METHOD.eq("uniform")].set_index("DATASET").ADS
    audit["DELTA_VS_UNIFORM"] = [
        row.ADS - baseline[row.DATASET] for row in audit.itertuples()
    ]
    audit.to_csv(args.output_dir / "locked_audit.csv", index=False)
    pd.DataFrame(weight_rows).to_csv(
        args.output_dir / "locked_router_weights.csv", index=False
    )
    bank_map = {bank.name: bank for bank in [*dev_banks, *audit_banks]}
    fused = fusion_audit(selected_models, bank_map, device)
    fused.to_csv(args.output_dir / "v18_w30_fusion_audit.csv", index=False)
    summary = {
        "uniform_dev_selection": uniform_selection,
        "selected_ensemble": selected_group_record,
        "member_runs": selected_records,
        "locked_min_delta": float(
            audit.loc[audit.METHOD.eq("attention_router"), "DELTA_VS_UNIFORM"].min()
        ),
        "locked_mean_delta": float(
            audit.loc[audit.METHOD.eq("attention_router"), "DELTA_VS_UNIFORM"].mean()
        ),
        "fused_min_delta": float(
            fused.loc[fused.METHOD.eq("attention_router"), "DELTA_VS_UNIFORM"].min()
        ),
        "fused_mean_delta": float(
            fused.loc[fused.METHOD.eq("attention_router"), "DELTA_VS_UNIFORM"].mean()
        ),
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print("\nLOCKED AUDIT\n" + audit.to_string(index=False))
    print("\nV18+W30 FUSION\n" + fused.to_string(index=False))
    print("\nSUMMARY\n" + json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
