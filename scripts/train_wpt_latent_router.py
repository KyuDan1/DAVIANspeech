#!/usr/bin/env python3
"""Compare fixed MoE with a bounded WPT-latent router over distinct experts."""

from __future__ import annotations

import argparse
import copy
import json
import random
import sys
from collections import Counter
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

from evaluate_diagnostic import official_eer, score_frame  # noqa: E402
from data_guard import assert_no_locked_eval_leakage  # noqa: E402
from latent_expert_router import (  # noqa: E402
    BoundedLatentExpertRouter, pairwise_ranking_loss,
)
from train_wpt_spectra_multitask import load_frame, truth_path  # noqa: E402


TRAIN_BANKS = (
    "multigen_music_presence_train_v1",
    "phone_presence_factorial_train_v1",
)
DEV_BANKS = (
    "mixfake_music_dev_v1",
    "external_mixed_v1",
    "source_disjoint_mixed_v1",
    "source_disjoint_mixed_equal_v1",
    "source_disjoint_music_v1",
    "factorial_eval_1200_v2_dev",
    "telephone_mixed_dev_v1",
)
LOCKED_BANKS = (
    "factorial_eval_1200_v2_holdout",
    "phone_factorial_1200_v1",
    "yue_cross_component_audit_v1",
)
COLUMNS = ("VOICE_FAKE_PROB", "MUSIC_FAKE_PROB", "FILE_FAKE_PROB")


@dataclass
class Bank:
    name: str
    ids: np.ndarray
    latent: np.ndarray
    expert_logits: np.ndarray
    targets: np.ndarray
    masks: np.ndarray
    truth: pd.DataFrame


def logit(values) -> np.ndarray:
    values = np.clip(np.asarray(values, dtype=np.float64), 1e-5, 1 - 1e-5)
    return np.log(values) - np.log1p(-values)


def sigmoid(values) -> np.ndarray:
    return np.exp(-np.logaddexp(0, -np.asarray(values, dtype=np.float64)))


def indexed_predictions(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, dtype={"ID": str})
    if frame.duplicated(["DATASET", "ID"]).any():
        raise ValueError(f"duplicate prediction IDs: {path}")
    return frame.set_index(["DATASET", "ID"])


def load_bank(
    name: str, latent_root: Path, wpt: pd.DataFrame, unified: pd.DataFrame,
    phone: pd.DataFrame | None, unified_latent_root: Path | None,
) -> Bank:
    truth = load_frame(name, "eval")
    keys = pd.MultiIndex.from_arrays((
        np.full(len(truth), name), truth.ID.to_numpy(dtype=str)
    ), names=("DATASET", "ID"))
    wpt_values = wpt.loc[keys, list(COLUMNS)].to_numpy(np.float64)
    unified_values = unified.loc[keys, list(COLUMNS)].to_numpy(np.float64)
    archive = np.load(latent_root / f"{name}_latent.npz", allow_pickle=False)
    latent_ids = archive["ids"].astype(str)
    if not np.array_equal(latent_ids, truth.ID.to_numpy(dtype=str)):
        raise ValueError(f"stale or misordered latent cache for {name}")
    stored_logits = archive["logits"]
    if not np.allclose(sigmoid(stored_logits), wpt_values, atol=2e-5):
        raise ValueError(f"WPT prediction/latent mismatch for {name}")
    targets = np.column_stack((
        truth.VOICE_FAKE.fillna(0),
        truth.MUSIC_FAKE.fillna(0),
        truth.FILE_FAKE,
    )).astype(np.float32)
    masks = np.column_stack((
        truth.VOICE_PRESENT.eq(1),
        truth.MUSIC_PRESENT.eq(1),
        np.ones(len(truth), dtype=bool),
    ))
    wpt_latent = archive["latent"].astype(np.float32)
    if unified_latent_root is not None:
        unified_archive = np.load(
            unified_latent_root / f"{name}_latent.npz", allow_pickle=False
        )
        if not np.array_equal(
            unified_archive["ids"].astype(str), truth.ID.to_numpy(dtype=str)
        ):
            raise ValueError(f"stale or misordered unified latent cache for {name}")
        # Expert order is Unified first, WPT second.  Keep the internal
        # embeddings in that same order so the attention router receives one
        # genuinely expert-specific token per detector.
        latent = np.concatenate((
            unified_archive["latent"].astype(np.float32), wpt_latent,
        ), axis=1)
    else:
        latent = wpt_latent
    global_features = []
    if {"VOICE_PRESENT_PROB", "MUSIC_PRESENT_PROB"}.issubset(unified.columns):
        component_presence = unified.loc[
            keys, ["VOICE_PRESENT_PROB", "MUSIC_PRESENT_PROB"]
        ].to_numpy(np.float32)
        voice_presence, music_presence = component_presence.T
        global_features.append(np.column_stack((
            logit(voice_presence), logit(music_presence),
            voice_presence, music_presence,
            voice_presence * music_presence,
        )).astype(np.float32))
    if phone is not None:
        phone_probability = phone.loc[keys, "PHONE_PROB"].to_numpy(np.float32)
        phone_features = np.column_stack((
            logit(phone_probability), phone_probability,
            phone_probability * (1 - phone_probability),
        )).astype(np.float32)
        global_features.append(phone_features)
    if global_features:
        latent = np.concatenate((latent, *global_features), axis=1)
    return Bank(
        name=name, ids=truth.ID.to_numpy(dtype=str),
        latent=latent,
        expert_logits=np.stack((logit(unified_values), logit(wpt_values)), axis=1)
        .astype(np.float32),
        targets=targets, masks=masks, truth=truth,
    )


def task_eer(bank: Bank, logits: np.ndarray, task: int) -> float:
    valid = bank.masks[:, task]
    labels = bank.targets[valid, task].astype(np.int64)
    if len(np.unique(labels)) < 2:
        return float("nan")
    return official_eer(labels, logits[valid, task])


def bank_metrics(bank: Bank, logits: np.ndarray) -> dict[str, float]:
    eers = [task_eer(bank, logits, task) for task in range(3)]
    return {
        "VOICE_EER": eers[0], "MUSIC_EER": eers[1], "FILE_EER": eers[2],
        "ADS": .2 * (1 - eers[0]) + .3 * (1 - eers[1]) + .5 * (1 - eers[2]),
    }


def music_selection(frame: pd.DataFrame) -> float:
    """Mean-plus-worst Music ranking score; only this routed axis is deployed."""
    values = 1 - frame.MUSIC_EER.dropna().to_numpy(np.float64)
    if not len(values):
        raise ValueError("router selection needs at least one Music EER")
    return float(.5 * values.mean() + .5 * values.min())


def fixed_logits(bank: Bank, prior: np.ndarray) -> np.ndarray:
    return (bank.expert_logits.transpose(0, 2, 1) * prior[None]).sum(-1)


def choose_prior(dev_banks: list[Bank], grid: np.ndarray) -> np.ndarray:
    """Choose each task's WPT weight using mean+worst development EER."""
    weights = []
    for task in range(3):
        rows = []
        for value in grid:
            current = []
            for bank in dev_banks:
                logits = (
                    (1 - value) * bank.expert_logits[:, 0, task]
                    + value * bank.expert_logits[:, 1, task]
                )
                eer = task_eer(
                    bank,
                    np.column_stack([logits] * 3),
                    task,
                )
                if np.isfinite(eer):
                    current.append(eer)
            rows.append((.5 * np.mean(current) + .5 * np.max(current), value))
        weights.append(min(rows)[1])
    # Non-zero weights are required by the residual router and also prevent a
    # dev grid from permanently deleting an expert before OOD evaluation.
    weights = np.clip(np.asarray(weights), .01, .99)
    return np.column_stack((1 - weights, weights)).astype(np.float32)


class RouterDataset(Dataset):
    def __init__(self, banks: list[Bank]) -> None:
        self.latent = np.concatenate([bank.latent for bank in banks])
        self.logits = np.concatenate([bank.expert_logits for bank in banks])
        self.targets = np.concatenate([bank.targets for bank in banks])
        self.masks = np.concatenate([bank.masks for bank in banks])
        self.bank = np.concatenate([
            np.full(len(item.ids), index, dtype=np.int64)
            for index, item in enumerate(banks)
        ])
        self.weight = self._weights(banks)

    @staticmethod
    def _weights(banks: list[Bank]) -> np.ndarray:
        keys = []
        for bank in banks:
            for row in bank.truth.fillna(0).itertuples(index=False):
                keys.append((
                    bank.name, int(row.VOICE_PRESENT), int(row.MUSIC_PRESENT),
                    int(row.VOICE_FAKE), int(row.MUSIC_FAKE), int(row.FILE_FAKE),
                ))
        counts = Counter(keys)
        strata_per_bank = Counter(key[0] for key in counts)
        result = np.asarray([
            1 / (counts[key] * strata_per_bank[key[0]]) for key in keys
        ], dtype=np.float64)
        return result / result.mean()

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, index: int):
        return (
            self.latent[index], self.logits[index], self.targets[index],
            self.masks[index], self.weight[index], self.bank[index],
        )


@torch.inference_mode()
def predict_router(
    models: list[BoundedLatentExpertRouter], bank: Bank,
    device: torch.device, batch_size: int = 1024, hard: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    all_logits, all_weights = [], []
    for model in models:
        current_logits, current_weights = [], []
        model.eval()
        for offset in range(0, len(bank.ids), batch_size):
            latent = torch.from_numpy(bank.latent[offset:offset + batch_size]).to(device)
            expert = torch.from_numpy(bank.expert_logits[offset:offset + batch_size]).to(device)
            logits, weights = model(latent, expert)
            current_logits.append(logits.cpu().numpy())
            current_weights.append(weights.cpu().numpy())
        all_logits.append(np.concatenate(current_logits))
        all_weights.append(np.concatenate(current_weights))
    weights = np.mean(all_weights, axis=0)
    if hard:
        selected = np.argmax(weights, axis=-1)
        weights = np.eye(bank.expert_logits.shape[1], dtype=np.float32)[selected]
        logits = (
            weights * bank.expert_logits.transpose(0, 2, 1)
        ).sum(-1)
    else:
        logits = np.mean(all_logits, axis=0)
    return logits, weights


def evaluate_models(
    models: list[BoundedLatentExpertRouter], banks: list[Bank],
    device: torch.device, hard: bool = False,
) -> tuple[pd.DataFrame, float]:
    rows = []
    for bank in banks:
        logits, _ = predict_router(models, bank, device, hard=hard)
        rows.append({"DATASET": bank.name, **bank_metrics(bank, logits)})
    frame = pd.DataFrame(rows)
    selection = music_selection(frame)
    return frame, float(selection)


def train_one(
    dataset: RouterDataset, dev_banks: list[Bank], prior: np.ndarray,
    strength: float, seed: int, device: torch.device,
    epochs: int, patience: int, samples_per_epoch: int, batch_size: int,
    expert_latent_widths: tuple[int, ...] | None = None,
    global_latent_width: int = 0,
) -> tuple[BoundedLatentExpertRouter, dict, pd.DataFrame]:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    sampler = WeightedRandomSampler(
        torch.from_numpy(dataset.weight), num_samples=samples_per_epoch,
        replacement=True, generator=torch.Generator().manual_seed(seed),
    )
    loader = DataLoader(dataset, batch_size=batch_size, sampler=sampler)
    model = BoundedLatentExpertRouter(
        dataset.latent.shape[1], torch.from_numpy(prior),
        strength=strength,
        expert_latent_widths=expert_latent_widths,
        global_latent_width=global_latent_width,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=8e-4, weight_decay=2e-2)
    best_state, best_score, best_epoch, stale = None, -np.inf, -1, 0
    history = []
    for epoch in range(epochs):
        model.train()
        losses = []
        for latent, expert, target, mask, sample_weight, _bank in loader:
            latent, expert = latent.to(device), expert.to(device)
            target, mask = target.to(device), mask.to(device)
            sample_weight = sample_weight.to(device)[:, None]
            fused, weights = model(latent, expert)
            all_task_ranking = pairwise_ranking_loss(fused, target, mask)
            music_valid = mask[:, 1].bool()
            music_positive = fused[music_valid & target[:, 1].eq(1), 1]
            music_negative = fused[music_valid & target[:, 1].eq(0), 1]
            music_ranking = (
                F.softplus(
                    -(music_positive[:, None] - music_negative[None, :])
                ).mean()
                if len(music_positive) and len(music_negative)
                else fused.sum() * 0
            )
            ranking = .25 * all_task_ranking + .75 * music_ranking
            raw = F.binary_cross_entropy_with_logits(fused, target, reduction="none")
            task_importance = torch.tensor(
                [.10, .70, .20], device=device
            )[None]
            weighted_mask = mask.float() * sample_weight * task_importance
            bce = (raw * weighted_mask).sum() / weighted_mask.sum().clamp_min(1)
            prior_tensor = model.prior[None]
            regularization = ((weights - prior_tensor) ** 2).mean()
            loss = ranking + .25 * bce + .10 * regularization
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 3.0)
            optimizer.step()
            losses.append(float(loss.detach()))
        dev_metrics, score = evaluate_models([model], dev_banks, device)
        history.append({
            "EPOCH": epoch, "LOSS": np.mean(losses), "SELECTION": score,
            "MEAN_ADS": dev_metrics.ADS.mean(), "WORST_ADS": dev_metrics.ADS.min(),
        })
        if score > best_score + 1e-5:
            best_state, best_score, best_epoch = (
                copy.deepcopy(model.state_dict()), score, epoch
            )
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                break
    if best_state is None:
        raise RuntimeError("router produced no checkpoint")
    model.load_state_dict(best_state)
    model.eval()
    return model, {
        "STRENGTH": strength, "SEED": seed, "BEST_EPOCH": best_epoch,
        "SELECTION": best_score,
    }, pd.DataFrame(history)


def locked_outer_music_audit(
    models: list[BoundedLatentExpertRouter], banks: list[Bank],
    prior: np.ndarray, joint_unified: pd.DataFrame, device: torch.device,
    hard: bool,
) -> pd.DataFrame:
    paths = {
        "factorial_eval_1200_v2_holdout": "factorial",
        "phone_factorial_1200_v1": "phone",
        "yue_cross_component_audit_v1": "yue",
    }
    root = ROOT / "reports/segmental_eat_music_v2/nested_v36b_locked"
    rows = []
    for bank in banks:
        short = paths[bank.name]
        anchor = pd.read_csv(
            root / f"{short}_predictions.csv", dtype={"ID": str}
        ).set_index("ID").loc[bank.ids]
        truth = pd.read_csv(
            root / f"{short}_truth.csv", dtype={"ID": str}
        ).set_index("ID").loc[bank.ids]
        keys = pd.MultiIndex.from_arrays((
            np.full(len(bank.ids), bank.name), bank.ids
        ), names=("DATASET", "ID"))
        joint_music = logit(
            joint_unified.loc[keys, "MUSIC_FAKE_PROB"].to_numpy(np.float64)
        )
        fixed = fixed_logits(bank, prior)
        routed, _ = predict_router(models, bank, device, hard=hard)
        candidates = {
            "v37_anchor": logit(anchor.MUSIC_FAKE_PROB),
            "v38_unified_w20": (
                .8 * logit(anchor.MUSIC_FAKE_PROB)
                + .2 * joint_music
            ),
            "fixed_distinct_moe_w20": (
                .8 * logit(anchor.MUSIC_FAKE_PROB) + .2 * fixed[:, 1]
            ),
            ("hard_router_w20" if hard else "latent_router_w20"): (
                .8 * logit(anchor.MUSIC_FAKE_PROB) + .2 * routed[:, 1]
            ),
        }
        for method, music_logit in candidates.items():
            prediction = anchor.copy()
            prediction["MUSIC_FAKE_PROB"] = sigmoid(music_logit)
            result = score_frame(truth.join(prediction))
            rows.append({"METHOD": method, "DATASET": short, **result})
    frame = pd.DataFrame(rows)
    baseline = frame.loc[frame.METHOD.eq("v38_unified_w20")].set_index("DATASET").ADS
    frame["DELTA_VS_V38"] = [row.ADS - baseline[row.DATASET] for row in frame.itertuples()]
    return frame


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wpt-root", type=Path, required=True)
    parser.add_argument("--unified-predictions", type=Path, required=True)
    parser.add_argument(
        "--unified-latent-root", type=Path,
        help="Optional task-wise internal embeddings from the unified expert.",
    )
    parser.add_argument(
        "--joint-unified-predictions", type=Path, required=True,
        help="Full EAT+SPEAR predictions used by the deployed v38 baseline.",
    )
    parser.add_argument(
        "--phone-predictions", type=Path,
        help="Optional file-local deployed phone-router scores appended to the latent.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--train-banks", nargs="+", default=list(TRAIN_BANKS))
    parser.add_argument("--dev-banks", nargs="+", default=list(DEV_BANKS))
    parser.add_argument("--locked-banks", nargs="+", default=list(LOCKED_BANKS))
    parser.add_argument("--strengths", type=float, nargs="+", default=[.10, .25, .50, 1.0])
    parser.add_argument("--seeds", type=int, nargs="+", default=[20260905, 20260906, 20260907])
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--samples-per-epoch", type=int, default=6000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    for name in args.train_banks:
        assert_no_locked_eval_leakage(
            truth_path(name), ROOT / "configs/data_partitions.yaml"
        )
    wpt = indexed_predictions(args.wpt_root / "predictions.csv")
    unified = indexed_predictions(args.unified_predictions)
    joint_unified = indexed_predictions(args.joint_unified_predictions)
    phone = (
        indexed_predictions(args.phone_predictions)
        if args.phone_predictions is not None else None
    )
    all_names = [*args.train_banks, *args.dev_banks, *args.locked_banks]
    banks = {
        name: load_bank(
            name, args.wpt_root, wpt, unified, phone,
            args.unified_latent_root,
        )
        for name in all_names
    }
    train = RouterDataset([banks[name] for name in args.train_banks])
    dev = [banks[name] for name in args.dev_banks]
    locked = [banks[name] for name in args.locked_banks]
    prior = choose_prior(dev, np.linspace(.05, .95, 19))
    expert_latent_widths = None
    global_latent_width = 0
    if args.unified_latent_root is not None:
        first_name = args.train_banks[0]
        unified_width = int(np.load(
            args.unified_latent_root / f"{first_name}_latent.npz",
            allow_pickle=False,
        )["latent"].shape[1])
        wpt_width = int(np.load(
            args.wpt_root / f"{first_name}_latent.npz",
            allow_pickle=False,
        )["latent"].shape[1])
        expert_latent_widths = (unified_width, wpt_width)
        global_latent_width = (
            train.latent.shape[1] - unified_width - wpt_width
        )
    device = torch.device(args.device)
    fixed_rows = [
        {"DATASET": bank.name, **bank_metrics(bank, fixed_logits(bank, prior))}
        for bank in dev
    ]
    fixed_dev = pd.DataFrame(fixed_rows)
    fixed_selection = music_selection(fixed_dev)

    runs, candidates, histories = [], {}, {}
    for strength in args.strengths:
        candidates[strength] = []
        for seed in args.seeds:
            model, record, history = train_one(
                train, dev, prior, strength, seed, device,
                args.epochs, args.patience, args.samples_per_epoch, args.batch_size,
                expert_latent_widths=expert_latent_widths,
                global_latent_width=global_latent_width,
            )
            runs.append(record)
            candidates[strength].append(model)
            histories[(strength, seed)] = history
            print(json.dumps(record), flush=True)
    ensemble_rows = [{
        "METHOD": "fixed_moe", "STRENGTH": 0.0,
        "SELECTION": fixed_selection, "MEAN_ADS": fixed_dev.ADS.mean(),
        "WORST_ADS": fixed_dev.ADS.min(), "DELTA_VS_FIXED": 0.0,
    }]
    for strength, models in candidates.items():
        for method, hard in (("soft_router", False), ("hard_router", True)):
            metric, selection = evaluate_models(models, dev, device, hard=hard)
            ensemble_rows.append({
                "METHOD": method, "STRENGTH": strength,
                "SELECTION": selection,
                "MEAN_ADS": metric.ADS.mean(), "WORST_ADS": metric.ADS.min(),
                "DELTA_VS_FIXED": selection - fixed_selection,
            })
    ensemble = pd.DataFrame(ensemble_rows).sort_values("SELECTION", ascending=False)
    selected_method = str(ensemble.iloc[0].METHOD)
    selected_strength = float(ensemble.iloc[0].STRENGTH)
    selected = (
        candidates[selected_strength] if selected_method != "fixed_moe" else []
    )
    selected_hard = selected_method == "hard_router"
    if selected:
        selected_dev, selected_score = evaluate_models(
            selected, dev, device, hard=selected_hard
        )
        selected_locked, _ = evaluate_models(
            selected, locked, device, hard=selected_hard
        )
    else:
        selected_dev, selected_score = fixed_dev.copy(), fixed_selection
        selected_locked = pd.DataFrame([
            {"DATASET": bank.name, **bank_metrics(bank, fixed_logits(bank, prior))}
            for bank in locked
        ])
    fixed_locked = pd.DataFrame([
        {"DATASET": bank.name, **bank_metrics(bank, fixed_logits(bank, prior))}
        for bank in locked
    ])
    # A fixed winner requires no learned routing at all.  Reuse one trained
    # model only to keep this diagnostic table rectangular; its output is not
    # selected or deployed in that case.
    outer_models = selected or candidates[min(candidates)]
    outer = locked_outer_music_audit(
        outer_models, locked, prior, joint_unified, device,
        hard=selected_hard,
    )
    if selected_method == "fixed_moe":
        fixed_rows = outer.loc[outer.METHOD.eq("fixed_distinct_moe_w20")].copy()
        fixed_rows["METHOD"] = "selected_fixed_moe_w20"
        outer = pd.concat((outer, fixed_rows), ignore_index=True)

    # Persist file-level outputs for the next, completely axis-controlled
    # comparison against v38.  This lets the same evaluator test whether the
    # selected standalone expert (fixed MoE or router) is useful for Voice and
    # File without retraining or reopening labels.
    prediction_rows, fixed_prediction_rows = [], []
    for bank in banks.values():
        fixed_bank_logits = fixed_logits(bank, prior)
        fixed_probability = sigmoid(fixed_bank_logits)
        fixed_prediction_rows.append(pd.DataFrame({
            "DATASET": bank.name,
            "ID": bank.ids,
            "VOICE_FAKE_PROB": fixed_probability[:, 0],
            "MUSIC_FAKE_PROB": fixed_probability[:, 1],
            "FILE_FAKE_PROB": fixed_probability[:, 2],
        }))
        if selected:
            logits, weights = predict_router(
                selected, bank, device, hard=selected_hard
            )
        else:
            logits = fixed_logits(bank, prior)
            weights = np.broadcast_to(
                prior[None], (len(bank.ids), *prior.shape)
            ).copy()
        probability = sigmoid(logits)
        prediction_rows.append(pd.DataFrame({
            "DATASET": bank.name,
            "ID": bank.ids,
            "VOICE_FAKE_PROB": probability[:, 0],
            "MUSIC_FAKE_PROB": probability[:, 1],
            "FILE_FAKE_PROB": probability[:, 2],
            "VOICE_UNIFIED_WEIGHT": weights[:, 0, 0],
            "VOICE_WPT_WEIGHT": weights[:, 0, 1],
            "MUSIC_UNIFIED_WEIGHT": weights[:, 1, 0],
            "MUSIC_WPT_WEIGHT": weights[:, 1, 1],
            "FILE_UNIFIED_WEIGHT": weights[:, 2, 0],
            "FILE_WPT_WEIGHT": weights[:, 2, 1],
        }))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(runs).to_csv(args.output_dir / "member_runs.csv", index=False)
    ensemble.to_csv(args.output_dir / "dev_strength_sweep.csv", index=False)
    fixed_dev.assign(METHOD="fixed_moe").to_csv(
        args.output_dir / "fixed_dev.csv", index=False
    )
    selected_dev.assign(METHOD=selected_method).to_csv(
        args.output_dir / "routed_dev.csv", index=False
    )
    pd.concat((
        fixed_locked.assign(METHOD="fixed_moe"),
        selected_locked.assign(METHOD=selected_method),
    )).to_csv(args.output_dir / "locked_standalone.csv", index=False)
    pd.concat(prediction_rows, ignore_index=True).to_csv(
        args.output_dir / "selected_predictions.csv", index=False
    )
    pd.concat(fixed_prediction_rows, ignore_index=True).to_csv(
        args.output_dir / "fixed_predictions.csv", index=False
    )
    outer.to_csv(args.output_dir / "locked_outer_music.csv", index=False)
    for index, model in enumerate(selected):
        torch.save({
            "model": {key: value.cpu() for key, value in model.state_dict().items()},
            "config": {
                "latent_width": train.latent.shape[1], "prior": prior,
                "strength": selected_strength, "model_width": 48,
                "heads": 4, "layers": 1, "dropout": .15, "temperature": 1.5,
                "expert_latent_widths": expert_latent_widths,
                "global_latent_width": global_latent_width,
            },
            "train_banks": list(args.train_banks),
            "dev_banks": list(args.dev_banks),
        }, args.output_dir / f"latent_router_{index:02d}.pt")
    summary = {
        "prior_unified_wpt": prior.tolist(),
        "fixed_dev_selection": float(fixed_selection),
        "selected_method": selected_method,
        "selected_strength": selected_strength,
        "selected_dev_selection": selected_score,
        "locked_outer_min_delta_vs_v38": float(
            outer.loc[
                outer.METHOD.eq(
                    "selected_fixed_moe_w20" if selected_method == "fixed_moe"
                    else ("hard_router_w20" if selected_hard else "latent_router_w20")
                ), "DELTA_VS_V38"
            ].min()
        ),
        "locked_outer_mean_delta_vs_v38": float(
            outer.loc[
                outer.METHOD.eq(
                    "selected_fixed_moe_w20" if selected_method == "fixed_moe"
                    else ("hard_router_w20" if selected_hard else "latent_router_w20")
                ), "DELTA_VS_V38"
            ].mean()
        ),
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print("\nDEV SWEEP\n" + ensemble.to_string(index=False))
    print("\nLOCKED OUTER MUSIC\n" + outer.to_string(index=False))
    print("\nSUMMARY\n" + json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
