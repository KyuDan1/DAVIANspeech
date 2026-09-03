#!/usr/bin/env python3
"""Train a tiny content-aware gate between v29 and the joint SPEAR expert.

Training uses clean/telephone pairs from three development sources.  The
factorial development split selects hyperparameters; factorial holdout,
phone-factorial, and YuE remain evaluation-only.  The gate sees only scores
that are available during inference and never dataset IDs or ground truth.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import random
import sys

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from evaluate_diagnostic import official_eer, score_frame  # noqa: E402
from evaluate_joint_domain_weight_router import (  # noqa: E402
    EXPERT_DATASETS, ROUTERS, TRUTHS, indexed,
)


PREDICTION_COLUMNS = [
    "FILE_FAKE_PROB", "VOICE_FAKE_PROB", "MUSIC_FAKE_PROB",
    "VOICE_PRESENT_PROB", "MUSIC_PRESENT_PROB",
]
TRAIN_DATASETS = [
    "external_mixed_v1", "source_disjoint_mixed_v1",
    "source_disjoint_mixed_equal_v1",
    "external_mixed_v1_telephone_v1",
    "source_disjoint_mixed_v1_telephone_v1",
    "source_disjoint_mixed_equal_v1_telephone_v1",
]


def logit(values) -> np.ndarray:
    values = np.clip(np.asarray(values, dtype=np.float64), 1e-5, 1 - 1e-5)
    return np.log(values) - np.log1p(-values)


def sigmoid(values) -> np.ndarray:
    return np.exp(-np.logaddexp(0, -np.asarray(values, dtype=np.float64)))


def joint_predictions() -> pd.DataFrame:
    regular = pd.read_csv(
        ROOT / "reports/spear_temporal_joint_v1/seed09/dev_predictions.csv",
        dtype={"ID": str},
    )
    equal_phone = pd.read_csv(
        ROOT / "reports/router_training_v1/equal_phone_joint.csv",
        dtype={"ID": str},
    )
    equal_phone.insert(0, "DATASET", "equal_phone")
    return pd.concat([regular, equal_phone], ignore_index=True)


def training_bank(joint: pd.DataFrame, name: str) -> dict:
    truth = indexed(ROOT / "data/eval" / name / "truth.csv")
    final = indexed(
        ROOT / "scratch/v30_router_bank_20260903" / name
        / "output/submission.csv"
    ).loc[truth.index]
    phone = name.endswith("_telephone_v1")
    if name == "source_disjoint_mixed_equal_v1_telephone_v1":
        expert = joint.loc[joint.DATASET.eq("equal_phone")].set_index("ID")
    else:
        dataset = "telephone_mixed_dev_v1" if phone else name
        expert = joint.loc[joint.DATASET.eq(dataset)].set_index("ID")
    expert = expert.loc[truth.index]
    anchor = final.copy()
    for column in ("FILE_FAKE_PROB", "VOICE_FAKE_PROB"):
        anchor[column] = sigmoid(
            (logit(final[column]) - .25 * logit(expert[column])) / .75
        )
    return {
        "truth": truth, "anchor": anchor, "expert": expert,
        "phone": np.full(len(truth), float(phone)), "dataset": name,
    }


def audit_bank(v29: pd.DataFrame, joint: pd.DataFrame, name: str) -> dict:
    truth = indexed(TRUTHS[name])
    anchor = (
        v29.loc[v29.DATASET.eq(name) & v29.METHOD.eq("v29")]
        .drop(columns=["DATASET", "METHOD"]).set_index("ID").loc[truth.index]
    )
    expert = (
        joint.loc[joint.DATASET.eq(EXPERT_DATASETS[name])]
        .set_index("ID").loc[truth.index]
    )
    if name == "yue":
        phone = np.zeros(len(truth), dtype=np.float64)
    else:
        phone = indexed(ROUTERS[name]).loc[
            truth.index, "PHONE_PROB"
        ].to_numpy(np.float64)
    return {
        "truth": truth, "anchor": anchor, "expert": expert,
        "phone": phone, "dataset": name,
    }


def features(bank: dict) -> np.ndarray:
    anchor = bank["anchor"][PREDICTION_COLUMNS].to_numpy(np.float64)
    expert = bank["expert"][PREDICTION_COLUMNS].to_numpy(np.float64)
    anchor_logit, expert_logit = logit(anchor), logit(expert)
    difference = expert_logit - anchor_logit
    phone = bank["phone"].reshape(-1, 1)
    # The interaction lets one shared gate adapt to telephone bandwidth while
    # remaining content-aware inside each domain.
    return np.concatenate([
        anchor_logit, expert_logit, difference, np.abs(difference),
        phone, phone * difference,
    ], axis=1).astype(np.float32)


def balanced_weights(banks: list[dict], axis: str) -> np.ndarray:
    values = []
    target_name = f"{axis}_FAKE"
    present_name = None if axis == "FILE" else f"{axis}_PRESENT"
    for bank in banks:
        truth = bank["truth"]
        selected = np.ones(len(truth), dtype=bool)
        if present_name:
            selected = truth[present_name].eq(1).to_numpy()
        label = truth.loc[selected, target_name].astype(int).to_numpy()
        count = np.bincount(label, minlength=2).clip(min=1)
        local = np.zeros(len(truth), dtype=np.float32)
        local[selected] = 1 / count[label] / len(banks)
        values.append(local)
    result = np.concatenate(values)
    return result / result[result > 0].mean()


class Router(nn.Module):
    def __init__(self, dimension: int, hidden: int, low: float, high: float):
        super().__init__()
        self.low, self.high = float(low), float(high)
        if hidden:
            self.network = nn.Sequential(
                nn.Linear(dimension, hidden), nn.Tanh(), nn.Linear(hidden, 2),
            )
        else:
            self.network = nn.Linear(dimension, 2)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        gate = self.network(value).sigmoid()
        return self.low + (self.high - self.low) * gate


def predict(model: Router, x: np.ndarray, bank: dict) -> pd.DataFrame:
    with torch.inference_mode():
        weights = model(torch.from_numpy(x)).cpu().numpy()
    output = bank["anchor"].copy()
    for index, axis in enumerate(("FILE", "VOICE")):
        column = f"{axis}_FAKE_PROB"
        output[column] = sigmoid(
            (1 - weights[:, index]) * logit(bank["anchor"][column])
            + weights[:, index] * logit(bank["expert"][column])
        )
    return output


def validation_loss(model: Router, x: np.ndarray, bank: dict) -> float:
    output = predict(model, x, bank)
    truth = bank["truth"]
    file_eer = official_eer(truth.FILE_FAKE.astype(int), output.FILE_FAKE_PROB)
    voice = truth.VOICE_PRESENT.eq(1)
    voice_eer = official_eer(
        truth.loc[voice, "VOICE_FAKE"].astype(int),
        output.loc[voice, "VOICE_FAKE_PROB"],
    )
    return .5 * file_eer + .2 * voice_eer


def train_one(
    train_banks: list[dict], validation: dict, hidden: int,
    low: float, high: float, regularization: float, seed: int,
) -> tuple[Router, float, int, np.ndarray, np.ndarray]:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    raw = np.concatenate([features(bank) for bank in train_banks])
    mean, std = raw.mean(axis=0), raw.std(axis=0).clip(min=1e-4)
    x = np.clip((raw - mean) / std, -8, 8).astype(np.float32)
    vx = np.clip((features(validation) - mean) / std, -8, 8).astype(np.float32)
    anchor = np.concatenate([
        bank["anchor"][["FILE_FAKE_PROB", "VOICE_FAKE_PROB"]].to_numpy()
        for bank in train_banks
    ])
    expert = np.concatenate([
        bank["expert"][["FILE_FAKE_PROB", "VOICE_FAKE_PROB"]].to_numpy()
        for bank in train_banks
    ])
    labels = np.concatenate([
        bank["truth"][["FILE_FAKE", "VOICE_FAKE"]].to_numpy(np.float32)
        for bank in train_banks
    ])
    voice_present = np.concatenate([
        bank["truth"].VOICE_PRESENT.eq(1).to_numpy() for bank in train_banks
    ])
    sample_weights = np.stack([
        balanced_weights(train_banks, "FILE"),
        balanced_weights(train_banks, "VOICE"),
    ], axis=1)
    sample_weights[~voice_present, 1] = 0
    tx = torch.from_numpy(x)
    ta, te = torch.from_numpy(logit(anchor).astype(np.float32)), torch.from_numpy(logit(expert).astype(np.float32))
    ty, tw = torch.from_numpy(labels), torch.from_numpy(sample_weights)
    model = Router(x.shape[1], hidden, low, high)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=2e-2)
    best_score, best_epoch, best_state, stale = np.inf, -1, None, 0
    for epoch in range(1001):
        model.train(); weight = model(tx); logits = (1 - weight) * ta + weight * te
        loss_matrix = F.binary_cross_entropy_with_logits(logits, ty, reduction="none")
        loss = (loss_matrix * tw).sum() / tw.sum()
        loss = loss + regularization * (weight - .25).square().mean()
        optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step()
        if epoch % 10:
            continue
        model.eval(); score = validation_loss(model, vx, validation)
        if score < best_score - 1e-8:
            best_score, best_epoch = score, epoch
            best_state = copy.deepcopy(model.state_dict()); stale = 0
        else:
            stale += 1
            if stale >= 30:
                break
    if best_state is None:
        raise RuntimeError("router training did not produce a checkpoint")
    model.load_state_dict(best_state); model.eval()
    return model, best_score, best_epoch, mean, std


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir", type=Path,
        default=ROOT / "reports/router_training_v1/content_gate",
    )
    args = parser.parse_args()
    train_joint = joint_predictions()
    train_banks = [training_bank(train_joint, name) for name in TRAIN_DATASETS]
    v29 = pd.read_csv(
        ROOT / "reports/spear_temporal_joint_v1/selected_v29_predictions.csv",
        dtype={"ID": str},
    )
    audit_joint = pd.read_csv(
        ROOT / "reports/spear_temporal_joint_v1/seed09_audit/predictions.csv",
        dtype={"ID": str},
    )
    audits = {
        name: audit_bank(v29, audit_joint, name) for name in TRUTHS
    }
    validation = audits["dev"]
    rows, candidates = [], []
    configurations = [
        (0, .10, .50, .1), (0, .10, .50, 1.0),
        (0, .15, .45, .1), (0, .15, .45, 1.0),
        (12, .10, .50, .1), (12, .10, .50, 1.0),
        (12, .15, .45, .1), (12, .15, .45, 1.0),
    ]
    for hidden, low, high, regularization in configurations:
        for seed in (20260903, 20260904, 20260905):
            model, selection, epoch, mean, std = train_one(
                train_banks, validation, hidden, low, high,
                regularization, seed,
            )
            record = {
                "HIDDEN": hidden, "LOW": low, "HIGH": high,
                "REGULARIZATION": regularization, "SEED": seed,
                "BEST_EPOCH": epoch, "SELECTION": selection,
            }
            for name, bank in audits.items():
                x = np.clip((features(bank) - mean) / std, -8, 8).astype(np.float32)
                metrics = score_frame(bank["truth"].join(predict(model, x, bank)))
                record[f"{name.upper()}_ADS"] = metrics["ADS"]
                record[f"{name.upper()}_FILE_EER"] = metrics["FILE_EER"]
                record[f"{name.upper()}_VOICE_EER"] = metrics["VOICE_EER"]
            rows.append(record)
            candidates.append((selection, record, model, mean, std))
            print(record, flush=True)
    frame = pd.DataFrame(rows).sort_values("SELECTION")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output_dir / "sweep.csv", index=False)
    selection, record, model, mean, std = min(candidates, key=lambda item: item[0])
    torch.save({
        "model": model.state_dict(), "mean": mean, "std": std,
        "config": {key: record[key] for key in (
            "HIDDEN", "LOW", "HIGH", "REGULARIZATION", "SEED",
        )},
    }, args.output_dir / "router.pt")
    (args.output_dir / "selected.json").write_text(
        json.dumps(record, indent=2), encoding="utf-8"
    )
    print("SELECTED", json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
