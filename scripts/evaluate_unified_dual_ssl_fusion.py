#!/usr/bin/env python3
"""Compare Music-specialist routers with a fixed, conservative soft MoE.

The anchor is the locked v37 prediction.  The candidate expert is the
separation-free unified EAT/SPEAR head.  Only ``MUSIC_FAKE_PROB`` is allowed to
change: the expert's held-out results show that its Voice and File heads are
not uniformly better than v37.  This makes the comparison axis-exact and
prevents a router from hiding regressions in unrelated outputs.
"""

from __future__ import annotations

import argparse
from itertools import product
from pathlib import Path
import sys

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from evaluate_diagnostic import official_eer, score_frame  # noqa: E402


BANKS = {
    "factorial": {
        "dataset": "factorial_eval_1200_v2_holdout",
        "anchor": ROOT / "reports/segmental_eat_music_v2/nested_v36b_locked/factorial_predictions.csv",
        "truth": ROOT / "reports/segmental_eat_music_v2/nested_v36b_locked/factorial_truth.csv",
        "phone": ROOT / "reports/phone_presence_probe_v1/factorial_eval_1200_v2_router.csv",
    },
    "phone": {
        "dataset": "phone_factorial_1200_v1",
        "anchor": ROOT / "reports/segmental_eat_music_v2/nested_v36b_locked/phone_predictions.csv",
        "truth": ROOT / "reports/segmental_eat_music_v2/nested_v36b_locked/phone_truth.csv",
        "phone": ROOT / "reports/phone_presence_probe_v1/phone_factorial_1200_v1_router.csv",
    },
    "yue": {
        "dataset": "yue_cross_component_audit_v1",
        "anchor": ROOT / "reports/segmental_eat_music_v2/nested_v36b_locked/yue_predictions.csv",
        "truth": ROOT / "reports/segmental_eat_music_v2/nested_v36b_locked/yue_truth.csv",
        "phone": None,
    },
}


def logit(values) -> np.ndarray:
    values = np.clip(np.asarray(values, dtype=np.float64), 1e-5, 1 - 1e-5)
    return np.log(values) - np.log1p(-values)


def sigmoid(values) -> np.ndarray:
    return np.exp(-np.logaddexp(0, -np.asarray(values, dtype=np.float64)))


def indexed(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, dtype={"ID": str}).set_index("ID")
    if frame.index.duplicated().any():
        raise ValueError(f"duplicate ID in {path}")
    return frame


def load_banks(expert_path: Path) -> dict[str, dict[str, object]]:
    expert_all = pd.read_csv(expert_path, dtype={"ID": str})
    if expert_all.duplicated(["DATASET", "ID"]).any():
        raise ValueError(f"duplicate expert IDs in {expert_path}")
    result = {}
    for name, paths in BANKS.items():
        truth = indexed(paths["truth"])
        anchor = indexed(paths["anchor"]).loc[truth.index]
        expert = (
            expert_all.loc[expert_all.DATASET.eq(paths["dataset"])]
            .set_index("ID").loc[truth.index]
        )
        if paths["phone"] is None:
            phone_probability = np.zeros(len(truth), dtype=np.float64)
        else:
            phone_probability = indexed(paths["phone"]).loc[
                truth.index, "PHONE_PROB"
            ].to_numpy(np.float64)
        result[name] = {
            "truth": truth,
            "anchor": anchor,
            "expert": expert,
            "phone_probability": phone_probability,
        }
    return result


def routed_weight(
    bank: dict[str, object], strategy: str, nonphone: float, phone: float
) -> np.ndarray:
    anchor = bank["anchor"]
    expert = bank["expert"]
    phone_probability = bank["phone_probability"]
    count = len(anchor)
    if strategy == "fixed_soft_moe":
        return np.full(count, nonphone)
    if strategy == "phone_soft_router":
        return nonphone + (phone - nonphone) * phone_probability
    if strategy == "phone_hard_router":
        gate = phone_probability >= .5
        return nonphone + (phone - nonphone) * gate
    anchor_logit = logit(anchor.MUSIC_FAKE_PROB)
    expert_logit = logit(expert.MUSIC_FAKE_PROB)
    if strategy == "confidence_hard_router":
        return (np.abs(expert_logit) > np.abs(anchor_logit)).astype(np.float64)
    if strategy == "confidence_soft_router":
        # A bounded gate: even a confident specialist cannot fully replace the
        # more general anchor.  ``phone`` is used as the maximum expert weight.
        confidence = sigmoid(np.abs(expert_logit) - np.abs(anchor_logit))
        return nonphone + (phone - nonphone) * confidence
    if strategy == "mixed_hard_router":
        mixed = (
            anchor.VOICE_PRESENT_PROB.to_numpy(np.float64) >= .7
        ) & (
            anchor.MUSIC_PRESENT_PROB.to_numpy(np.float64) >= .7
        )
        return np.where(mixed, phone, nonphone)
    raise ValueError(strategy)


def evaluate(
    banks: dict[str, dict[str, object]], strategy: str,
    nonphone: float, phone: float,
) -> tuple[list[dict[str, object]], list[np.ndarray]]:
    rows, predictions = [], []
    for name, bank in banks.items():
        truth = bank["truth"]
        anchor = bank["anchor"]
        expert = bank["expert"]
        weight = routed_weight(bank, strategy, nonphone, phone)
        prediction = anchor.copy()
        prediction["MUSIC_FAKE_PROB"] = sigmoid(
            (1 - weight) * logit(anchor.MUSIC_FAKE_PROB)
            + weight * logit(expert.MUSIC_FAKE_PROB)
        )
        metrics = score_frame(truth.join(prediction))
        rows.append({
            "DATASET": name,
            "STRATEGY": strategy,
            "NONPHONE_WEIGHT": nonphone,
            "PHONE_WEIGHT": phone,
            "MEAN_EXPERT_WEIGHT": float(weight.mean()),
            **metrics,
        })
        predictions.append(prediction.MUSIC_FAKE_PROB.to_numpy(np.float64))
    return rows, predictions


def music_scores(
    bank: dict[str, object], strategy: str, nonphone: float, phone: float,
) -> np.ndarray:
    anchor = bank["anchor"]
    expert = bank["expert"]
    weight = routed_weight(bank, strategy, nonphone, phone)
    return sigmoid(
        (1 - weight) * logit(anchor.MUSIC_FAKE_PROB)
        + weight * logit(expert.MUSIC_FAKE_PROB)
    )


def robustness_diagnostics(
    banks: dict[str, dict[str, object]], output_dir: Path,
    bootstrap_repetitions: int,
) -> None:
    """Measure whether a small routing gain survives cells and resampling."""
    candidates = {
        "anchor": ("fixed_soft_moe", 0, 0),
        "fixed_moe_020": ("fixed_soft_moe", .20, .20),
        "mixed_router_030_050": ("mixed_hard_router", .30, .50),
        "confidence_hard": ("confidence_hard_router", 0, 1),
    }
    grouped_rows = []
    for bank_name, bank in banks.items():
        truth = bank["truth"]
        for candidate, setting in candidates.items():
            scores = music_scores(bank, *setting)
            for group_column in ("AUDIO_TYPE", "MIX_MODE", "CONDITION", "CODEC"):
                if group_column not in truth:
                    continue
                for value, indices in truth.groupby(group_column, dropna=False).groups.items():
                    selected = truth.index.get_indexer(indices)
                    present = truth.iloc[selected].MUSIC_PRESENT.eq(1).to_numpy()
                    selected = selected[present]
                    labels = truth.iloc[selected].MUSIC_FAKE.astype(int)
                    if len(selected) < 8 or labels.nunique() < 2:
                        continue
                    grouped_rows.append({
                        "DATASET": bank_name,
                        "CANDIDATE": candidate,
                        "GROUP": group_column,
                        "VALUE": str(value),
                        "N": len(selected),
                        "MUSIC_EER": official_eer(labels, scores[selected]),
                    })
    pd.DataFrame(grouped_rows).to_csv(
        output_dir / "grouped_music_eer.csv", index=False
    )

    if bootstrap_repetitions <= 0:
        return
    rng = np.random.default_rng(20260904)
    bootstrap_rows = []
    for bank_name, bank in banks.items():
        truth = bank["truth"]
        present = truth.MUSIC_PRESENT.eq(1).to_numpy()
        labels = truth.loc[present, "MUSIC_FAKE"].astype(int).to_numpy()
        score_by_candidate = {
            name: music_scores(bank, *setting)[present]
            for name, setting in candidates.items()
        }
        strata = [np.flatnonzero(labels == value) for value in (0, 1)]
        deltas = {name: [] for name in candidates if name != "anchor"}
        for _ in range(bootstrap_repetitions):
            sampled = np.concatenate([
                rng.choice(indices, size=len(indices), replace=True)
                for indices in strata
            ])
            anchor_eer = official_eer(labels[sampled], score_by_candidate["anchor"][sampled])
            for name in deltas:
                candidate_eer = official_eer(
                    labels[sampled], score_by_candidate[name][sampled]
                )
                # Positive is better. Only the 0.3 Music contribution changes.
                deltas[name].append(.3 * (anchor_eer - candidate_eer))
        for name, values in deltas.items():
            values = np.asarray(values)
            bootstrap_rows.append({
                "DATASET": bank_name,
                "CANDIDATE": name,
                "ADS_DELTA_MEAN": float(values.mean()),
                "ADS_DELTA_P025": float(np.quantile(values, .025)),
                "ADS_DELTA_P975": float(np.quantile(values, .975)),
                "P_IMPROVES": float((values > 0).mean()),
            })
    pd.DataFrame(bootstrap_rows).to_csv(
        output_dir / "paired_bootstrap.csv", index=False
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--expert", type=Path,
        default=ROOT / "reports/unified_dual_ssl_v1/ensemble_locked/predictions.csv",
    )
    parser.add_argument(
        "--weights", type=float, nargs="+",
        default=[0, .025, .05, .075, .10, .15, .20, .25, .30, .40, .50, 1.0],
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=ROOT / "reports/unified_dual_ssl_v1/router_vs_moe",
    )
    parser.add_argument("--bootstrap-repetitions", type=int, default=500)
    args = parser.parse_args()
    banks = load_banks(args.expert)

    settings: list[tuple[str, float, float]] = []
    settings.extend(("fixed_soft_moe", weight, weight) for weight in args.weights)
    for nonphone, phone in product(args.weights[:-1], args.weights[:-1]):
        settings.append(("phone_soft_router", nonphone, phone))
        settings.append(("phone_hard_router", nonphone, phone))
        settings.append(("mixed_hard_router", nonphone, phone))
    settings.extend([
        ("confidence_hard_router", 0, 1),
        ("confidence_soft_router", 0, .25),
        ("confidence_soft_router", .05, .30),
        ("confidence_soft_router", .10, .40),
    ])

    rows = []
    for strategy, nonphone, phone in settings:
        current, _ = evaluate(banks, strategy, nonphone, phone)
        rows.extend(current)
    per_dataset = pd.DataFrame(rows)
    anchor = (
        per_dataset.loc[
            per_dataset.STRATEGY.eq("fixed_soft_moe")
            & per_dataset.NONPHONE_WEIGHT.eq(0)
        ].set_index("DATASET").ADS
    )
    per_dataset["ADS_DELTA_VS_ANCHOR"] = [
        row.ADS - anchor[row.DATASET] for row in per_dataset.itertuples()
    ]
    group = ["STRATEGY", "NONPHONE_WEIGHT", "PHONE_WEIGHT"]
    summary = per_dataset.groupby(group, sort=False).agg(
        MEAN_ADS=("ADS", "mean"),
        WORST_ADS=("ADS", "min"),
        MIN_DELTA=("ADS_DELTA_VS_ANCHOR", "min"),
        MEAN_DELTA=("ADS_DELTA_VS_ANCHOR", "mean"),
        MAX_MUSIC_EER=("MUSIC_EER", "max"),
    ).reset_index()
    summary["ROBUST_SELECTION"] = (
        summary.MEAN_ADS + summary.WORST_ADS + summary.MIN_DELTA
    )
    summary = summary.sort_values(
        ["MIN_DELTA", "MEAN_DELTA", "WORST_ADS"], ascending=False
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    per_dataset.to_csv(args.output_dir / "per_dataset.csv", index=False)
    summary.to_csv(args.output_dir / "summary.csv", index=False)
    robustness_diagnostics(
        banks, args.output_dir, args.bootstrap_repetitions
    )
    print("Top configurations (prioritizing no held-out regression):")
    print(summary.head(20).round(6).to_string(index=False))
    selected = summary.iloc[0]
    selection_mask = np.logical_and.reduce([
        per_dataset.STRATEGY.eq(selected.STRATEGY),
        per_dataset.NONPHONE_WEIGHT.eq(selected.NONPHONE_WEIGHT),
        per_dataset.PHONE_WEIGHT.eq(selected.PHONE_WEIGHT),
    ])
    detail = per_dataset.loc[selection_mask, [
        "DATASET", "ADS", "ADS_DELTA_VS_ANCHOR", "FILE_EER",
        "VOICE_EER", "MUSIC_EER", "MEAN_EXPERT_WEIGHT",
    ]]
    print("\nSelected per-dataset result:")
    print(detail.round(6).to_string(index=False))


if __name__ == "__main__":
    main()
