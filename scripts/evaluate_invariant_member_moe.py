#!/usr/bin/env python3
"""Compare fixed member mixtures and phone routing for the v34 residual."""

from __future__ import annotations

import argparse
from itertools import product
from pathlib import Path
import sys

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from evaluate_codec_invariant_fusion import AUDITS, fuse, reconstruct_v18  # noqa: E402
from evaluate_diagnostic import score_frame  # noqa: E402
from evaluate_presence_weighted_file_fusion import reconstruct_dev_v18  # noqa: E402
from evaluate_v18_attention_music_residuals import (  # noqa: E402
    DATASETS, ROUTERS, load, select,
)


MEMBERS = (
    "seed00_fixedteacher_lr1e4", "seed00_ch01", "seed01_ch01", "seed02_ch01",
)
INVARIANT_DATASETS = {
    "dev": "factorial_eval_1200_v2_dev",
    "factorial": "factorial_eval_1200_v2_holdout",
    "phone": "phone_factorial_1200_v1",
    "yue": "yue_cross_component_audit_v1",
}
FUSION_WEIGHTS = {
    "FILE_FAKE_PROB": .125,
    "VOICE_FAKE_PROB": .20,
    "MUSIC_FAKE_PROB": .10,
}
GENERALIZATION_DATASETS = {
    "mixfake": "mixfake_music_dev_v1",
    "external": "external_mixed_v1",
    "source": "source_disjoint_mixed_v1",
    "equal": "source_disjoint_mixed_equal_v1",
    "factorial_dev": "factorial_eval_1200_v2_dev",
    "telephone_dev": "telephone_mixed_dev_v1",
}


def compositions(units: int):
    for values in product(range(units + 1), repeat=len(MEMBERS)):
        if sum(values) == units:
            yield np.asarray(values, dtype=np.float64) / units


def prediction_path(member: str, bank: str) -> Path:
    partition = "dev" if bank == "dev" else "audit"
    return (
        ROOT / "reports/invariant_dual_domain_v4_paired/router_audit"
        / f"{member}_{partition}" / "predictions.csv"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--units", type=int, default=8)
    parser.add_argument(
        "--output-dir", type=Path,
        default=ROOT / "reports/invariant_dual_domain_v4_paired/member_moe",
    )
    args = parser.parse_args()
    if args.units <= 0:
        parser.error("units must be positive")

    temporal_dev = load(ROOT / "reports/temporal_dual_domain_hybrid/ensemble_dev/predictions.csv")
    temporal_audit = load(ROOT / "reports/temporal_dual_domain_hybrid/ensemble_audit/predictions.csv")
    fakeprint = load(ROOT / "reports/modern_fakeprint_v1/independent_margin_scores.csv")
    invariant_dev = load(ROOT / "reports/invariant_dual_domain_v2/dec_n1/dev_predictions.csv")
    invariant_audit = load(ROOT / "reports/invariant_dual_domain_v2/dec_n1_audit/predictions.csv")
    attention = load(ROOT / "reports/spear_temporal_attention_v1/attention_channel_s12_audit/predictions.csv")
    music = load(ROOT / "reports/spear_temporal_bin_v1/mlp64_seed05_audit/predictions.csv")

    banks = {}
    for name, dataset in DATASETS.items():
        if name == "dev":
            truth, anchor = reconstruct_dev_v18(temporal_dev, fakeprint, invariant_dev)
        else:
            truth, anchor = reconstruct_v18(
                AUDITS[name], temporal_audit, fakeprint, invariant_audit
            )
        phone = np.zeros(len(anchor), dtype=bool)
        if name in ROUTERS:
            phone = (
                load(ROUTERS[name]).set_index("ID")
                .loc[anchor.index, "IS_PHONE"].to_numpy(bool)
            )
        attention_score = select(attention, dataset, anchor.index)
        music_score = select(music, dataset, anchor.index)
        v33 = anchor.copy()
        v33["FILE_FAKE_PROB"] = fuse(
            anchor.FILE_FAKE_PROB, attention_score.FILE_FAKE_PROB,
            np.where(phone, .25, .20),
        )
        v33["MUSIC_FAKE_PROB"] = fuse(
            anchor.MUSIC_FAKE_PROB,
            music_score.TEMPORAL_BIN_MUSIC_PROB, .50,
        )
        member_values = []
        for member in MEMBERS:
            member_values.append(
                select(
                    load(prediction_path(member, name)),
                    INVARIANT_DATASETS[name], anchor.index,
                )[["FILE_FAKE_PROB", "VOICE_FAKE_PROB", "MUSIC_FAKE_PROB"]]
                .to_numpy(np.float64)
            )
        banks[name] = truth, v33, np.stack(member_values)

    baseline = {
        name: score_frame(truth.join(v33))["ADS"]
        for name, (truth, v33, _) in banks.items()
    }
    standalone_banks = {}
    for label, dataset in GENERALIZATION_DATASETS.items():
        truth_path = (
            ROOT / "data/eval/factorial_eval_1200_v2/truth_dev.csv"
            if dataset == "factorial_eval_1200_v2_dev"
            else ROOT / "data/eval" / dataset / "truth.csv"
        )
        truth = load(truth_path).set_index("ID")
        member_values = []
        for member in MEMBERS:
            member_values.append(
                select(
                    load(prediction_path(member, "dev")), dataset, truth.index,
                )[["FILE_FAKE_PROB", "VOICE_FAKE_PROB", "MUSIC_FAKE_PROB"]]
                .to_numpy(np.float64)
            )
        standalone_banks[label] = truth, np.stack(member_values)
    uniform = np.full(len(MEMBERS), 1 / len(MEMBERS))
    standalone_baseline = {}
    for label, (truth, members) in standalone_banks.items():
        expert = np.tensordot(uniform, members, axes=(0, 0))
        prediction = pd.DataFrame(
            expert, index=truth.index, columns=tuple(FUSION_WEIGHTS),
        )
        standalone_baseline[label] = score_frame(truth.join(prediction))["ADS"]

    rows = []
    for weights in compositions(args.units):
        record = {
            f"W_{member}": weight for member, weight in zip(MEMBERS, weights)
        }
        for name, (truth, v33, members) in banks.items():
            expert = np.tensordot(weights, members, axes=(0, 0))
            candidate = v33.copy()
            for index, (column, weight) in enumerate(FUSION_WEIGHTS.items()):
                candidate[column] = fuse(v33[column], expert[:, index], weight)
            ads = score_frame(truth.join(candidate))["ADS"]
            record[f"{name}_ADS"] = ads
            record[f"{name}_DELTA"] = ads - baseline[name]
        deltas = [record[f"{name}_DELTA"] for name in banks]
        record["MIN_DELTA"] = min(deltas)
        record["MEAN_DELTA"] = np.mean(deltas)
        generalization_deltas = []
        for label, (truth, members) in standalone_banks.items():
            expert = np.tensordot(weights, members, axes=(0, 0))
            prediction = pd.DataFrame(
                expert, index=truth.index, columns=tuple(FUSION_WEIGHTS),
            )
            ads = score_frame(truth.join(prediction))["ADS"]
            delta = ads - standalone_baseline[label]
            record[f"{label}_EXPERT_ADS"] = ads
            record[f"{label}_DELTA_VS_UNIFORM"] = delta
            generalization_deltas.append(delta)
        record["GENERAL_MIN_DELTA"] = min(generalization_deltas)
        record["GENERAL_MEAN_DELTA"] = np.mean(generalization_deltas)
        rows.append(record)
    result = pd.DataFrame(rows).sort_values(
        ["MIN_DELTA", "MEAN_DELTA"], ascending=False
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output_dir / "fixed_member_grid.csv", index=False)
    print(result.head(30).to_string(index=False))
    print("\ngeneralization-first\n")
    print(result.sort_values(
        ["GENERAL_MIN_DELTA", "MIN_DELTA", "GENERAL_MEAN_DELTA"],
        ascending=False,
    ).head(20).to_string(index=False))


if __name__ == "__main__":
    main()
