#!/usr/bin/env python3
"""Ablate which authenticity axes may use the bounded attention router."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from attention_expert_router import BoundedAttentionRouter  # noqa: E402
from evaluate_codec_invariant_fusion import AUDITS, fuse, reconstruct_v18  # noqa: E402
from evaluate_presence_weighted_file_fusion import reconstruct_dev_v18  # noqa: E402
from train_attention_expert_router import (  # noqa: E402
    RoutedBank, ensemble_router_probabilities, task_weights, truth_path,
)
from train_dual_domain_head import metrics  # noqa: E402


DATASETS = {
    "dev": "factorial_eval_1200_v2_dev",
    "factorial": "factorial_eval_1200_v2_holdout",
    "phone": "phone_factorial_1200_v1",
    "yue": "yue_cross_component_audit_v1",
}
AXES = {
    "uniform": (),
    "file": (2,),
    "voice": (0,),
    "music": (1,),
    "file_music": (2, 1),
    "file_voice": (2, 0),
    "all": (0, 1, 2),
}


def cached_bank(cache_dir: Path, name: str) -> RoutedBank:
    archive = np.load(cache_dir / f"{name}__clean.npz", allow_pickle=False)
    ids = archive["ids"].astype(str)
    truth = pd.read_csv(truth_path(name), dtype={"ID": str}).set_index("ID")
    truth = truth.loc[ids].reset_index()
    bank = RoutedBank(
        name=name, channel="clean", ids=ids,
        hidden=archive["hidden"], probabilities=archive["probabilities"],
        targets=archive["targets"], task_weights=archive["task_weights"],
        truth=truth,
    )
    # Detect stale cache contents rather than silently scoring wrong labels.
    if not np.allclose(bank.task_weights, task_weights(type("B", (), {
        "truth": truth, "targets": bank.targets, "ids": ids,
    })())):
        raise ValueError(f"stale task weights for {name}")
    return bank


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint", type=Path,
        default=ROOT / "reports/attention_expert_router_v1_s025e/attention_router.pt",
    )
    parser.add_argument(
        "--cache-dir", type=Path,
        default=ROOT / "reports/attention_expert_router_v1/cache",
    )
    parser.add_argument(
        "--output", type=Path,
        default=ROOT / "reports/attention_expert_router_v1_s025e/axis_ablation.csv",
    )
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    device = torch.device(args.device)
    models = []
    for model_state in state["models"]:
        model = BoundedAttentionRouter(**state["config"]).to(device)
        model.load_state_dict(model_state)
        model.eval()
        models.append(model)

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

    rows = []
    for label, name in DATASETS.items():
        bank = cached_bank(args.cache_dir, name)
        uniform = bank.probabilities.mean(axis=1)
        routed, _ = ensemble_router_probabilities(models, bank, device)
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
        uniform, routed = uniform[order], routed[order]
        for method, routed_axes in AXES.items():
            expert = uniform.copy()
            for axis in routed_axes:
                expert[:, axis] = routed[:, axis]
            candidate = anchor.copy()
            for axis, column in enumerate((
                "VOICE_FAKE_PROB", "MUSIC_FAKE_PROB", "FILE_FAKE_PROB",
            )):
                candidate[column] = fuse(candidate[column], expert[:, axis], .30)
            rows.append({
                "METHOD": method, "DATASET": label,
                **metrics(
                    truth.reset_index(), candidate[[
                        "VOICE_FAKE_PROB", "MUSIC_FAKE_PROB", "FILE_FAKE_PROB",
                    ]].to_numpy(),
                ),
            })
    result = pd.DataFrame(rows)
    baseline = result.loc[result.METHOD.eq("uniform")].set_index("DATASET").ADS
    result["DELTA_VS_UNIFORM"] = [
        row.ADS - baseline[row.DATASET] for row in result.itertuples()
    ]
    summary = result.groupby("METHOD").agg(
        MIN_DELTA=("DELTA_VS_UNIFORM", "min"),
        MEAN_DELTA=("DELTA_VS_UNIFORM", "mean"),
    ).sort_values(["MIN_DELTA", "MEAN_DELTA"], ascending=False)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)
    print(summary.to_string())
    print("\n" + result.pivot(
        index="METHOD", columns="DATASET", values="DELTA_VS_UNIFORM"
    ).to_string())


if __name__ == "__main__":
    main()
