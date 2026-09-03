#!/usr/bin/env python3
"""Compare joint MoE with EAT-music/SPEAR-voice stream specialists."""

from __future__ import annotations

import argparse
import gc
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from evaluate_codec_invariant_fusion import AUDITS, fuse, reconstruct_v18  # noqa: E402
from evaluate_presence_weighted_file_fusion import reconstruct_dev_v18  # noqa: E402
from train_attention_expert_router import DEFAULT_MEMBERS, load_members  # noqa: E402
from train_dual_domain_head import DEV_DEFAULT, load_bank, metrics  # noqa: E402


AUDIT_DATASETS = (
    "factorial_eval_1200_v2_holdout",
    "phone_factorial_1200_v1",
    "yue_cross_component_audit_v1",
)
MODES = ("joint", "eat", "spear")
STRATEGIES = ("joint", "hard_axis", "soft_axis_025", "soft_axis_050", "soft_axis_075")


@torch.inference_mode()
def extract(
    bank, members, device: torch.device, batch_size: int,
) -> np.ndarray:
    batches = []
    for offset in range(0, len(bank.ids), batch_size):
        eat = torch.from_numpy(bank.eat[offset:offset + batch_size]).to(
            device=device, dtype=torch.float32
        )
        spear = torch.from_numpy(bank.spear[offset:offset + batch_size]).to(
            device=device, dtype=torch.float32
        )
        eat_mask = torch.from_numpy(bank.eat_mask[offset:offset + batch_size]).to(device)
        spear_mask = torch.from_numpy(bank.spear_mask[offset:offset + batch_size]).to(device)
        member_values = []
        for model, normal in members:
            normalized_eat = ((eat - normal["eat_mean"]) / normal["eat_std"]).clamp(-8, 8)
            normalized_spear = ((spear - normal["spear_mean"]) / normal["spear_std"]).clamp(-8, 8)
            mode_values = []
            for mode in MODES:
                local_eat_mask = eat_mask if mode != "spear" else torch.zeros_like(eat_mask)
                local_spear_mask = spear_mask if mode != "eat" else torch.zeros_like(spear_mask)
                with torch.autocast(
                    device_type=device.type, dtype=torch.bfloat16,
                    enabled=device.type == "cuda",
                ):
                    task_logits, joint_logits = model(
                        normalized_eat, normalized_spear,
                        local_eat_mask, local_spear_mask,
                    )
                mode_values.append(model.probabilities(
                    task_logits.float(), joint_logits.float()
                ).cpu())
            member_values.append(torch.stack(mode_values, dim=1))
        batches.append(torch.stack(member_values, dim=1))
    return torch.cat(batches).numpy()  # [N,E,mode,task]


def predict(values: np.ndarray, strategy: str) -> np.ndarray:
    joint, eat, spear = values.mean(axis=1).transpose(1, 0, 2)
    specialist = joint.copy()
    specialist[:, 0] = spear[:, 0]  # voice authenticity
    specialist[:, 1] = eat[:, 1]  # music authenticity
    if strategy == "joint":
        return joint
    if strategy == "hard_axis":
        return specialist
    if strategy.startswith("soft_axis_"):
        weight = int(strategy.rsplit("_", 1)[1]) / 100
        return (1 - weight) * joint + weight * specialist
    raise ValueError(strategy)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--members", nargs="+", type=Path, default=list(DEFAULT_MEMBERS))
    parser.add_argument(
        "--stats-root", type=Path,
        default=ROOT / "output/dual_domain_stats_v1",
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=ROOT / "reports/stream_specialist_router_v1",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()
    device = torch.device(args.device)
    members = load_members(args.members, device)
    banks = {}
    for name in (*DEV_DEFAULT, *AUDIT_DATASETS):
        cache = args.output_dir / "cache" / f"{name}.npz"
        if cache.is_file():
            archive = np.load(cache, allow_pickle=False)
            bank = load_bank(args.stats_root, name, "clean")
            if not np.array_equal(archive["ids"].astype(str), bank.ids):
                raise ValueError(f"stale cache ID order for {name}")
            values = archive["probabilities"]
        else:
            bank = load_bank(args.stats_root, name, "clean")
            values = extract(bank, members, device, args.batch_size)
            cache.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(cache, ids=bank.ids, probabilities=values)
        banks[name] = bank, values
        gc.collect()
    del members
    torch.cuda.empty_cache()

    rows = []
    for name, (bank, values) in banks.items():
        split = "dev" if name in DEV_DEFAULT else "locked"
        for strategy in STRATEGIES:
            rows.append({
                "SPLIT": split, "DATASET": name, "STRATEGY": strategy,
                **metrics(bank.truth, predict(values, strategy)),
            })
    result = pd.DataFrame(rows)
    baseline = result.loc[result.STRATEGY.eq("joint")].set_index("DATASET").ADS
    result["DELTA_VS_JOINT"] = [
        row.ADS - baseline[row.DATASET] for row in result.itertuples()
    ]
    result.to_csv(args.output_dir / "standalone.csv", index=False)
    summary = result.groupby(["SPLIT", "STRATEGY"]).agg(
        MIN_DELTA=("DELTA_VS_JOINT", "min"),
        MEAN_DELTA=("DELTA_VS_JOINT", "mean"),
    ).reset_index()
    summary.to_csv(args.output_dir / "summary.csv", index=False)

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
    audit_map = {
        "dev": "factorial_eval_1200_v2_dev",
        "factorial": "factorial_eval_1200_v2_holdout",
        "phone": "phone_factorial_1200_v1",
        "yue": "yue_cross_component_audit_v1",
    }
    fusion_rows = []
    for label, name in audit_map.items():
        bank, values = banks[name]
        if label == "dev":
            truth, anchor = reconstruct_dev_v18(temporal_dev, fakeprint, invariant_dev)
        else:
            truth, anchor = reconstruct_v18(
                AUDITS[label], temporal_audit, fakeprint, invariant_audit
            )
        order_map = {item: index for index, item in enumerate(bank.ids)}
        order = np.asarray([order_map[item] for item in anchor.index])
        for strategy in STRATEGIES:
            expert = predict(values, strategy)[order]
            candidate = anchor.copy()
            for task, column in enumerate((
                "VOICE_FAKE_PROB", "MUSIC_FAKE_PROB", "FILE_FAKE_PROB",
            )):
                candidate[column] = fuse(candidate[column], expert[:, task], .30)
            fusion_rows.append({
                "DATASET": label, "STRATEGY": strategy,
                **metrics(
                    truth.reset_index(), candidate[[
                        "VOICE_FAKE_PROB", "MUSIC_FAKE_PROB", "FILE_FAKE_PROB",
                    ]].to_numpy(),
                ),
            })
    fusion = pd.DataFrame(fusion_rows)
    baseline = fusion.loc[fusion.STRATEGY.eq("joint")].set_index("DATASET").ADS
    fusion["DELTA_VS_JOINT"] = [
        row.ADS - baseline[row.DATASET] for row in fusion.itertuples()
    ]
    fusion.to_csv(args.output_dir / "v18_w30_fusion.csv", index=False)
    print(summary.to_string(index=False))
    print("\nFUSION DELTAS\n" + fusion.pivot(
        index="STRATEGY", columns="DATASET", values="DELTA_VS_JOINT"
    ).to_string())


if __name__ == "__main__":
    main()
