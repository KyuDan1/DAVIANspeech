#!/usr/bin/env python3
"""Select and audit a dedicated Music residual without using blind sets."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]

from channel_invariant_music_inference import (  # noqa: E402
    load_music_heads, predict_music_from_statistics, residual_music_fusion,
)
from evaluate_diagnostic import official_eer  # noqa: E402
from train_channel_invariant_music_head import DEV_DEFAULT  # noqa: E402
from train_dual_domain_head import Bank, load_bank  # noqa: E402


BLIND = ("codec_mixed_blind_v5", "codec_mixed_blind_v6")
WEIGHTS = (0.0, .01, .025, .0375, .05, .075, .10, .15, .20)
SPECIAL_ANCHORS = {
    "codec_mixed_dev_v4": ROOT / "reports/v50_candidate_codec_dev_v4/predictions.csv",
    "codec_mixed_blind_v5": ROOT / "reports/v50_candidate_blind_v5/predictions.csv",
    "codec_mixed_blind_v6":
        ROOT / "reports/v50_blind_v6_one_shot/v50_frozen/output/submission.csv",
}


def _anchor_path(name: str) -> Path:
    if name in SPECIAL_ANCHORS:
        return SPECIAL_ANCHORS[name]
    cache_name = (
        "factorial_eval_1200_v2" if name.startswith("factorial_eval_1200_v2_")
        else name
    )
    return (
        ROOT / "reports/v47_anchor_cache_train_dev_v2/cache/datasets"
        / cache_name / "exact_v47_predictions.csv"
    )


def load_anchor(bank: Bank) -> np.ndarray:
    """Load exact v50 Music (the v47 Music output is bit-preserved by v50)."""
    path = _anchor_path(bank.name)
    frame = pd.read_csv(path, dtype={"ID": str}).set_index("ID")
    missing = set(bank.ids) - set(frame.index)
    if missing:
        raise ValueError(f"anchor misses {len(missing)} IDs for {bank.name}")
    return frame.loc[bank.ids, "MUSIC_FAKE_PROB"].to_numpy(np.float32)


def music_eer(bank: Bank, scores: np.ndarray) -> float:
    present = bank.truth.MUSIC_PRESENT.eq(1).to_numpy()
    labels = bank.truth.loc[present, "MUSIC_FAKE"].astype(int).to_numpy()
    return float(official_eer(labels, scores[present]))


def candidate_metrics(
    banks: list[Bank], scores: dict[str, np.ndarray], candidate: str,
) -> pd.DataFrame:
    rows = []
    for bank in banks:
        eer = music_eer(bank, scores[bank.name])
        rows.append({
            "CANDIDATE": candidate, "DATASET": bank.name,
            "MUSIC_EER": eer, "MUSIC_SCORE": 1 - eer,
        })
    return pd.DataFrame(rows)


def selection_summary(metrics: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for candidate, group in metrics.groupby("CANDIDATE", sort=False):
        scores = group.MUSIC_SCORE
        rows.append({
            "CANDIDATE": candidate,
            "MEAN_MUSIC_SCORE": scores.mean(),
            "WORST_MUSIC_SCORE": scores.min(),
            "SELECTION": .5 * scores.mean() + .5 * scores.min(),
        })
    return pd.DataFrame(rows).sort_values(
        ["SELECTION", "CANDIDATE"], ascending=[False, True],
    )


def slice_metrics(
    bank: Bank, anchor: np.ndarray, expert: np.ndarray, fused: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    truth = bank.truth.copy()
    truth["ANCHOR"] = anchor
    truth["EXPERT"] = expert
    truth["FUSED"] = fused
    rows = []
    for column in ("CHANNEL", "MIX_MODE", "VOICE_FAKE", "MUSIC_GENERATOR"):
        if column not in truth:
            continue
        for value, group in truth.groupby(column, dropna=False):
            if group.MUSIC_FAKE.nunique() < 2:
                continue
            for method in ("ANCHOR", "EXPERT", "FUSED"):
                eer = official_eer(group.MUSIC_FAKE.astype(int), group[method])
                rows.append({
                    "DATASET": bank.name, "SLICE": column, "VALUE": value,
                    "METHOD": method, "N": len(group), "MUSIC_EER": eer,
                })
    cells = []
    for case, group in truth.groupby("COMPONENT_CASE"):
        for method in ("ANCHOR", "EXPERT", "FUSED"):
            cells.append({
                "DATASET": bank.name, "COMPONENT_CASE": case,
                "METHOD": method, "N": len(group),
                "MEAN_SCORE": group[method].mean(),
                "MEDIAN_SCORE": group[method].median(),
            })
    return pd.DataFrame(rows), pd.DataFrame(cells)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoints", nargs="+", type=Path, required=True)
    parser.add_argument(
        "--stats-root", type=Path, default=ROOT / "output/dual_domain_stats_v1",
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=ROOT / "reports/channel_invariant_music_v52/evaluation",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=96)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    checkpoint_metadata = [
        torch.load(path, map_location="cpu", weights_only=False)
        for path in args.checkpoints
    ]
    reference_selection = float(checkpoint_metadata[0]["selection"])
    # Predeclared rule: seed two joins seed one only if its independently
    # selected development maximin is no worse than seed one's.
    ensemble_allowed = len(args.checkpoints) > 1 and all(
        float(item["selection"]) >= reference_selection - 1e-5
        for item in checkpoint_metadata[1:]
    )
    eligibility = pd.DataFrame({
        "CHECKPOINT": [str(path) for path in args.checkpoints],
        "SEED": [item["seed"] for item in checkpoint_metadata],
        "DEV_SELECTION": [item["selection"] for item in checkpoint_metadata],
        "ENSEMBLE_ELIGIBLE": [
            bool(ensemble_allowed) for _ in args.checkpoints
        ],
    })
    eligibility.to_csv(args.output_dir / "seed_eligibility.csv", index=False)

    authorized = [load_bank(args.stats_root, name, "clean") for name in DEV_DEFAULT]
    diagnostics = [load_bank(args.stats_root, name, "clean") for name in BLIND]
    all_banks = authorized + diagnostics
    expert_variants = {}
    anchor = {}
    for path, metadata in zip(args.checkpoints, checkpoint_metadata):
        name = f"seed{metadata['seed']}"
        heads = load_music_heads([path], device)
        expert_variants[name] = {}
        for bank in all_banks:
            expert_variants[name][bank.name] = predict_music_from_statistics(
                heads, bank.eat, bank.spear, bank.eat_mask, bank.spear_mask,
                device=device, batch_size=args.batch_size,
            )
    if ensemble_allowed:
        heads = load_music_heads(args.checkpoints, device)
        expert_variants["eligible_ensemble"] = {}
        for bank in all_banks:
            expert_variants["eligible_ensemble"][bank.name] = (
                predict_music_from_statistics(
                    heads, bank.eat, bank.spear, bank.eat_mask, bank.spear_mask,
                    device=device, batch_size=args.batch_size,
                )
            )
    for bank in all_banks:
        anchor[bank.name] = load_anchor(bank)

    authorized_metrics = []
    candidate_scores = {}
    # Anchor is seed-independent and included exactly once.
    candidate_scores["anchor_w0"] = anchor
    authorized_metrics.append(candidate_metrics(authorized, anchor, "anchor_w0"))
    for expert_name, expert in expert_variants.items():
        for weight in WEIGHTS[1:]:
            name = f"{expert_name}_w{weight:g}"
            candidate_scores[name] = {
                bank.name: residual_music_fusion(
                    anchor[bank.name], expert[bank.name], weight,
                )
                for bank in all_banks
            }
            authorized_metrics.append(candidate_metrics(
                authorized, candidate_scores[name], name,
            ))
        name = f"{expert_name}_w1"
        candidate_scores[name] = expert
        authorized_metrics.append(candidate_metrics(authorized, expert, name))
    authorized_metrics = pd.concat(authorized_metrics, ignore_index=True)
    selection = selection_summary(authorized_metrics)
    chosen = str(selection.iloc[0].CANDIDATE)
    chosen_scores = candidate_scores[chosen]
    chosen_expert_name = chosen.rsplit("_w", 1)[0]
    chosen_expert = expert_variants.get(chosen_expert_name, anchor)
    authorized_metrics.to_csv(args.output_dir / "authorized_dev_grid.csv", index=False)
    selection.to_csv(args.output_dir / "authorized_dev_selection.csv", index=False)

    blind_metrics = pd.concat([
        candidate_metrics(diagnostics, scores, name)
        for name, scores in candidate_scores.items()
    ], ignore_index=True)
    blind_metrics["ROLE"] = blind_metrics.DATASET.map({
        "codec_mixed_blind_v5": "retired_blind_nonselection",
        "codec_mixed_blind_v6": "diagnostic_only_nonselection",
    })
    blind_metrics.to_csv(args.output_dir / "blind_diagnostic_grid.csv", index=False)

    slices, cells, predictions = [], [], []
    for bank in (authorized[-1], *diagnostics):
        sliced, cell = slice_metrics(
            bank, anchor[bank.name], chosen_expert[bank.name], chosen_scores[bank.name],
        )
        slices.append(sliced)
        cells.append(cell)
        predictions.extend({
            "DATASET": bank.name, "ID": item,
            "ANCHOR_MUSIC_FAKE_PROB": float(old),
            "EXPERT_MUSIC_FAKE_PROB": float(new),
            "FUSED_MUSIC_FAKE_PROB": float(fused),
        } for item, old, new, fused in zip(
            bank.ids, anchor[bank.name], chosen_expert[bank.name], chosen_scores[bank.name],
        ))
    pd.concat(slices, ignore_index=True).to_csv(
        args.output_dir / "slice_eer.csv", index=False,
    )
    pd.concat(cells, ignore_index=True).to_csv(
        args.output_dir / "component_case_scores.csv", index=False,
    )
    pd.DataFrame(predictions).to_csv(args.output_dir / "audit_predictions.csv", index=False)

    blind_chosen = blind_metrics.loc[blind_metrics.CANDIDATE.eq(chosen)]
    v4 = authorized_metrics.loc[
        authorized_metrics.CANDIDATE.eq(chosen)
        & authorized_metrics.DATASET.eq("codec_mixed_dev_v4"), "MUSIC_EER",
    ].iloc[0]
    v4_anchor = authorized_metrics.loc[
        authorized_metrics.CANDIDATE.eq("anchor_w0")
        & authorized_metrics.DATASET.eq("codec_mixed_dev_v4"), "MUSIC_EER",
    ].iloc[0]
    v5 = blind_chosen.loc[
        blind_chosen.DATASET.eq("codec_mixed_blind_v5"), "MUSIC_EER",
    ].iloc[0]
    v5_anchor = blind_metrics.loc[
        blind_metrics.CANDIDATE.eq("anchor_w0")
        & blind_metrics.DATASET.eq("codec_mixed_blind_v5"), "MUSIC_EER",
    ].iloc[0]
    summary = {
        "chosen_by_authorized_dev_only": chosen,
        "authorized_selection": float(selection.iloc[0].SELECTION),
        "ensemble_allowed": ensemble_allowed,
        "eligible_seeds": (
            [int(value) for value in eligibility.SEED]
            if ensemble_allowed else [int(eligibility.SEED.iloc[0])]
        ),
        "codec_v4_anchor_eer": float(v4_anchor),
        "codec_v4_chosen_eer": float(v4),
        "blind_v5_anchor_eer": float(v5_anchor),
        "blind_v5_chosen_eer": float(v5),
        "overfit_warning": bool((v5 - v5_anchor) > .01),
        "blind_v6_used_for_selection": False,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8",
    )
    print(selection.to_string(index=False))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
