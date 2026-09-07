#!/usr/bin/env python3
"""Select hierarchical-EAT fusion on dev, then open locked audits once."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from evaluate_codec_invariant_fusion import AUDITS, fuse, reconstruct_v18  # noqa: E402
from evaluate_diagnostic import score_frame  # noqa: E402
from evaluate_presence_weighted_file_fusion import reconstruct_dev_v18  # noqa: E402
from evaluate_long_horizon_music_routing import ROUTERS  # noqa: E402


def load(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, dtype={"ID": str})
    if frame.duplicated(["DATASET", "ID"]).any():
        raise ValueError(f"duplicate prediction IDs in {path}")
    return frame


def select(frame: pd.DataFrame, dataset: str, ids: pd.Index) -> np.ndarray:
    block = frame.loc[frame.DATASET.eq(dataset)].set_index("ID")
    missing = ids.difference(block.index)
    if len(missing):
        raise ValueError(f"missing {len(missing)} predictions for {dataset}")
    return block.loc[ids, "HIERARCHICAL_EAT_MUSIC_PROB"].to_numpy(float)


def apply(
    anchor: pd.DataFrame,
    expert: np.ndarray,
    music_weight: float,
    file_weight: float,
    presence_gate: float,
    phone_mask: np.ndarray | None = None,
    phone_music_weight: float = 0.30,
    phone_file_weight: float = 0.30,
) -> pd.DataFrame:
    if phone_mask is None:
        phone_mask = np.zeros(len(anchor), dtype=bool)
    phone_mask = np.asarray(phone_mask, dtype=bool)
    if phone_mask.shape != (len(anchor),):
        raise ValueError("phone mask has incompatible shape")
    result = anchor.copy()
    result["MUSIC_FAKE_PROB"] = fuse(
        result.MUSIC_FAKE_PROB, expert,
        np.where(phone_mask, phone_music_weight, music_weight),
    )
    route = result.MUSIC_PRESENT_PROB.to_numpy(float) >= presence_gate
    result["FILE_FAKE_PROB"] = fuse(
        result.FILE_FAKE_PROB, expert,
        np.where(phone_mask, phone_file_weight, file_weight) * route,
    )
    return result


def phone_mask(path: Path | None, ids: pd.Index) -> np.ndarray:
    if path is None:
        return np.zeros(len(ids), dtype=bool)
    frame = pd.read_csv(path, dtype={"ID": str}).set_index("ID")
    missing = ids.difference(frame.index)
    if len(missing):
        raise ValueError(f"phone router misses {len(missing)} IDs")
    return frame.loc[ids, "IS_PHONE"].to_numpy(bool)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dev-predictions", type=Path, required=True)
    parser.add_argument("--audit-predictions", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--weights", type=float, nargs="+",
        default=[0, .025, .05, .075, .10, .15, .20, .30, .40, .50],
    )
    parser.add_argument(
        "--presence-gates", type=float, nargs="+", default=[.5, .7, 1.1]
    )
    parser.add_argument("--phone-music-weight", type=float, default=.30)
    parser.add_argument("--phone-file-weight", type=float, default=.30)
    parser.add_argument(
        "--max-file-weight", type=float, default=.15,
        help=(
            "Development-selection safety cap for the normal-domain file head. "
            "The file target propagates either component error, so large music-only "
            "residuals are deliberately prevented from dominating it."
        ),
    )
    args = parser.parse_args()
    if any(not 0 <= weight <= 1 for weight in args.weights):
        parser.error("fusion weights must lie in [0, 1]")
    if not 0 <= args.max_file_weight <= 1:
        parser.error("--max-file-weight must lie in [0, 1]")
    if not 0 <= args.phone_music_weight <= 1:
        parser.error("--phone-music-weight must lie in [0, 1]")
    if not 0 <= args.phone_file_weight <= 1:
        parser.error("--phone-file-weight must lie in [0, 1]")
    dev_score = load(args.dev_predictions)
    audit_score = load(args.audit_predictions)
    temporal_dev = load(
        ROOT / "reports/temporal_dual_domain_hybrid/ensemble_dev/predictions.csv"
    )
    temporal_audit = load(
        ROOT / "reports/temporal_dual_domain_hybrid/ensemble_audit/predictions.csv"
    )
    fakeprint = load(ROOT / "reports/modern_fakeprint_v1/independent_margin_scores.csv")
    invariant_dev = load(ROOT / "reports/invariant_dual_domain_v2/dec_n1/dev_predictions.csv")
    invariant_audit = load(
        ROOT / "reports/invariant_dual_domain_v2/dec_n1_audit/predictions.csv"
    )

    dev_truth, dev_anchor = reconstruct_dev_v18(
        temporal_dev, fakeprint, invariant_dev
    )
    dev_expert = select(
        dev_score, "factorial_eval_1200_v2_dev", dev_anchor.index
    )
    dev_phone = phone_mask(ROUTERS["factorial"], dev_anchor.index)
    baseline = score_frame(dev_truth.join(dev_anchor))
    sweep = []
    for music_weight in args.weights:
        for file_weight in args.weights:
            if file_weight > args.max_file_weight:
                continue
            for gate in args.presence_gates:
                prediction = apply(
                    dev_anchor, dev_expert, music_weight, file_weight, gate,
                    phone_mask=dev_phone,
                    phone_music_weight=args.phone_music_weight,
                    phone_file_weight=args.phone_file_weight,
                )
                metrics = score_frame(dev_truth.join(prediction))
                sweep.append({
                    "MUSIC_WEIGHT": music_weight,
                    "FILE_WEIGHT": file_weight,
                    "PRESENCE_GATE": gate,
                    "PHONE_COUNT": int(dev_phone.sum()),
                    "DELTA": metrics["ADS"] - baseline["ADS"],
                    **metrics,
                })
    sweep = pd.DataFrame(sweep).sort_values(
        ["ADS", "MUSIC_WEIGHT", "FILE_WEIGHT", "PRESENCE_GATE"],
        ascending=[False, True, True, False],
    )
    chosen = sweep.iloc[0]
    settings = {
        "music_weight": float(chosen.MUSIC_WEIGHT),
        "file_weight": float(chosen.FILE_WEIGHT),
        "presence_gate": float(chosen.PRESENCE_GATE),
        "phone_music_weight": args.phone_music_weight,
        "phone_file_weight": args.phone_file_weight,
        "max_file_weight": args.max_file_weight,
        "dev_ads": float(chosen.ADS),
        "dev_delta": float(chosen.DELTA),
    }

    rows = []
    combined_truth, combined_prediction = [], []
    for name, spec in AUDITS.items():
        truth, anchor = reconstruct_v18(
            spec, temporal_audit, fakeprint, invariant_audit
        )
        expert = select(audit_score, spec.temporal_name, anchor.index)
        routed = phone_mask(ROUTERS.get(name), anchor.index)
        prediction = apply(anchor, expert, phone_mask=routed, **{
            key: settings[key]
            for key in (
                "music_weight", "file_weight", "presence_gate",
                "phone_music_weight", "phone_file_weight",
            )
        })
        rows.append({
            "DATASET": name, "PHONE_COUNT": int(routed.sum()),
            **score_frame(truth.join(prediction)),
        })
        if name in {"factorial", "phone"}:
            truth, prediction = truth.copy(), prediction.copy()
            truth.index = name + "_" + truth.index
            prediction.index = truth.index
            combined_truth.append(truth)
            combined_prediction.append(prediction)
    rows.append({
        "DATASET": "factorial_plus_phone",
        **score_frame(pd.concat(combined_truth).join(pd.concat(combined_prediction))),
    })
    args.output_dir.mkdir(parents=True, exist_ok=True)
    sweep.to_csv(args.output_dir / "dev_fusion_sweep.csv", index=False)
    metrics = pd.DataFrame(rows)
    metrics.to_csv(args.output_dir / "locked_metrics.csv", index=False)
    (args.output_dir / "selection.json").write_text(
        json.dumps(settings, indent=2), encoding="utf-8"
    )
    print(json.dumps(settings, indent=2))
    print(metrics[[
        "DATASET", "FILE_EER", "VOICE_EER", "MUSIC_EER", "ADS"
    ]].to_string(index=False))


if __name__ == "__main__":
    main()
