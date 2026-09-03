#!/usr/bin/env python3
"""Audit a component/channel-invariant residual on the exact v33 stack."""

from __future__ import annotations

import argparse
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
    DATASETS, ROUTERS, cross_component_diagnostics, load, select,
)
from presence_weighted_file_fusion import component_file_evidence  # noqa: E402


INVARIANT_DATASETS = {
    "dev": "factorial_eval_1200_v2_dev",
    "factorial": "factorial_eval_1200_v2_holdout",
    "phone": "phone_factorial_1200_v1",
    "yue": "yue_cross_component_audit_v1",
}
PREDICTION_COLUMNS = {
    "file": "FILE_FAKE_PROB",
    "voice": "VOICE_FAKE_PROB",
    "music": "MUSIC_FAKE_PROB",
}


def logit(values) -> np.ndarray:
    values = np.clip(np.asarray(values, dtype=np.float64), 1e-5, 1 - 1e-5)
    return np.log(values) - np.log1p(-values)


def sigmoid(values) -> np.ndarray:
    return np.exp(-np.logaddexp(0., -np.asarray(values, dtype=np.float64)))


def file_route(
    prediction: pd.DataFrame, mode: str, margin: float,
) -> np.ndarray:
    """Return a file-local gate without using other evaluation samples."""
    voice = logit(prediction.VOICE_FAKE_PROB)
    music = logit(prediction.MUSIC_FAKE_PROB)
    voice_present = prediction.VOICE_PRESENT_PROB.to_numpy() >= .5
    music_present = prediction.MUSIC_PRESENT_PROB.to_numpy() >= .5
    voice_presence_probability = prediction.VOICE_PRESENT_PROB.to_numpy(float)
    music_presence_probability = prediction.MUSIC_PRESENT_PROB.to_numpy(float)
    if mode == "global":
        return np.ones(len(prediction), dtype=bool)
    if mode == "voice_dominant":
        return voice - music >= margin
    if mode == "music_dominant":
        return music - voice >= margin
    if mode == "voice_present_dominant":
        return voice_present & (voice - music >= margin)
    if mode == "predicted_mixed_voice_dominant":
        return voice_present & music_present & (voice - music >= margin)
    if mode == "voice_soft_t05":
        return sigmoid((voice - music - margin) / .5)
    if mode == "voice_soft_t10":
        return sigmoid(voice - music - margin)
    if mode == "voice_soft_t20":
        return sigmoid((voice - music - margin) / 2.)
    if mode == "voice_presence_soft":
        return voice_presence_probability * sigmoid(voice - music - margin)
    if mode == "mixed_voice_soft":
        return (
            np.sqrt(voice_presence_probability * music_presence_probability)
            * sigmoid(voice - music - margin)
        )
    raise ValueError(f"unknown file route: {mode}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--attention-predictions", type=Path,
        default=ROOT / "reports/spear_temporal_attention_v1/attention_channel_s12_audit/predictions.csv",
    )
    parser.add_argument(
        "--music-predictions", type=Path,
        default=ROOT / "reports/spear_temporal_bin_v1/mlp64_seed05_audit/predictions.csv",
    )
    parser.add_argument(
        "--invariant-dev-predictions", type=Path,
        default=ROOT / "reports/invariant_dual_domain_v4_paired/fixed_ch01e4_dev/predictions.csv",
    )
    parser.add_argument(
        "--invariant-audit-predictions", type=Path,
        default=ROOT / "reports/invariant_dual_domain_v4_paired/fixed_ch01e4_audit/predictions.csv",
    )
    parser.add_argument(
        "--weights", type=float, nargs="+",
        default=[0, .025, .05, .075, .10, .125, .15, .20],
    )
    parser.add_argument(
        "--axes", nargs="+", choices=tuple(PREDICTION_COLUMNS),
        default=list(PREDICTION_COLUMNS),
    )
    parser.add_argument("--selected-file-weight", type=float, default=.125)
    parser.add_argument("--selected-voice-weight", type=float, default=.20)
    parser.add_argument("--selected-music-weight", type=float, default=.10)
    parser.add_argument(
        "--output-dir", type=Path,
        default=ROOT / "reports/invariant_dual_domain_v4_paired/v33_residual",
    )
    args = parser.parse_args()
    selected_weights = {
        "FILE_FAKE_PROB": args.selected_file_weight,
        "VOICE_FAKE_PROB": args.selected_voice_weight,
        "MUSIC_FAKE_PROB": args.selected_music_weight,
    }
    if any(not 0 <= value <= 1 for value in [*args.weights, *selected_weights.values()]):
        parser.error("fusion weights must be in [0, 1]")

    temporal_dev = load(ROOT / "reports/temporal_dual_domain_hybrid/ensemble_dev/predictions.csv")
    temporal_audit = load(ROOT / "reports/temporal_dual_domain_hybrid/ensemble_audit/predictions.csv")
    fakeprint = load(ROOT / "reports/modern_fakeprint_v1/independent_margin_scores.csv")
    invariant_dev = load(ROOT / "reports/invariant_dual_domain_v2/dec_n1/dev_predictions.csv")
    invariant_audit = load(ROOT / "reports/invariant_dual_domain_v2/dec_n1_audit/predictions.csv")
    attention = load(args.attention_predictions)
    music = load(args.music_predictions)
    new_dev = load(args.invariant_dev_predictions)
    new_audit = load(args.invariant_audit_predictions)

    banks = {}
    for name, dataset in DATASETS.items():
        if name == "dev":
            truth, anchor = reconstruct_dev_v18(
                temporal_dev, fakeprint, invariant_dev
            )
            new = new_dev
        else:
            truth, anchor = reconstruct_v18(
                AUDITS[name], temporal_audit, fakeprint, invariant_audit
            )
            new = new_audit
        attention_score = select(attention, dataset, anchor.index)
        music_score = select(music, dataset, anchor.index)
        new_score = select(new, INVARIANT_DATASETS[name], anchor.index)
        phone = np.zeros(len(anchor), dtype=bool)
        if name in ROUTERS:
            phone = (
                load(ROUTERS[name]).set_index("ID")
                .loc[anchor.index, "IS_PHONE"].to_numpy(bool)
            )
        v33 = anchor.copy()
        v33["FILE_FAKE_PROB"] = fuse(
            anchor.FILE_FAKE_PROB, attention_score.FILE_FAKE_PROB,
            np.where(phone, .25, .20),
        )
        v33["MUSIC_FAKE_PROB"] = fuse(
            anchor.MUSIC_FAKE_PROB,
            music_score.TEMPORAL_BIN_MUSIC_PROB, .50,
        )
        banks[name] = truth, v33, new_score

    columns = [PREDICTION_COLUMNS[axis] for axis in args.axes]
    rows = []
    for weight in args.weights:
        record = {"WEIGHT": weight, "AXES": "+".join(args.axes)}
        for name, (truth, v33, expert) in banks.items():
            candidate = v33.copy()
            for column in columns:
                candidate[column] = fuse(v33[column], expert[column], weight)
            metrics = score_frame(truth.join(candidate))
            baseline = score_frame(truth.join(v33))
            record[f"{name}_ADS"] = metrics["ADS"]
            record[f"{name}_DELTA"] = metrics["ADS"] - baseline["ADS"]
        deltas = [record[f"{name}_DELTA"] for name in banks]
        record["MIN_DELTA"] = min(deltas)
        record["MEAN_DELTA"] = np.mean(deltas)
        rows.append(record)
    sweep = pd.DataFrame(rows).sort_values(
        ["MIN_DELTA", "MEAN_DELTA"], ascending=False
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    sweep.to_csv(args.output_dir / "sweep.csv", index=False)

    diagnostic_rows = []
    for name, (truth, v33, expert) in banks.items():
        candidate = v33.copy()
        for column in columns:
            candidate[column] = fuse(
                v33[column], expert[column], selected_weights[column]
            )
        diagnostic_rows.extend(cross_component_diagnostics(
            name, "v33", truth, v33
        ))
        diagnostic_rows.extend(cross_component_diagnostics(
            name, "v34_invariant_f125_v200_m100", truth, candidate
        ))
    pd.DataFrame(diagnostic_rows).to_csv(
        args.output_dir / "subgroups.csv", index=False
    )

    # A direct component OR helps fake-voice/real-music, but can hurt the
    # opposite cross-component cell.  Test whether a same-file soft residual
    # can safely target only the voice-dominant cases.  This is an audit, not
    # part of v34.
    v34_banks = {}
    for name, (truth, v33, expert) in banks.items():
        candidate = v33.copy()
        for column in PREDICTION_COLUMNS.values():
            candidate[column] = fuse(
                v33[column], expert[column], selected_weights[column]
            )
        v34_banks[name] = truth, candidate
    base_metrics = {
        name: score_frame(truth.join(prediction))
        for name, (truth, prediction) in v34_banks.items()
    }
    route_rows = []
    for mode in (
        "global", "voice_dominant", "music_dominant",
        "voice_present_dominant", "predicted_mixed_voice_dominant",
        "voice_soft_t05", "voice_soft_t10", "voice_soft_t20",
        "voice_presence_soft", "mixed_voice_soft",
    ):
        for margin in (0., .5, 1., 1.5, 2.):
            if mode == "global" and margin:
                continue
            for file_weight in (.10, .20, .30, .40, .50, .60):
                record = {
                    "ROUTE": mode, "MARGIN": margin,
                    "FILE_WEIGHT": file_weight,
                }
                route_rates = []
                for name, (truth, prediction) in v34_banks.items():
                    gate = file_route(prediction, mode, margin)
                    evidence = component_file_evidence(
                        prediction.VOICE_FAKE_PROB,
                        prediction.MUSIC_FAKE_PROB,
                        prediction.VOICE_PRESENT_PROB,
                        prediction.MUSIC_PRESENT_PROB,
                        presence_logit_weight=0.,
                    )
                    candidate = prediction.copy()
                    candidate["FILE_FAKE_PROB"] = fuse(
                        prediction.FILE_FAKE_PROB, evidence,
                        file_weight * gate,
                    )
                    metrics = score_frame(truth.join(candidate))
                    record[f"{name}_DELTA"] = (
                        metrics["ADS"] - base_metrics[name]["ADS"]
                    )
                    record[f"{name}_FILE_EER"] = metrics["FILE_EER"]
                    record[f"{name}_ROUTE_RATE"] = gate.mean()
                    route_rates.append(gate.mean())
                deltas = [record[f"{name}_DELTA"] for name in v34_banks]
                record["MIN_DELTA"] = min(deltas)
                record["MEAN_DELTA"] = np.mean(deltas)
                record["MEAN_ROUTE_RATE"] = np.mean(route_rates)
                route_rows.append(record)
    file_router = pd.DataFrame(route_rows).sort_values(
        ["MIN_DELTA", "MEAN_DELTA"], ascending=False
    )
    file_router.to_csv(args.output_dir / "file_router_grid.csv", index=False)
    routed_diagnostics = []
    selected_routes = (
        ("global_or_w50", "global", 0., .50),
        ("voice_hard_m05_w50", "voice_dominant", .5, .50),
        ("voice_soft_t20_m20_w50", "voice_soft_t20", 2., .50),
    )
    for name, (truth, prediction) in v34_banks.items():
        routed_diagnostics.extend(cross_component_diagnostics(
            name, "v34", truth, prediction
        ))
        evidence = component_file_evidence(
            prediction.VOICE_FAKE_PROB, prediction.MUSIC_FAKE_PROB,
            prediction.VOICE_PRESENT_PROB, prediction.MUSIC_PRESENT_PROB,
            presence_logit_weight=0.,
        )
        for label, mode, margin, weight in selected_routes:
            candidate = prediction.copy()
            candidate["FILE_FAKE_PROB"] = fuse(
                prediction.FILE_FAKE_PROB, evidence,
                weight * file_route(prediction, mode, margin),
            )
            routed_diagnostics.extend(cross_component_diagnostics(
                name, label, truth, candidate
            ))
    pd.DataFrame(routed_diagnostics).to_csv(
        args.output_dir / "file_router_subgroups.csv", index=False
    )
    print(sweep.to_string(index=False))
    print("\nfile router\n", file_router.head(20).to_string(index=False))


if __name__ == "__main__":
    main()
