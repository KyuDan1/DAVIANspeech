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
    print(sweep.to_string(index=False))


if __name__ == "__main__":
    main()
