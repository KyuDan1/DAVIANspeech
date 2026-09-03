#!/usr/bin/env python3
"""Select safe WPT Voice/File residuals on dev and audit once over v38."""

from __future__ import annotations

import argparse
import json
import sys
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from evaluate_codec_invariant_fusion import (  # noqa: E402
    AUDITS, fuse, reconstruct_v18,
)
from evaluate_diagnostic import score_frame  # noqa: E402
from evaluate_hierarchical_eat_fusion import apply as apply_hierarchical  # noqa: E402
from evaluate_presence_weighted_file_fusion import reconstruct_dev_v18  # noqa: E402


LOCKED = {
    "factorial": "factorial_eval_1200_v2_holdout",
    "phone": "phone_factorial_1200_v1",
    "yue": "yue_cross_component_audit_v1",
}


def logit(values) -> np.ndarray:
    values = np.clip(np.asarray(values, dtype=np.float64), 1e-5, 1 - 1e-5)
    return np.log(values) - np.log1p(-values)


def sigmoid(values) -> np.ndarray:
    return np.exp(-np.logaddexp(0, -np.asarray(values, dtype=np.float64)))


def indexed(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, dtype={"ID": str}).set_index("ID")


def selected(
    frame: pd.DataFrame, dataset: str, ids: pd.Index,
) -> pd.DataFrame:
    block = frame.loc[frame.DATASET.eq(dataset)].set_index("ID")
    missing = ids.difference(block.index)
    if len(missing):
        raise ValueError(f"{dataset}: {len(missing)} predictions missing")
    return block.loc[ids]


def reconstruct_dev_anchors() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    temporal = pd.read_csv(
        ROOT / "reports/temporal_dual_domain_hybrid/ensemble_dev/predictions.csv",
        dtype={"ID": str},
    )
    fakeprint = pd.read_csv(
        ROOT / "reports/modern_fakeprint_v1/independent_margin_scores.csv",
        dtype={"ID": str},
    )
    invariant = pd.read_csv(
        ROOT / "reports/invariant_dual_domain_v2/dec_n1/dev_predictions.csv",
        dtype={"ID": str},
    )
    truth, v18 = reconstruct_dev_v18(temporal, fakeprint, invariant)
    hierarchical = pd.read_csv(
        ROOT / "reports/hierarchical_eat_music_v1/ensemble02_dev/predictions.csv",
        dtype={"ID": str},
    )
    h = selected(
        hierarchical, "factorial_eval_1200_v2_dev", v18.index
    ).HIERARCHICAL_EAT_MUSIC_PROB
    segment_paths = (
        ROOT / "reports/segmental_eat_music_v2/content_seed00/dev_predictions.csv",
        ROOT / "reports/segmental_eat_music_v2/content_seed02/dev_predictions.csv",
    )
    segment = []
    for path in segment_paths:
        frame = pd.read_csv(path, dtype={"ID": str})
        # The saved segmental development table intentionally contains only
        # MUSIC_PRESENT=1 examples.  In deployed v37 every file is scored, but
        # Music EER ignores the missing voice-only rows and the outer File
        # fusion is presence-gated.  Use the hierarchical score as the neutral
        # fallback for those rows instead of silently dropping 50 examples.
        block = frame.loc[
            frame.DATASET.eq("factorial_eval_1200_v2_dev")
        ].set_index("ID")
        unknown = block.index.difference(v18.index)
        if len(unknown):
            raise ValueError(f"segmental dev has {len(unknown)} unknown IDs")
        aligned = block.HIERARCHICAL_EAT_MUSIC_PROB.reindex(v18.index)
        aligned = aligned.fillna(pd.Series(h, index=v18.index))
        segment.append(logit(aligned))
    s = np.mean(segment, axis=0)
    inner = sigmoid(.75 * logit(h) + .25 * s)
    phone = indexed(
        ROOT / "reports/phone_presence_probe_v1/factorial_eval_1200_v2_router.csv"
    ).loc[v18.index, "IS_PHONE"].to_numpy(bool)
    v37 = apply_hierarchical(
        v18, inner, music_weight=.20, file_weight=.10,
        presence_gate=.50, phone_mask=phone,
        phone_music_weight=.30, phone_file_weight=.30,
    )
    return truth, v18, v37


def reconstruct_v18_locked(short: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    temporal = pd.read_csv(
        ROOT / "reports/temporal_dual_domain_hybrid/ensemble_audit/predictions.csv",
        dtype={"ID": str},
    )
    fakeprint = pd.read_csv(
        ROOT / "reports/modern_fakeprint_v1/independent_margin_scores.csv",
        dtype={"ID": str},
    )
    invariant = pd.read_csv(
        ROOT / "reports/invariant_dual_domain_v2/dec_n1_audit/predictions.csv",
        dtype={"ID": str},
    )
    return reconstruct_v18(AUDITS[short], temporal, fakeprint, invariant)


def add_v38_music(
    anchor: pd.DataFrame, unified: pd.DataFrame, dataset: str,
) -> pd.DataFrame:
    expert = selected(unified, dataset, anchor.index)
    result = anchor.copy()
    result["MUSIC_FAKE_PROB"] = fuse(
        result.MUSIC_FAKE_PROB, expert.MUSIC_FAKE_PROB, .20
    )
    return result


def add_wpt(
    anchor: pd.DataFrame, expert: pd.DataFrame,
    phone_probability: np.ndarray,
    normal_voice: float, phone_voice: float,
    normal_file: float, phone_file: float,
) -> pd.DataFrame:
    result = anchor.copy()
    voice_weight = normal_voice + (
        phone_voice - normal_voice
    ) * phone_probability
    file_weight = normal_file + (
        phone_file - normal_file
    ) * phone_probability
    result["VOICE_FAKE_PROB"] = sigmoid(
        (1 - voice_weight) * logit(result.VOICE_FAKE_PROB)
        + voice_weight * logit(expert.VOICE_FAKE_PROB)
    )
    result["FILE_FAKE_PROB"] = sigmoid(
        (1 - file_weight) * logit(result.FILE_FAKE_PROB)
        + file_weight * logit(expert.FILE_FAKE_PROB)
    )
    return result


def grouped_score(
    truth: pd.DataFrame, prediction: pd.DataFrame,
) -> tuple[dict[str, float], float, list[dict[str, object]]]:
    overall = score_frame(truth.join(prediction))
    groups = []
    if "CHANNEL" not in truth:
        return overall, float(overall["ADS"]), [{"CHANNEL": "ALL", **overall}]
    for channel, ids in truth.groupby("CHANNEL").groups.items():
        try:
            metric = score_frame(truth.loc[ids].join(prediction.loc[ids]))
        except ValueError:
            continue
        if np.isfinite(metric["ADS"]):
            groups.append({"CHANNEL": channel, **metric})
    worst = min(item["ADS"] for item in groups)
    return overall, worst, groups


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wpt-predictions", type=Path, required=True)
    parser.add_argument(
        "--unified-predictions", type=Path,
        default=ROOT / "reports/unified_dual_ssl_v1/joint_router_banks/predictions.csv",
    )
    parser.add_argument(
        "--phone-predictions", type=Path,
        default=ROOT / "reports/router_training_v1/wpt_router_phone_predictions.csv",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--anchor", choices=("v18", "v38"), default="v18",
        help=(
            "Use the leaderboard-validated v18 anchor by default; v38 is an "
            "experimental nested Music residual and is evaluated separately."
        ),
    )
    parser.add_argument(
        "--weights", type=float, nargs="+",
        default=[0, .025, .05, .075, .10, .15, .20],
    )
    parser.add_argument(
        "--strategies", nargs="+",
        choices=("fixed_moe", "phone_soft_router"),
        default=["fixed_moe", "phone_soft_router"],
    )
    args = parser.parse_args()
    wpt = pd.read_csv(args.wpt_predictions, dtype={"ID": str})
    unified = pd.read_csv(args.unified_predictions, dtype={"ID": str})
    phone = pd.read_csv(args.phone_predictions, dtype={"ID": str})

    truth, v18, v37 = reconstruct_dev_anchors()
    baseline_prediction = (
        v18 if args.anchor == "v18"
        else add_v38_music(v37, unified, "factorial_eval_1200_v2_dev")
    )
    wpt_dev = selected(
        wpt, "factorial_eval_1200_v2_dev", baseline_prediction.index
    )
    phone_dev = selected(
        phone, "factorial_eval_1200_v2_dev", baseline_prediction.index
    ).PHONE_PROB.to_numpy(np.float64)
    baseline, baseline_worst, baseline_groups = grouped_score(
        truth, baseline_prediction
    )
    sweep = []
    settings = list(product(args.weights, repeat=2))
    # First compare a global MoE; then allow only the independently validated
    # phone signal to interpolate a second pair of bounded weights.
    candidates = []
    if "fixed_moe" in args.strategies:
        candidates.extend(
            ("fixed_moe", voice, voice, file, file)
            for voice, file in settings
        )
    if "phone_soft_router" in args.strategies:
        candidates.extend(
            ("phone_soft_router", voice, phone_voice, file, phone_file)
            for (voice, file), (phone_voice, phone_file)
            in product(settings, settings)
        )
    for method, voice, phone_voice, file, phone_file in candidates:
        prediction = add_wpt(
            baseline_prediction, wpt_dev, phone_dev,
            voice, phone_voice, file, phone_file,
        )
        metric, worst, groups = grouped_score(truth, prediction)
        group_delta = min(
            item["ADS"] - next(
                base["ADS"] for base in baseline_groups
                if base["CHANNEL"] == item["CHANNEL"]
            )
            for item in groups
        )
        sweep.append({
            "METHOD": method, "VOICE_WEIGHT": voice,
            "PHONE_VOICE_WEIGHT": phone_voice, "FILE_WEIGHT": file,
            "PHONE_FILE_WEIGHT": phone_file,
            "ADS": metric["ADS"], "WORST_CHANNEL_ADS": worst,
            "MIN_CHANNEL_DELTA": group_delta,
            "ROBUST_SELECTION": .5 * metric["ADS"] + .5 * worst,
        })
    sweep = pd.DataFrame(sweep)
    safe = sweep.loc[sweep.MIN_CHANNEL_DELTA >= -1e-12]
    pool = safe if len(safe) else sweep
    chosen = pool.sort_values(
        ["ROBUST_SELECTION", "ADS", "VOICE_WEIGHT", "FILE_WEIGHT"],
        ascending=[False, False, True, True],
    ).iloc[0]

    locked_rows = []
    root = ROOT / "reports/segmental_eat_music_v2/nested_v36b_locked"
    for short, dataset in LOCKED.items():
        if args.anchor == "v18":
            locked_truth, locked_anchor = reconstruct_v18_locked(short)
        else:
            locked_truth = indexed(root / f"{short}_truth.csv")
            v37_locked = indexed(
                root / f"{short}_predictions.csv"
            ).loc[locked_truth.index]
            locked_anchor = add_v38_music(v37_locked, unified, dataset)
        wpt_locked = selected(wpt, dataset, locked_anchor.index)
        phone_locked = selected(
            phone, dataset, locked_anchor.index
        ).PHONE_PROB.to_numpy(np.float64)
        candidate = add_wpt(
            locked_anchor, wpt_locked, phone_locked,
            float(chosen.VOICE_WEIGHT), float(chosen.PHONE_VOICE_WEIGHT),
            float(chosen.FILE_WEIGHT), float(chosen.PHONE_FILE_WEIGHT),
        )
        for method, prediction in (
            (args.anchor, locked_anchor), ("wpt_residual", candidate)
        ):
            locked_rows.append({
                "DATASET": short, "METHOD": method,
                **score_frame(locked_truth.join(prediction)),
            })
    locked = pd.DataFrame(locked_rows)
    baseline_locked = locked.loc[
        locked.METHOD.eq(args.anchor)
    ].set_index("DATASET").ADS
    locked["DELTA_VS_ANCHOR"] = [
        row.ADS - baseline_locked[row.DATASET] for row in locked.itertuples()
    ]
    accepted = float(
        locked.loc[
            locked.METHOD.eq("wpt_residual"), "DELTA_VS_ANCHOR"
        ].min()
    ) >= 0

    args.output_dir.mkdir(parents=True, exist_ok=True)
    sweep.to_csv(args.output_dir / "dev_sweep.csv", index=False)
    locked.to_csv(args.output_dir / "locked_audit.csv", index=False)
    summary = {
        "anchor": args.anchor,
        "dev_anchor_ads": baseline["ADS"],
        "dev_anchor_worst_channel_ads": baseline_worst,
        "selected": chosen.to_dict(),
        "locked_all_non_regressing": accepted,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    print(locked.to_string(index=False))


if __name__ == "__main__":
    main()
