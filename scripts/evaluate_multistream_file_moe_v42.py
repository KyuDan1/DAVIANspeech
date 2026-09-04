#!/usr/bin/env python3
"""Select and audit multistream fixed-MoE/router residuals over canonical v41."""

from __future__ import annotations

import json
import sys
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from evaluate_diagnostic import official_eer, score_frame  # noqa: E402
from evaluate_wpt_v38_residual import (  # noqa: E402
    add_v38_music, indexed, logit, reconstruct_dev_anchors, selected, sigmoid,
)


DATASETS = {
    "factorial": "factorial_eval_1200_v2_holdout",
    "phone": "phone_factorial_1200_v1",
    "yue": "yue_cross_component_audit_v1",
}


def fuse(anchor, expert, weight) -> np.ndarray:
    weight = np.asarray(weight, dtype=np.float64)
    return sigmoid((1 - weight) * logit(anchor) + weight * logit(expert))


def canonical_v41(
    v37: pd.DataFrame,
    dataset: str,
    unified: pd.DataFrame,
    wpt: pd.DataFrame,
    multistream: pd.DataFrame | None = None,
    file_weight: float | np.ndarray = 0.0,
    music_weight: float | np.ndarray = 0.0,
) -> pd.DataFrame:
    ids = v37.index
    unified_bank = selected(unified, dataset, ids)
    wpt_bank = selected(wpt, dataset, ids)
    v38 = add_v38_music(v37, unified, dataset)
    result = v38.copy()
    result["VOICE_FAKE_PROB"] = fuse(
        v38.VOICE_FAKE_PROB,
        fuse(unified_bank.VOICE_FAKE_PROB, wpt_bank.VOICE_FAKE_PROB, .90),
        .10,
    )
    file_expert = wpt_bank.FILE_FAKE_PROB
    if multistream is not None:
        stream = selected(multistream, dataset, ids)
        file_expert = fuse(file_expert, stream.FILE_FAKE_PROB, file_weight)
        result["MUSIC_FAKE_PROB"] = sigmoid(
            logit(v38.MUSIC_FAKE_PROB)
            + .20 * np.asarray(music_weight, dtype=np.float64)
            * (
                logit(stream.MUSIC_FAKE_PROB)
                - logit(unified_bank.MUSIC_FAKE_PROB)
            )
        )
    result["FILE_FAKE_PROB"] = fuse(
        v38.FILE_FAKE_PROB,
        fuse(unified_bank.FILE_FAKE_PROB, file_expert, .80),
        .75,
    )
    return result


def task_eer(
    truth: pd.DataFrame, prediction: pd.DataFrame, task: str,
    ids: pd.Index | None = None,
) -> float:
    if ids is None:
        ids = truth.index
    frame = truth.loc[ids]
    if task in {"VOICE", "MUSIC"}:
        frame = frame.loc[frame[f"{task}_PRESENT"].eq(1)]
    if frame[f"{task}_FAKE"].nunique() != 2:
        return float("nan")
    return official_eer(
        frame[f"{task}_FAKE"], prediction.loc[frame.index, f"{task}_FAKE_PROB"],
    )


def channel_summary(
    truth: pd.DataFrame, prediction: pd.DataFrame, task: str,
) -> tuple[float, float, dict[str, float]]:
    overall = task_eer(truth, prediction, task)
    groups = {
        str(channel): task_eer(truth, prediction, task, ids)
        for channel, ids in truth.groupby("CHANNEL").groups.items()
    }
    finite = [value for value in groups.values() if np.isfinite(value)]
    return overall, max(finite), groups


def main() -> None:
    output = ROOT / "reports/multistream_prompt_spectra_v1/v42_router_vs_moe"
    output.mkdir(parents=True, exist_ok=True)
    unified = pd.read_csv(
        ROOT / "reports/unified_dual_ssl_v1/joint_router_banks/predictions.csv",
        dtype={"ID": str},
    )
    wpt = pd.read_csv(
        ROOT
        / "reports/wpt_spectra_v1/seed06_views5_file_t2_exact/predictions.csv",
        dtype={"ID": str},
    )
    stream3_dev = pd.read_csv(
        ROOT / "reports/multistream_prompt_spectra_v1/seed_20260908/dev_predictions.csv",
        dtype={"ID": str},
    )
    stream3_locked = pd.read_csv(
        ROOT
        / "reports/multistream_prompt_spectra_v1/seed_20260908_locked/predictions.csv",
        dtype={"ID": str},
    )
    stream2 = pd.read_csv(
        ROOT
        / "reports/multistream_prompt_spectra_v1/seed_20260908_views2/predictions.csv",
        dtype={"ID": str},
    )
    truth, _v18, v37 = reconstruct_dev_anchors()
    dataset = "factorial_eval_1200_v2_dev"
    baseline = canonical_v41(v37, dataset, unified, wpt)
    base_file, base_worst, base_groups = channel_summary(
        truth, baseline, "FILE"
    )

    fixed_rows = []
    weights = np.linspace(0, 1, 21)
    for views, stream in ((2, stream2), (3, stream3_dev)):
        for weight in weights:
            candidate = canonical_v41(
                v37, dataset, unified, wpt, stream, file_weight=weight,
            )
            eer, worst, groups = channel_summary(truth, candidate, "FILE")
            fixed_rows.append({
                "VIEWS": views, "WEIGHT": weight, "FILE_EER": eer,
                "WORST_CHANNEL_EER": worst,
                "MIN_CHANNEL_DELTA": min(
                    base_groups[name] - value for name, value in groups.items()
                    if np.isfinite(value)
                ),
            })
    fixed_sweep = pd.DataFrame(fixed_rows)
    fixed_sweep.to_csv(output / "fixed_file_sweep.csv", index=False)
    selected_fixed = fixed_sweep.loc[fixed_sweep.VIEWS.eq(2)].sort_values(
        ["FILE_EER", "WEIGHT"], ascending=[True, True],
    ).iloc[0]

    phone_dev = indexed(
        ROOT / "reports/phone_presence_probe_v1/factorial_eval_1200_v2_router.csv"
    ).loc[truth.index, "PHONE_PROB"].to_numpy(np.float64)
    router_rows = []
    router_weights = np.linspace(0, 1, 11)
    for views, stream, method, normal, phone in (
        (views, stream, method, normal, phone)
        for views, stream in ((2, stream2), (3, stream3_dev))
        for method in ("soft", "hard")
        for normal, phone in product(router_weights, repeat=2)
    ):
        weight = (
            normal + (phone - normal) * phone_dev
            if method == "soft" else np.where(phone_dev >= .5, phone, normal)
        )
        candidate = canonical_v41(
            v37, dataset, unified, wpt, stream, file_weight=weight,
        )
        eer, worst, groups = channel_summary(truth, candidate, "FILE")
        router_rows.append({
            "VIEWS": views, "METHOD": method, "NORMAL_WEIGHT": normal,
            "PHONE_WEIGHT": phone, "FILE_EER": eer,
            "WORST_CHANNEL_EER": worst,
            "MIN_CHANNEL_DELTA": min(
                base_groups[name] - value for name, value in groups.items()
                if np.isfinite(value)
            ),
            "ROBUST_OBJECTIVE": .5 * eer + .5 * worst,
        })
    router_sweep = pd.DataFrame(router_rows)
    router_sweep.to_csv(output / "phone_router_sweep.csv", index=False)
    safe_router = router_sweep.loc[
        (router_sweep.FILE_EER <= base_file + 1e-12)
        & (router_sweep.MIN_CHANNEL_DELTA >= -1e-12)
    ].copy()
    # Prefer continuous soft routing when EER is tied. Hard switches amplify
    # small router calibration errors and have no development advantage here.
    safe_router["METHOD_PREFERENCE"] = safe_router.METHOD.map({
        "soft": 0, "hard": 1,
    })
    selected_router = safe_router.sort_values([
        "ROBUST_OBJECTIVE", "FILE_EER", "METHOD_PREFERENCE", "NORMAL_WEIGHT",
        "PHONE_WEIGHT", "VIEWS",
    ]).iloc[0].drop(labels="METHOD_PREFERENCE")

    music_rows = []
    base_music, base_music_worst, base_music_groups = channel_summary(
        truth, baseline, "MUSIC"
    )
    for weight in weights:
        candidate = canonical_v41(
            v37, dataset, unified, wpt, stream3_dev,
            music_weight=weight,
        )
        eer, worst, groups = channel_summary(truth, candidate, "MUSIC")
        music_rows.append({
            "WEIGHT": weight, "MUSIC_EER": eer,
            "WORST_CHANNEL_EER": worst,
            "MIN_CHANNEL_DELTA": min(
                base_music_groups[name] - value
                for name, value in groups.items() if np.isfinite(value)
            ),
            "ROBUST_OBJECTIVE": .5 * eer + .5 * worst,
        })
    music_sweep = pd.DataFrame(music_rows)
    music_sweep.to_csv(output / "music_sweep.csv", index=False)
    safe_music = music_sweep.loc[
        (music_sweep.MUSIC_EER <= base_music + 1e-12)
        & (music_sweep.MIN_CHANNEL_DELTA >= -1e-12)
    ]
    selected_music = safe_music.sort_values([
        "ROBUST_OBJECTIVE", "MUSIC_EER", "WEIGHT",
    ]).iloc[0]

    # All weights above are now frozen. Locked banks are used only as a
    # one-shot acceptance gate, never to choose or revise those weights.
    router_predictions = pd.read_csv(
        ROOT / "reports/router_training_v1/wpt_router_phone_predictions.csv",
        dtype={"ID": str},
    )
    locked_root = ROOT / "reports/segmental_eat_music_v2/nested_v36b_locked"
    audit_rows = []
    for short, locked_dataset in DATASETS.items():
        locked_truth = indexed(locked_root / f"{short}_truth.csv")
        locked_v37 = indexed(
            locked_root / f"{short}_predictions.csv"
        ).loc[locked_truth.index]
        phone = selected(
            router_predictions, locked_dataset, locked_truth.index
        ).PHONE_PROB.to_numpy(np.float64)
        router_weight = (
            float(selected_router.NORMAL_WEIGHT)
            + (
                float(selected_router.PHONE_WEIGHT)
                - float(selected_router.NORMAL_WEIGHT)
            ) * phone
        )
        router_stream = (
            stream2 if int(selected_router.VIEWS) == 2 else stream3_locked
        )
        candidates = {
            "v41": canonical_v41(
                locked_v37, locked_dataset, unified, wpt,
            ),
            "fixed_file_2view": canonical_v41(
                locked_v37, locked_dataset, unified, wpt, stream2,
                file_weight=float(selected_fixed.WEIGHT),
            ),
            "phone_soft_router_selected": canonical_v41(
                locked_v37, locked_dataset, unified, wpt, router_stream,
                file_weight=router_weight,
            ),
            "fixed_music_3view": canonical_v41(
                locked_v37, locked_dataset, unified, wpt, stream3_locked,
                music_weight=float(selected_music.WEIGHT),
            ),
        }
        baseline_ads = score_frame(locked_truth.join(candidates["v41"]))["ADS"]
        for method, prediction in candidates.items():
            metric = score_frame(locked_truth.join(prediction))
            audit_rows.append({
                "DATASET": short, "METHOD": method,
                "FILE_EER": metric["FILE_EER"],
                "VOICE_EER": metric["VOICE_EER"],
                "MUSIC_EER": metric["MUSIC_EER"],
                "ADS": metric["ADS"], "DELTA_VS_V41": metric["ADS"] - baseline_ads,
            })
    audit = pd.DataFrame(audit_rows)
    audit.to_csv(output / "locked_audit.csv", index=False)
    fixed_delta = audit.loc[
        audit.METHOD.eq("fixed_file_2view"), "DELTA_VS_V41"
    ]
    summary = {
        "baseline_dev_file_eer": base_file,
        "selected_fixed_file": selected_fixed.to_dict(),
        "selected_phone_router": selected_router.to_dict(),
        "selected_music": selected_music.to_dict(),
        "accepted": "fixed_file_2view" if fixed_delta.min() >= -1e-12 else "v41",
        "locked_min_delta": float(fixed_delta.min()),
        "locked_mean_delta": float(fixed_delta.mean()),
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))
    print(audit.to_string(index=False))


if __name__ == "__main__":
    main()
