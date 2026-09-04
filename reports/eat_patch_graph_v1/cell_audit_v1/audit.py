#!/usr/bin/env python3
"""Reproduce the locked v18 versus EAT patch-graph cell audit.

This script reads frozen predictions only.  It never opens audio or fits a
threshold/model.  The 5% and 15% candidates match v44/v45: logit fusion is
applied to File and Music while Voice remains exactly v18.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve


ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]

from evaluate_diagnostic import official_eer, score_frame  # noqa: E402
from evaluate_wpt_v38_residual import reconstruct_v18_locked  # noqa: E402


DATASETS = {
    "factorial": "factorial_eval_1200_v2_holdout",
    "phone": "phone_factorial_1200_v1",
    "yue": "yue_cross_component_audit_v1",
}
TASKS = ("FILE", "VOICE", "MUSIC")
MODELS = ("v18", "patch", "fused05", "fused15")
EXPERT_PATH = (
    ROOT / "reports/eat_patch_graph_v1/seed01_balanced_fm_locked/predictions.csv"
)
OUTPUT_DIR = Path(__file__).resolve().parent


def logit(values) -> np.ndarray:
    values = np.clip(np.asarray(values, dtype=np.float64), 1e-6, 1 - 1e-6)
    return np.log(values) - np.log1p(-values)


def sigmoid(values) -> np.ndarray:
    return np.exp(-np.logaddexp(0, -np.asarray(values, dtype=np.float64)))


def auc(labels, scores) -> float:
    labels = np.asarray(labels, dtype=np.int64)
    if np.unique(labels).size < 2:
        return float("nan")
    return float(roc_auc_score(labels, scores))


def metric(frame: pd.DataFrame, model: str, task: str) -> dict[str, float | int]:
    label = f"{task}_FAKE"
    score = f"{model}_{task}"
    selected = frame.dropna(subset=[label, score])
    labels = selected[label].astype(int)
    scores = selected[score].astype(float)
    if labels.nunique() < 2:
        return {"N": len(selected), "POS": int(labels.sum()), "EER": np.nan,
                "AUC": np.nan}
    return {
        "N": len(selected), "POS": int(labels.sum()),
        "EER": official_eer(labels, scores), "AUC": auc(labels, scores),
    }


def load_frames() -> dict[str, pd.DataFrame]:
    expert_all = pd.read_csv(EXPERT_PATH, dtype={"ID": str})
    result = {}
    for short, dataset in DATASETS.items():
        truth, anchor = reconstruct_v18_locked(short)
        expert = expert_all.loc[expert_all.DATASET.eq(dataset)].set_index("ID")
        expert = expert.loc[anchor.index]
        frame = truth.copy()
        for task in TASKS:
            column = f"{task}_FAKE_PROB"
            frame[f"v18_{task}"] = anchor[column]
            frame[f"patch_{task}"] = expert[column]
            for weight, name in ((.05, "fused05"), (.15, "fused15")):
                if task == "VOICE":
                    frame[f"{name}_{task}"] = anchor[column]
                else:
                    frame[f"{name}_{task}"] = sigmoid(
                        (1 - weight) * logit(anchor[column])
                        + weight * logit(expert[column])
                    )
        result[short] = enrich_generators(short, frame)
    return result


def enrich_generators(short: str, frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    if short != "phone":
        frame["VOICE_GENERATOR_AUDIT"] = frame.get("VOICE_GENERATOR")
        frame["MUSIC_GENERATOR_AUDIT"] = frame.get("MUSIC_GENERATOR")
        return frame

    voice = pd.read_csv(
        ROOT / "data/eval/asvspoof_voice_v1/truth.csv", dtype={"ID": str}
    ).set_index("ID")
    music = pd.read_csv(
        ROOT / "data/eval/source_disjoint_music_v1/truth.csv", dtype={"ID": str}
    ).set_index("ID")
    voice_ids = frame.VOICE_SOURCE_ID.fillna(frame.PARENT_ID).astype(str)
    music_ids = frame.MUSIC_SOURCE_ID.fillna(frame.PARENT_ID).astype(str)
    frame["VOICE_GENERATOR_AUDIT"] = voice.GENERATOR.reindex(voice_ids).to_numpy()
    frame["VOICE_VOCODER_AUDIT"] = voice.VOCODER.reindex(voice_ids).to_numpy()
    frame["MUSIC_GENERATOR_AUDIT"] = music.GENERATOR.reindex(music_ids).to_numpy()
    return frame


def overall_table(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for dataset, frame in frames.items():
        for model in MODELS:
            scored = frame.copy()
            for task in TASKS:
                scored[f"{task}_FAKE_PROB"] = scored[f"{model}_{task}"]
            rows.append({"DATASET": dataset, "MODEL": model, **score_frame(scored)})
    return pd.DataFrame(rows)


def add_contrast(
    contrasts: list[tuple[str, str, pd.DataFrame]], name: str, task: str,
    frame: pd.DataFrame,
) -> None:
    label = f"{task}_FAKE"
    selected = frame.dropna(subset=[label])
    if len(selected) and selected[label].nunique() == 2:
        contrasts.append((name, task, selected))


def controlled_sets(short: str, frame: pd.DataFrame):
    contrasts: list[tuple[str, str, pd.DataFrame]] = []
    if short in {"factorial", "yue"}:
        for mode, task in (("voice_only", "VOICE"), ("music_only", "MUSIC")):
            add_contrast(contrasts, mode, task, frame.loc[frame.MIX_MODE.eq(mode)])
        if short == "yue":
            add_contrast(
                contrasts, "native_yue_vocal", "MUSIC",
                frame.loc[frame.MIX_MODE.eq("native_yue_vocal")],
            )
        modes = ("concurrent", "partial_overlap", "sequential")
    else:
        speech = frame.loc[frame.CONDITION.eq("speech_only")]
        music = frame.loc[frame.CONDITION.eq("source_disjoint")]
        add_contrast(contrasts, "speech_only", "VOICE", speech)
        add_contrast(contrasts, "speech_only_file", "FILE", speech)
        add_contrast(contrasts, "music_only", "MUSIC", music)
        add_contrast(contrasts, "music_only_file", "FILE", music)
        modes = ("simultaneous",)

    mode_column = "MIX_MODE" if short != "phone" else "CONDITION"
    for mode in modes:
        mixed = frame.loc[frame[mode_column].eq(mode)]
        for other in (0, 1):
            add_contrast(
                contrasts, f"{mode}|music_{other}", "VOICE",
                mixed.loc[mixed.MUSIC_FAKE.eq(other)],
            )
            add_contrast(
                contrasts, f"{mode}|voice_{other}", "MUSIC",
                mixed.loc[mixed.VOICE_FAKE.eq(other)],
            )
        rr = mixed.loc[mixed.VOICE_FAKE.eq(0) & mixed.MUSIC_FAKE.eq(0)]
        for voice, music, case in (
            (1, 0, "voice_fake+music_real"),
            (0, 1, "voice_real+music_fake"),
            (1, 1, "voice_fake+music_fake"),
        ):
            positive = mixed.loc[
                mixed.VOICE_FAKE.eq(voice) & mixed.MUSIC_FAKE.eq(music)
            ]
            add_contrast(
                contrasts, f"{mode}|file_rr_vs_{case}", "FILE",
                pd.concat([rr, positive]),
            )
    return contrasts


def controlled_table(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for dataset, frame in frames.items():
        for contrast, task, selected in controlled_sets(dataset, frame):
            for model in MODELS:
                rows.append({
                    "DATASET": dataset, "CONTRAST": contrast, "TASK": task,
                    "MODEL": model, **metric(selected, model, task),
                })
    return pd.DataFrame(rows)


def complementarity_table(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for dataset, frame in frames.items():
        for contrast, task, selected in controlled_sets(dataset, frame):
            label = f"{task}_FAKE"
            positive = selected.loc[selected[label].eq(1)]
            negative = selected.loc[selected[label].eq(0)]
            states = {}
            for model in MODELS:
                pos = positive[f"{model}_{task}"].to_numpy()[:, None]
                neg = negative[f"{model}_{task}"].to_numpy()[None, :]
                states[model] = pos > neg
            anchor = states["v18"]
            patch = states["patch"]
            row = {
                "DATASET": dataset, "CONTRAST": contrast, "TASK": task,
                "PAIRS": int(anchor.size),
                "V18_PAIR_ERROR": float(1 - anchor.mean()),
                "PATCH_PAIR_ERROR": float(1 - patch.mean()),
                "PATCH_CORRECTS_ALL": float((~anchor & patch).mean()),
                "PATCH_HARMS_ALL": float((anchor & ~patch).mean()),
                "PATCH_CORRECTS_V18_ERRORS": (
                    float((~anchor & patch).sum() / (~anchor).sum())
                    if (~anchor).any() else np.nan
                ),
            }
            for model in ("fused05", "fused15"):
                state = states[model]
                row[f"{model.upper()}_PAIR_ERROR"] = float(1 - state.mean())
            rows.append(row)
    return pd.DataFrame(rows)


def subgroup_table(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    columns = (
        "MIX_MODE", "CHANNEL", "CONDITION", "SOURCE_DATASET", "SNR_DB",
        "OVERLAP_FRACTION", "ORDER",
    )
    for dataset, frame in frames.items():
        for column in columns:
            if column not in frame:
                continue
            for value, group in frame.groupby(column, dropna=False):
                for task in TASKS:
                    for model in MODELS:
                        values = metric(group, model, task)
                        if np.isfinite(values["EER"]):
                            rows.append({
                                "DATASET": dataset, "GROUP": column,
                                "VALUE": str(value), "TASK": task,
                                "MODEL": model, **values,
                            })
    return pd.DataFrame(rows)


def conditional_generator_auc(
    frame: pd.DataFrame, task: str, generator: str,
) -> tuple[float, float, int]:
    label = f"{task}_FAKE"
    gen_column = f"{task}_GENERATOR_AUDIT"
    positive = frame.loc[frame[label].eq(1) & frame[gen_column].eq(generator)]
    negative = frame.loc[frame[label].eq(0)]
    other = "MUSIC_FAKE" if task == "VOICE" else "VOICE_FAKE"
    mode = "MIX_MODE" if "MIX_MODE" in frame else "CONDITION"
    strata = []
    for keys, pos in positive.groupby([mode, other], dropna=False):
        key_mode, key_other = keys
        mask = negative[mode].eq(key_mode)
        mask &= negative[other].eq(key_other) | (
            negative[other].isna() & pd.isna(key_other)
        )
        neg = negative.loc[mask]
        if len(neg):
            strata.append((pos, neg))
    return positive, negative, strata


def generator_table(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for dataset, frame in frames.items():
        for task in ("VOICE", "MUSIC"):
            gen_column = f"{task}_GENERATOR_AUDIT"
            if gen_column not in frame:
                continue
            label = f"{task}_FAKE"
            generators = sorted(frame.loc[frame[label].eq(1), gen_column].dropna().unique())
            for generator in generators:
                positive, negative, strata = conditional_generator_auc(
                    frame, task, str(generator)
                )
                if not len(positive) or not len(negative):
                    continue
                pooled = pd.concat([positive, negative])
                for model in MODELS:
                    conditional_aucs = []
                    conditional_pairs = 0
                    for pos, neg in strata:
                        ps = pos[f"{model}_{task}"].to_numpy()[:, None]
                        ns = neg[f"{model}_{task}"].to_numpy()[None, :]
                        conditional_aucs.append(float((ps > ns).mean()))
                        conditional_pairs += ps.size * ns.size
                    rows.append({
                        "DATASET": dataset, "TASK": task,
                        "GENERATOR": str(generator), "MODEL": model,
                        "POS_N": len(positive), "REAL_N": len(negative),
                        "MATCHED_STRATA": len(strata),
                        "MATCHED_PAIRS": conditional_pairs,
                        "CONDITIONAL_MACRO_AUC": (
                            float(np.mean(conditional_aucs)) if conditional_aucs else np.nan
                        ),
                        **metric(pooled, model, task),
                    })
    return pd.DataFrame(rows)


def phone_vocoder_table(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    frame = frames["phone"]
    negative = frame.loc[frame.VOICE_FAKE.eq(0)]
    rows = []
    for vocoder, positive in frame.loc[frame.VOICE_FAKE.eq(1)].groupby(
        "VOICE_VOCODER_AUDIT"
    ):
        selected = pd.concat([positive, negative])
        for model in MODELS:
            rows.append({
                "VOCODER": str(vocoder), "MODEL": model,
                "FAKE_N": len(positive), "REAL_N": len(negative),
                **metric(selected, model, "VOICE"),
            })
    return pd.DataFrame(rows)


def eer_threshold(labels, scores) -> float:
    fpr, tpr, thresholds = roc_curve(
        np.asarray(labels, dtype=int), np.asarray(scores, dtype=float),
        pos_label=1, drop_intermediate=False,
    )
    idx = int(np.argmin(np.abs(fpr - (1 - tpr))))
    return float(thresholds[idx])


def cell_error_table(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for dataset, frame in frames.items():
        cell_column = "EVAL_CELL" if "EVAL_CELL" in frame else "CONDITION"
        for model in MODELS:
            for task in TASKS:
                label = f"{task}_FAKE"
                score = f"{model}_{task}"
                available = frame.dropna(subset=[label, score])
                if available[label].nunique() < 2:
                    continue
                threshold = eer_threshold(available[label], available[score])
                for cell, group in available.groupby(cell_column):
                    predicted = group[score].ge(threshold).astype(int)
                    rows.append({
                        "DATASET": dataset, "CELL": str(cell), "TASK": task,
                        "MODEL": model, "N": len(group),
                        "LABEL_MEAN": float(group[label].mean()),
                        "GLOBAL_EER_THRESHOLD": threshold,
                        "ERROR_RATE": float(predicted.ne(group[label].astype(int)).mean()),
                        "SCORE_Q10": float(group[score].quantile(.1)),
                        "SCORE_MEDIAN": float(group[score].median()),
                        "SCORE_Q90": float(group[score].quantile(.9)),
                    })
    return pd.DataFrame(rows)


def component_or_table(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Measure whether a monotone component OR can repair the File head."""
    rows = []
    for base in ("v18", "fused05", "fused15"):
        for weight in (0, .05, .10, .15, .20, .30, .50, 1.0):
            for dataset, frame in frames.items():
                component_or = 1 - (
                    (1 - frame[f"{base}_VOICE"]) * (1 - frame[f"{base}_MUSIC"])
                )
                prediction = frame.copy()
                prediction["FILE_FAKE_PROB"] = sigmoid(
                    (1 - weight) * logit(frame[f"{base}_FILE"])
                    + weight * logit(component_or)
                )
                prediction["VOICE_FAKE_PROB"] = frame[f"{base}_VOICE"]
                prediction["MUSIC_FAKE_PROB"] = frame[f"{base}_MUSIC"]
                rows.append({
                    "BASE": base, "OR_WEIGHT": weight, "DATASET": dataset,
                    **score_frame(prediction),
                })
    return pd.DataFrame(rows)


def main() -> None:
    frames = load_frames()
    outputs = {
        "overall.csv": overall_table(frames),
        "controlled_contrasts.csv": controlled_table(frames),
        "pair_complementarity.csv": complementarity_table(frames),
        "subgroups.csv": subgroup_table(frames),
        "generator_contrasts.csv": generator_table(frames),
        "phone_vocoder_contrasts.csv": phone_vocoder_table(frames),
        "cell_error_rates.csv": cell_error_table(frames),
        "component_or_sweep.csv": component_or_table(frames),
    }
    for name, table in outputs.items():
        table.to_csv(OUTPUT_DIR / name, index=False)
        print(f"{name}: {len(table)} rows")


if __name__ == "__main__":
    main()
