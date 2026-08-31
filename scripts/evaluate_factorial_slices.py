#!/usr/bin/env python3
"""Evaluate factorial cells without assuming the private-set class mixture."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.evaluate_diagnostic import PREDICTION_COLUMNS, official_eer, score_frame


def eer(frame: pd.DataFrame, score: str, label: str) -> float:
    selected = frame.dropna(subset=[score, label])
    if selected.empty or selected[label].nunique() < 2:
        return float("nan")
    return official_eer(selected[label].astype(int), selected[score].astype(float))


def contrast(name: str, frame: pd.DataFrame, score: str, label: str) -> dict:
    return {
        "KIND": "contrast_eer", "EVALUATION": name, "N": len(frame),
        "SCORE_COLUMN": score, "LABEL_COLUMN": label,
        "EER": eer(frame, score, label),
    }


def evaluate(prediction_path: Path, truth_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    prediction = pd.read_csv(prediction_path, dtype={"ID": str})
    truth = pd.read_csv(truth_path, dtype={"ID": str})
    frame = truth.merge(
        prediction[["ID", *PREDICTION_COLUMNS]], on="ID", validate="one_to_one"
    )
    summaries: list[dict] = []
    distributions: list[dict] = []
    split_frames = [("all", frame), *frame.groupby("SPLIT", sort=True)]
    for split, part in split_frames:
        for mode, group in part.groupby("MIX_MODE", sort=True):
            summaries.append({
                "SPLIT": split, "KIND": "mode_score", "EVALUATION": mode,
                **score_frame(group),
            })

        voice_only = part[part.MIX_MODE == "voice_only"]
        music_only = part[part.MIX_MODE == "music_only"]
        summaries.append({"SPLIT": split, **contrast(
            "voice_only_real_vs_fake", voice_only,
            "VOICE_FAKE_PROB", "VOICE_FAKE")})
        summaries.append({"SPLIT": split, **contrast(
            "music_only_real_vs_fake", music_only,
            "MUSIC_FAKE_PROB", "MUSIC_FAKE")})

        for mode in ("concurrent", "partial_overlap", "sequential"):
            mixed = part[part.MIX_MODE == mode]
            for music_label in (0, 1):
                controlled = mixed[mixed.MUSIC_FAKE == music_label]
                summaries.append({"SPLIT": split, **contrast(
                    f"{mode}__voice_eer__music_{'fake' if music_label else 'real'}",
                    controlled, "VOICE_FAKE_PROB", "VOICE_FAKE")})
            for voice_label in (0, 1):
                controlled = mixed[mixed.VOICE_FAKE == voice_label]
                summaries.append({"SPLIT": split, **contrast(
                    f"{mode}__music_eer__voice_{'fake' if voice_label else 'real'}",
                    controlled, "MUSIC_FAKE_PROB", "MUSIC_FAKE")})

            rr = mixed[(mixed.VOICE_FAKE == 0) & (mixed.MUSIC_FAKE == 0)]
            for voice_label, music_label, case in (
                (1, 0, "fake_voice_real_music"),
                (0, 1, "real_voice_fake_music"),
                (1, 1, "fake_voice_fake_music"),
            ):
                positive = mixed[
                    (mixed.VOICE_FAKE == voice_label)
                    & (mixed.MUSIC_FAKE == music_label)
                ]
                controlled = pd.concat([rr, positive], ignore_index=True)
                summaries.append({"SPLIT": split, **contrast(
                    f"{mode}__file_rr_vs_{case}", controlled,
                    "FILE_FAKE_PROB", "FILE_FAKE")})

        for cell, group in part.groupby("EVAL_CELL", sort=True):
            record = {"SPLIT": split, "EVAL_CELL": cell, "N": len(group)}
            for column in PREDICTION_COLUMNS:
                values = pd.to_numeric(group[column], errors="coerce").dropna()
                record[f"{column}_Q10"] = values.quantile(0.10)
                record[f"{column}_MEDIAN"] = values.median()
                record[f"{column}_Q90"] = values.quantile(0.90)
            distributions.append(record)
    return pd.DataFrame(summaries), pd.DataFrame(distributions)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("prediction", type=Path)
    parser.add_argument("truth", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    summary, distributions = evaluate(args.prediction, args.truth)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.output_dir / "factorial_contrasts.csv", index=False)
    distributions.to_csv(args.output_dir / "factorial_cell_distributions.csv", index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
