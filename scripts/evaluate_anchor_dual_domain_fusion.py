#!/usr/bin/env python3
"""Select robust per-task fusion of the verified anchor and dual-domain head."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from evaluate_diagnostic import official_eer  # noqa: E402


TASKS = {
    "FILE": ("FILE_FAKE", "FILE_FAKE_PROB", 0.50),
    "VOICE": ("VOICE_FAKE", "VOICE_FAKE_PROB", 0.20),
    "MUSIC": ("MUSIC_FAKE", "MUSIC_FAKE_PROB", 0.30),
}


def logit(value: np.ndarray) -> np.ndarray:
    value = np.clip(value, 1e-5, 1 - 1e-5)
    return np.log(value) - np.log1p(-value)


def fuse(anchor: np.ndarray, head: np.ndarray, weight: float, mode: str) -> np.ndarray:
    if mode == "probability":
        return (1 - weight) * anchor + weight * head
    return 1 / (1 + np.exp(-((1 - weight) * logit(anchor) + weight * logit(head))))


def task_eer(frame: pd.DataFrame, task: str, score: np.ndarray) -> float:
    target, _, _ = TASKS[task]
    mask = np.ones(len(frame), dtype=bool)
    if task != "FILE":
        mask = frame[f"{task}_PRESENT"].eq(1).to_numpy()
    return official_eer(frame.loc[mask, target], score[mask])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--truth", type=Path, required=True)
    parser.add_argument("--anchor", type=Path, required=True)
    parser.add_argument("--head", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--frozen-config", type=Path,
        help="Apply modes/weights from a prior dev summary without reselection.",
    )
    parser.add_argument(
        "--weights", type=float, nargs="+",
        default=[0, .025, .05, .075, .10, .15, .20, .25, .30, .40, .50],
    )
    parser.add_argument(
        "--max-bank-regression", type=float, default=.01,
        help="Maximum allowed EER increase relative to anchor in any dev bank.",
    )
    args = parser.parse_args()

    truth = pd.read_csv(args.truth, dtype={"ID": str})
    if "SOURCE_BANK" not in truth:
        truth["SOURCE_BANK"] = f"{args.truth.parent.name}:{args.truth.stem}"
    anchor = pd.read_csv(args.anchor, dtype={"ID": str})
    head = pd.read_csv(args.head, dtype={"ID": str})
    head = head[head.ID.isin(truth.ID)].copy()
    if head.ID.duplicated().any():
        raise ValueError("Dual-domain predictions contain duplicate union IDs")
    head = head[["ID", *[value[1] for value in TASKS.values()]]].rename(
        columns={value[1]: f"HEAD_{value[1]}" for value in TASKS.values()}
    )
    frame = truth.merge(anchor, on="ID", validate="one_to_one").merge(
        head, on="ID", validate="one_to_one"
    )
    if len(frame) != len(truth):
        raise ValueError(f"Prediction coverage mismatch: {len(frame)} != {len(truth)}")
    banks = sorted(frame.SOURCE_BANK.unique())

    records = []
    selected = {}
    frozen = None
    if args.frozen_config is not None:
        frozen = json.loads(args.frozen_config.read_text("utf-8"))["selected"]
    for task, (_, probability, _) in TASKS.items():
        anchor_score = frame[probability].to_numpy()
        head_score = frame[f"HEAD_{probability}"].to_numpy()
        baseline = {
            bank: task_eer(
                frame[frame.SOURCE_BANK.eq(bank)], task,
                frame.loc[frame.SOURCE_BANK.eq(bank), probability].to_numpy(),
            )
            for bank in banks
        }
        candidates = []
        modes_and_weights = (
            [(frozen[task]["mode"], float(frozen[task]["weight"]))]
            if frozen is not None else
            [(mode, weight) for mode in ("probability", "logit")
             for weight in args.weights]
        )
        for mode, weight in modes_and_weights:
            score = fuse(anchor_score, head_score, weight, mode)
            eers = {
                bank: task_eer(
                    frame[frame.SOURCE_BANK.eq(bank)], task,
                    score[frame.SOURCE_BANK.eq(bank).to_numpy()],
                )
                for bank in banks
            }
            regression = max(eers[bank] - baseline[bank] for bank in banks)
            quality = [1 - value for value in eers.values()]
            selection = 0.5 * np.mean(quality) + 0.5 * min(quality)
            row = {
                "TASK": task, "MODE": mode, "WEIGHT": weight,
                "SELECTION": selection, "MAX_BANK_REGRESSION": regression,
                **{f"EER_{bank}": value for bank, value in eers.items()},
            }
            records.append(row)
            if frozen is not None or regression <= args.max_bank_regression + 1e-12:
                candidates.append((selection, -regression, mode, weight, row))
        if not candidates:
            raise RuntimeError(f"No safe fusion candidate for {task}")
        best = max(candidates)
        selected[task] = {"mode": best[2], "weight": best[3], "metrics": best[4]}

    output = frame[["ID"]].copy()
    for task, (_, probability, _) in TASKS.items():
        item = selected[task]
        output[probability] = fuse(
            frame[probability].to_numpy(), frame[f"HEAD_{probability}"].to_numpy(),
            item["weight"], item["mode"],
        )
    # Presence is not touched by this ADS experiment.
    output["VOICE_PRESENT_PROB"] = frame.VOICE_PRESENT_PROB
    output["MUSIC_PRESENT_PROB"] = frame.MUSIC_PRESENT_PROB

    overall = {}
    for bank in banks:
        part = frame[frame.SOURCE_BANK.eq(bank)].copy()
        prediction = output.set_index("ID").loc[part.ID]
        eers = {
            task: task_eer(part, task, prediction[probability].to_numpy())
            for task, (_, probability, _) in TASKS.items()
        }
        overall[bank] = {
            **{f"{task}_EER": value for task, value in eers.items()},
            "ADS": sum(TASKS[task][2] * (1 - value) for task, value in eers.items()),
        }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records).to_csv(args.output_dir / "sweep.csv", index=False)
    output.to_csv(args.output_dir / "predictions.csv", index=False)
    summary = {"selected": selected, "banks": overall}
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
