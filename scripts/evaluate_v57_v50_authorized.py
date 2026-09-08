#!/usr/bin/env python3
"""Rebase frozen v57 residuals on exact v50 for authorised codec development."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from evaluate_diagnostic import PREDICTION_COLUMNS  # noqa: E402
from evaluate_three_stream_v57 import (  # noqa: E402
    add_diagnostic_axes,
    cell_operating_rows,
    evaluate_model,
    full_residual_predictions,
    join_predictions,
    load_candidate_predictions,
    parse_candidate,
    regression_table,
    task_isolated_predictions,
)


DATASET = "codec_mixed_dev_v4"


def load_exact(path: Path, *, dataset: str = DATASET) -> pd.DataFrame:
    frame = pd.read_csv(path, dtype={"ID": str})
    required = {"ID", *PREDICTION_COLUMNS}
    if missing := required.difference(frame):
        raise ValueError(f"exact v50 prediction misses columns: {sorted(missing)}")
    if "DATASET" not in frame:
        frame.insert(0, "DATASET", dataset)
    if frame[["DATASET", "ID"]].duplicated().any():
        raise ValueError("exact v50 prediction has duplicate keys")
    return frame[["DATASET", "ID", *PREDICTION_COLUMNS]]


def output_change_audit(
    base: pd.DataFrame, current: pd.DataFrame, expected: set[str],
) -> dict[str, object]:
    base = base.set_index(["DATASET", "ID"]).sort_index()
    current = current.set_index(["DATASET", "ID"]).sort_index()
    if not base.index.equals(current.index):
        raise ValueError("output audit key mismatch")
    changed = {
        column: int((base[column] != current[column]).sum())
        for column in PREDICTION_COLUMNS
    }
    unexpected = {
        column: count for column, count in changed.items()
        if column not in expected and count
    }
    if unexpected:
        raise RuntimeError(f"unexpected v50 output changes: {unexpected}")
    return {
        "expected_changed_columns": sorted(expected),
        "changed_rows_by_column": changed,
        "other_columns_bit_exact": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--truth", type=Path, required=True)
    parser.add_argument("--v50-prediction", type=Path, required=True)
    parser.add_argument("--candidate", action="append", type=parse_candidate, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--minimum-group-size", type=int, default=20)
    args = parser.parse_args()
    if args.output_dir.exists():
        parser.error(f"refusing to overwrite {args.output_dir}")
    truth = pd.read_csv(args.truth, dtype={"ID": str})
    truth.insert(0, "DATASET", DATASET)
    truth = add_diagnostic_axes(truth)
    v50 = load_exact(args.v50_prediction)
    if len(v50) != 600:
        raise ValueError("authorised exact-v50 codec audit must contain 600 rows")
    predictions = {"v50": v50}
    audits: dict[str, object] = {}
    for variant, path in args.candidate:
        candidate = load_candidate_predictions(path)
        candidate = candidate.loc[candidate.DATASET.eq(DATASET)].copy()
        models = {
            f"{variant}__full": full_residual_predictions(v50, candidate),
            f"{variant}__voice_only": task_isolated_predictions(v50, candidate, "voice"),
            f"{variant}__music_only": task_isolated_predictions(v50, candidate, "music"),
            f"{variant}__file_only": task_isolated_predictions(v50, candidate, "file"),
        }
        expected = {
            "full": {"FILE_FAKE_PROB", "VOICE_FAKE_PROB", "MUSIC_FAKE_PROB"},
            "voice_only": {"VOICE_FAKE_PROB"},
            "music_only": {"MUSIC_FAKE_PROB"},
            "file_only": {"FILE_FAKE_PROB"},
        }
        for name, prediction in models.items():
            predictions[name] = prediction
            suffix = name.split("__", 1)[1]
            audits[name] = output_change_audit(v50, prediction, expected[suffix])

    overall_rows, domain_rows, subgroup_rows, cell_rows = [], [], [], []
    for name, prediction in predictions.items():
        pooled, domains, subgroups = evaluate_model(
            name, join_predictions(truth, prediction), args.minimum_group_size,
        )
        overall_rows.append(pooled)
        domain_rows.extend(domains)
        subgroup_rows.extend(subgroups)
        cell_rows.extend(cell_operating_rows(name, join_predictions(truth, prediction)))
    overall = pd.DataFrame(overall_rows)
    domains = pd.DataFrame(domain_rows)
    subgroups = pd.DataFrame(subgroup_rows)
    cell_operating = pd.DataFrame(cell_rows)
    baseline_cells = cell_operating.loc[cell_operating.MODEL.eq("v50")].set_index("CELL")
    for task in ("FILE", "VOICE", "MUSIC", "WEIGHTED_CELL"):
        column = f"{task}_ERROR"
        cell_operating[f"DELTA_{column}"] = cell_operating.apply(
            lambda row: float(row[column] - baseline_cells.loc[row.CELL, column]),
            axis=1,
        )
    # regression_table expects its baseline to be called identity.
    regressions = regression_table(
        domains.assign(MODEL=domains.MODEL.replace("v50", "identity")),
        subgroups.assign(MODEL=subgroups.MODEL.replace("v50", "identity")),
    )
    args.output_dir.mkdir(parents=True)
    overall.to_csv(args.output_dir / "overall.csv", index=False)
    subgroups.to_csv(args.output_dir / "subgroups.csv", index=False)
    regressions.to_csv(args.output_dir / "regressions.csv", index=False)
    cell_operating.to_csv(args.output_dir / "cell_operating_points.csv", index=False)
    for name, prediction in predictions.items():
        if name != "v50":
            prediction.to_csv(args.output_dir / f"{name}_predictions.csv", index=False)
    summary = {
        "scope": "authorised codec_mixed_dev_v4 only",
        "rows": len(truth),
        "anchor": "exact v50 predictions",
        "audits": audits,
        "forbidden_roles_used": [],
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(overall.to_string(index=False))


if __name__ == "__main__":
    main()
