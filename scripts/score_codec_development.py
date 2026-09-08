#!/usr/bin/env python3
"""Score one candidate on a paired codec development bank.

Unlike the one-shot prospective scorer, this command is explicitly intended
for repeated checkpoint selection on a manifest registered as development.
It never fits a threshold or searches a fusion weight.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import tempfile

import pandas as pd

try:
    from .score_prospective_paired_eval import (
        OUTPUT_FILES,
        score_prospective_paired_eval,
    )
except ImportError:  # Direct script execution.
    from score_prospective_paired_eval import (  # type: ignore[no-redef]
        OUTPUT_FILES,
        score_prospective_paired_eval,
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_development_report(
    truth_path: Path,
    prediction_path: Path,
    output_dir: Path,
    dataset_name: str | None = None,
) -> dict[str, pd.DataFrame]:
    if output_dir.exists() or output_dir.is_symlink():
        raise FileExistsError(f"refusing to overwrite development report: {output_dir}")
    truth = pd.read_csv(truth_path, dtype={"ID": str})
    prediction = pd.read_csv(prediction_path, dtype={"ID": str})
    if dataset_name is not None:
        dataset_name = dataset_name.strip()
        if not dataset_name:
            raise ValueError("dataset_name cannot be blank")
        if "DATASET" not in prediction:
            raise ValueError("prediction has no DATASET column for filtering")
        prediction = prediction.loc[
            prediction["DATASET"].astype(str).eq(dataset_name)
        ].copy()
        if prediction.empty:
            raise ValueError(f"prediction has no rows for dataset {dataset_name!r}")
    reports = score_prospective_paired_eval(truth, prediction)
    provenance = {
        "purpose": "repeatable_development_checkpoint_evaluation",
        "threshold_or_weight_search": False,
        "truth": {
            "path": str(truth_path.resolve()),
            "sha256": _sha256(truth_path),
            "rows": len(truth),
        },
        "prediction": {
            "path": str(prediction_path.resolve()),
            "sha256": _sha256(prediction_path),
            "rows": len(prediction),
            "dataset_filter": dataset_name,
        },
    }
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(
        prefix=f".{output_dir.name}.staging-", dir=output_dir.parent,
    ))
    try:
        for key, filename in OUTPUT_FILES.items():
            reports[key].to_csv(staging / filename, index=False)
        (staging / "provenance.json").write_text(
            json.dumps(provenance, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        staging.replace(output_dir)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return reports


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--truth", type=Path, required=True)
    parser.add_argument("--prediction", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--dataset",
        help="Optional DATASET value when scoring one partition from trainer output",
    )
    args = parser.parse_args()
    reports = write_development_report(
        args.truth, args.prediction, args.output_dir, args.dataset,
    )
    print(reports["overall"].to_string(index=False))


if __name__ == "__main__":
    main()
