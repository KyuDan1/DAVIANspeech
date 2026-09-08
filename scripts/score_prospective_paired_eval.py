#!/usr/bin/env python3
"""Score frozen v47/v48 outputs on one prospective paired evaluation bank.

This command has no model-selection, threshold-fitting, or sweep interface.
Input paths are always explicit; importing the module does not read any bank.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import tempfile

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve


TASKS = ("FILE", "VOICE", "MUSIC")
PROBABILITY_COLUMNS = tuple(f"{task}_FAKE_PROB" for task in TASKS) + (
    "VOICE_PRESENT_PROB", "MUSIC_PRESENT_PROB",
)
LABEL_COLUMNS = tuple(f"{task}_FAKE" for task in TASKS) + (
    "VOICE_PRESENT", "MUSIC_PRESENT",
)
CHANNELS = (
    "clean", "g711_ulaw", "g722_wb", "opus_nb_8k",
    "transcode_g711_opus",
)
TELEPHONE_CHANNELS = CHANNELS[1:]
LAYOUTS = ("concurrent", "partial_overlap", "sequential")
COMPONENT_CASES = ("RR", "RF", "FR", "FF")
OUTPUT_FILES = {
    "overall": "overall.csv",
    "channels": "channels.csv",
    "cell_contrasts": "cell_contrasts.csv",
    "paired_deltas": "paired_deltas.csv",
    "paired_rank_flips": "paired_rank_flips.csv",
}
PROVENANCE_FILE = "provenance.json"
SCHEMA_VERSION = "prospective_paired_eval_v1"


def official_eer(labels, scores) -> float:
    """Competition EER using the complete ROC threshold sequence."""
    labels = np.asarray(labels, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    if np.unique(labels).size != 2:
        return float("nan")
    fpr, tpr, _ = roc_curve(
        labels, scores, pos_label=1, drop_intermediate=False
    )
    fnr = 1 - tpr
    index = int(np.argmin(np.abs(fpr - fnr)))
    return float((fpr[index] + fnr[index]) / 2)


def _auc(labels, scores) -> float:
    labels = np.asarray(labels, dtype=np.int64)
    if np.unique(labels).size != 2:
        return float("nan")
    return float(roc_auc_score(labels, np.asarray(scores, dtype=np.float64)))


def _normalized_strings(frame: pd.DataFrame, column: str, table: str) -> pd.Series:
    if column not in frame:
        raise ValueError(f"{table} is missing required column {column}")
    values = frame[column].astype("string").str.strip()
    if values.isna().any() or values.eq("").any():
        raise ValueError(f"{table}.{column} contains blank values")
    return values.astype(str)


def _validate_ids(frame: pd.DataFrame, table: str) -> pd.DataFrame:
    result = frame.copy()
    result["ID"] = _normalized_strings(result, "ID", table)
    if result.ID.duplicated().any():
        duplicate = result.loc[result.ID.duplicated(), "ID"].iloc[0]
        raise ValueError(f"{table}.ID contains duplicate {duplicate!r}")
    return result


def _binary_column(frame: pd.DataFrame, column: str, table: str) -> pd.Series:
    if column not in frame:
        raise ValueError(f"{table} is missing required column {column}")
    try:
        values = pd.to_numeric(frame[column], errors="raise")
    except (TypeError, ValueError) as error:
        raise ValueError(f"{table}.{column} must contain binary labels") from error
    if values.isna().any() or not values.isin((0, 1)).all():
        raise ValueError(f"{table}.{column} must contain only 0/1 labels")
    return values.astype(np.int64)


def _probability_column(
    frame: pd.DataFrame, column: str, table: str,
) -> pd.Series:
    if column not in frame:
        raise ValueError(f"{table} is missing required column {column}")
    try:
        values = pd.to_numeric(frame[column], errors="raise").astype(np.float64)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{table}.{column} must contain probabilities") from error
    if not np.isfinite(values).all() or not values.between(0, 1).all():
        raise ValueError(f"{table}.{column} must be finite and within [0, 1]")
    return values


def _parent_column(truth: pd.DataFrame) -> str:
    available = [column for column in ("BASE_ID", "PARENT_ID") if column in truth]
    if not available:
        raise ValueError("truth must contain BASE_ID or PARENT_ID")
    if len(available) == 2:
        base = _normalized_strings(truth, "BASE_ID", "truth")
        parent = _normalized_strings(truth, "PARENT_ID", "truth")
        if not base.equals(parent):
            raise ValueError("truth BASE_ID and PARENT_ID disagree")
    return available[0]


def validate_and_align(
    truth: pd.DataFrame, prediction: pd.DataFrame,
) -> tuple[pd.DataFrame, str]:
    """Validate inputs and return one row per truth ID in truth order."""
    truth = _validate_ids(truth, "truth")
    prediction = _validate_ids(prediction, "prediction")
    missing = sorted(set(truth.ID) - set(prediction.ID))
    extra = sorted(set(prediction.ID) - set(truth.ID))
    if missing or extra:
        raise ValueError(
            "truth/prediction ID mismatch: "
            f"missing={len(missing)} {missing[:5]}, extra={len(extra)} {extra[:5]}"
        )

    for column in LABEL_COLUMNS:
        truth[column] = _binary_column(truth, column, "truth")
    for column in PROBABILITY_COLUMNS:
        prediction[column] = _probability_column(
            prediction, column, "prediction"
        )
    expected_file = truth.VOICE_FAKE | truth.MUSIC_FAKE
    if not truth.FILE_FAKE.equals(expected_file):
        raise ValueError("truth FILE_FAKE must equal VOICE_FAKE OR MUSIC_FAKE")

    parent_column = _parent_column(truth)
    truth[parent_column] = _normalized_strings(truth, parent_column, "truth")
    truth["CHANNEL"] = _normalized_strings(truth, "CHANNEL", "truth")
    truth["MIX_MODE"] = _normalized_strings(truth, "MIX_MODE", "truth")
    unknown_channels = sorted(set(truth.CHANNEL) - set(CHANNELS))
    if unknown_channels or set(truth.CHANNEL) != set(CHANNELS):
        raise ValueError(
            f"truth CHANNEL must contain exactly {list(CHANNELS)}; "
            f"unknown={unknown_channels}"
        )
    unknown_layouts = sorted(set(truth.MIX_MODE) - set(LAYOUTS))
    if unknown_layouts:
        raise ValueError(f"truth has unsupported MIX_MODE values: {unknown_layouts}")

    derived_case = np.where(
        truth.VOICE_FAKE.eq(1), "F", "R"
    ) + np.where(truth.MUSIC_FAKE.eq(1), "F", "R")
    if "COMPONENT_CASE" in truth:
        supplied = _normalized_strings(truth, "COMPONENT_CASE", "truth").str.upper()
        if not supplied.equals(pd.Series(derived_case, index=truth.index)):
            raise ValueError("truth COMPONENT_CASE disagrees with fake labels")
    truth["COMPONENT_CASE"] = derived_case

    if not truth.VOICE_PRESENT.eq(1).all() or not truth.MUSIC_PRESENT.eq(1).all():
        raise ValueError("the 12-cell paired bank requires both components present")
    expected_cells = {(layout, case) for layout in LAYOUTS for case in COMPONENT_CASES}
    cells = set(zip(truth.MIX_MODE, truth.COMPONENT_CASE))
    if cells != expected_cells:
        missing_cells = sorted(expected_cells - cells)
        extra_cells = sorted(cells - expected_cells)
        raise ValueError(
            f"truth must contain all 12 layout/component cells; "
            f"missing={missing_cells}, extra={extra_cells}"
        )

    if truth.duplicated([parent_column, "CHANNEL"]).any():
        raise ValueError("truth contains duplicate parent/channel rows")
    invariant = [*LABEL_COLUMNS, "MIX_MODE", "COMPONENT_CASE"]
    for parent, group in truth.groupby(parent_column, sort=False):
        channels = set(group.CHANNEL)
        if len(group) != len(CHANNELS) or channels != set(CHANNELS):
            raise ValueError(
                f"parent {parent!r} must have exactly one row for every channel"
            )
        if any(group[column].nunique(dropna=False) != 1 for column in invariant):
            raise ValueError(f"parent {parent!r} changes labels or cell across channels")
    parent_cells = truth.drop_duplicates(parent_column)
    cell_counts = parent_cells.groupby(
        ["MIX_MODE", "COMPONENT_CASE"]
    ).size()
    if cell_counts.nunique() != 1:
        raise ValueError("the 12 layout/component cells must have equal parent counts")

    aligned = prediction.set_index("ID").loc[truth.ID, PROBABILITY_COLUMNS]
    aligned = aligned.reset_index(drop=True)
    frame = truth.reset_index(drop=True).copy()
    for column in PROBABILITY_COLUMNS:
        frame[column] = aligned[column].to_numpy()
    frame["PARENT_KEY"] = frame[parent_column]
    return frame, parent_column


def score_metrics(frame: pd.DataFrame) -> dict[str, float | int | bool]:
    values: dict[str, float | int | bool] = {"N": len(frame)}
    task_eers = {}
    for task in TASKS:
        selected = frame
        if task != "FILE":
            selected = selected.loc[selected[f"{task}_PRESENT"].eq(1)]
        eer = official_eer(selected[f"{task}_FAKE"], selected[f"{task}_FAKE_PROB"])
        task_eers[task] = eer
        values[f"{task}_N"] = len(selected)
        values[f"{task}_EER"] = eer
    voice_auc = _auc(frame.VOICE_PRESENT, frame.VOICE_PRESENT_PROB)
    music_auc = _auc(frame.MUSIC_PRESENT, frame.MUSIC_PRESENT_PROB)
    ads = (
        .5 * (1 - task_eers["FILE"])
        + .2 * (1 - task_eers["VOICE"])
        + .3 * (1 - task_eers["MUSIC"])
    )
    cps_defined = bool(np.isfinite(voice_auc) and np.isfinite(music_auc))
    cps = .5 * voice_auc + .5 * music_auc if cps_defined else float("nan")
    total = .9 * ads + .1 * cps if cps_defined else float("nan")
    values.update({
        "VOICE_PRESENCE_AUC": voice_auc,
        "MUSIC_PRESENCE_AUC": music_auc,
        "ADS": float(ads), "CPS": float(cps), "CPS_DEFINED": cps_defined,
        "TOTAL": float(total),
    })
    return values


def channel_table(frame: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame([
        {"CHANNEL": channel, **score_metrics(frame.loc[frame.CHANNEL.eq(channel)])}
        for channel in CHANNELS
    ])


def _contrast_eer(
    frame: pd.DataFrame, layout: str, positive_case: str,
    negative_case: str, task: str,
) -> tuple[float, int, int]:
    block = frame.loc[
        frame.MIX_MODE.eq(layout)
        & frame.COMPONENT_CASE.isin((positive_case, negative_case))
    ]
    positive = block.loc[block.COMPONENT_CASE.eq(positive_case)]
    negative = block.loc[block.COMPONENT_CASE.eq(negative_case)]
    labels = np.concatenate((
        np.ones(len(positive), dtype=np.int64),
        np.zeros(len(negative), dtype=np.int64),
    ))
    scores = np.concatenate((
        positive[f"{task}_FAKE_PROB"].to_numpy(np.float64),
        negative[f"{task}_FAKE_PROB"].to_numpy(np.float64),
    ))
    return official_eer(labels, scores), len(negative), len(positive)


def cell_contrast_table(frame: pd.DataFrame) -> pd.DataFrame:
    """Return one row per layout/case with label-matched reference contrasts."""
    rows = []
    for layout in LAYOUTS:
        for case in COMPONENT_CASES:
            parents = frame.loc[
                frame.MIX_MODE.eq(layout) & frame.COMPONENT_CASE.eq(case),
                "PARENT_KEY",
            ].nunique()
            row: dict[str, object] = {
                "MIX_MODE": layout, "COMPONENT_CASE": case,
                "PARENTS": parents, "N": parents * len(CHANNELS),
            }
            references = {
                "FILE": "RR" if case != "RR" else None,
                "VOICE": ("R" + case[1]) if case[0] == "F" else None,
                "MUSIC": (case[0] + "R") if case[1] == "F" else None,
            }
            for task, reference in references.items():
                row[f"{task}_REFERENCE_CASE"] = reference
                if reference is None:
                    row[f"{task}_EER"] = np.nan
                    row[f"{task}_NEG_N"] = 0
                    row[f"{task}_POS_N"] = 0
                else:
                    eer, negative_n, positive_n = _contrast_eer(
                        frame, layout, case, reference, task
                    )
                    row[f"{task}_EER"] = eer
                    row[f"{task}_NEG_N"] = negative_n
                    row[f"{task}_POS_N"] = positive_n
            rows.append(row)
    return pd.DataFrame(rows)


def paired_tables(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    indexed = frame.set_index(["PARENT_KEY", "CHANNEL"])
    if not indexed.index.is_unique:
        raise ValueError("validated parent/channel index unexpectedly is not unique")
    delta_rows = []
    rank_rows = []
    parents = frame.drop_duplicates("PARENT_KEY").set_index("PARENT_KEY")
    for channel in TELEPHONE_CHANNELS:
        for task in TASKS:
            label_column = f"{task}_FAKE"
            probability_column = f"{task}_FAKE_PROB"
            eligible = parents.index
            if task != "FILE":
                eligible = parents.index[parents[f"{task}_PRESENT"].eq(1)]
            labels = parents.loc[eligible, label_column].astype(np.int64)
            clean = indexed.loc[(eligible, "clean"), probability_column].to_numpy(
                np.float64
            )
            telephone = indexed.loc[(eligible, channel), probability_column].to_numpy(
                np.float64
            )
            delta = telephone - clean
            signed_delta = delta * (2 * labels.to_numpy() - 1)
            for parent, label, clean_score, channel_score, change, signed in zip(
                eligible, labels, clean, telephone, delta, signed_delta
            ):
                delta_rows.append({
                    "PARENT_ID": parent, "CHANNEL": channel, "TASK": task,
                    "LABEL": int(label), "CLEAN_PROB": clean_score,
                    "CHANNEL_PROB": channel_score, "SCORE_DELTA": change,
                    "SIGNED_DELTA": signed,
                })

            positive = labels.to_numpy() == 1
            negative = ~positive
            clean_margin = clean[positive, None] - clean[negative][None, :]
            channel_margin = (
                telephone[positive, None] - telephone[negative][None, :]
            )
            clean_correct = clean_margin > 0
            channel_correct = channel_margin > 0
            clean_state = np.sign(clean_margin)
            channel_state = np.sign(channel_margin)
            pair_count = clean_correct.size
            rank_rows.append({
                "CHANNEL": channel, "TASK": task, "PARENTS": len(eligible),
                "POSITIVE_PARENTS": int(positive.sum()),
                "NEGATIVE_PARENTS": int(negative.sum()),
                "PAIRS": pair_count,
                "CLEAN_EER": official_eer(labels, clean),
                "CHANNEL_EER": official_eer(labels, telephone),
                "CLEAN_PAIR_ERROR": (
                    float(1 - clean_correct.mean()) if pair_count else np.nan
                ),
                "CHANNEL_PAIR_ERROR": (
                    float(1 - channel_correct.mean()) if pair_count else np.nan
                ),
                "CORRECT_TO_INCORRECT": int(
                    (clean_correct & ~channel_correct).sum()
                ),
                "INCORRECT_TO_CORRECT": int(
                    (~clean_correct & channel_correct).sum()
                ),
                "CLEAN_TIES": int((clean_state == 0).sum()),
                "CHANNEL_TIES": int((channel_state == 0).sum()),
                "RANK_STATE_FLIPS": int((clean_state != channel_state).sum()),
                "MEAN_SCORE_DELTA": float(delta.mean()),
                "MEAN_SIGNED_DELTA": float(signed_delta.mean()),
                "MEAN_PAIR_MARGIN_DELTA": (
                    float((channel_margin - clean_margin).mean())
                    if pair_count else np.nan
                ),
            })
    return pd.DataFrame(delta_rows), pd.DataFrame(rank_rows)


def score_prospective_paired_eval(
    truth: pd.DataFrame, prediction: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    frame, _ = validate_and_align(truth, prediction)
    deltas, ranks = paired_tables(frame)
    return {
        "overall": pd.DataFrame([score_metrics(frame)]),
        "channels": channel_table(frame),
        "cell_contrasts": cell_contrast_table(frame),
        "paired_deltas": deltas,
        "paired_rank_flips": ranks,
    }


def score_frozen_v47_v48(
    truth: pd.DataFrame, v47_prediction: pd.DataFrame,
    v48_prediction: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    """Score exactly the two predeclared frozen candidates in one operation."""
    by_model = {
        "v47": score_prospective_paired_eval(truth, v47_prediction),
        "v48": score_prospective_paired_eval(truth, v48_prediction),
    }
    combined = {}
    for report_key in OUTPUT_FILES:
        blocks = []
        for model, reports in by_model.items():
            block = reports[report_key].copy()
            block.insert(0, "MODEL", model)
            blocks.append(block)
        combined[report_key] = pd.concat(blocks, ignore_index=True)
    return combined


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_provenance(
    truth_path: Path,
    v47_prediction_path: Path,
    v48_prediction_path: Path,
    *,
    truth_rows: int,
    v47_rows: int,
    v48_rows: int,
) -> dict[str, object]:
    """Describe the immutable inputs and fixed two-model scoring contract."""
    inputs = {}
    for name, path, rows in (
        ("truth", truth_path, truth_rows),
        ("v47_prediction", v47_prediction_path, v47_rows),
        ("v48_prediction", v48_prediction_path, v48_rows),
    ):
        path = Path(path)
        inputs[name] = {
            "path": str(path.resolve()),
            "sha256": _sha256_file(path),
            "bytes": path.stat().st_size,
            "rows": int(rows),
        }
    scorer_path = Path(__file__)
    scorer_hash = _sha256_file(scorer_path)
    fingerprint_payload = {
        "schema_version": SCHEMA_VERSION,
        "frozen_models": ["v47", "v48"],
        "scorer_sha256": scorer_hash,
        "input_sha256": {
            name: metadata["sha256"] for name, metadata in inputs.items()
        },
    }
    evaluation_fingerprint = hashlib.sha256(
        json.dumps(fingerprint_payload, sort_keys=True).encode("utf-8")
    ).hexdigest()
    return {
        "schema_version": SCHEMA_VERSION,
        "evaluation_fingerprint": evaluation_fingerprint,
        "single_evaluation": True,
        "frozen_models": ["v47", "v48"],
        "selection_or_sweep_supported": False,
        "eer_drop_intermediate": False,
        "scorer": {
            "path": str(scorer_path.resolve()),
            "sha256": scorer_hash,
        },
        "inputs": inputs,
    }


def _refuse_existing_output(output_dir: Path) -> None:
    if Path(output_dir).exists():
        raise FileExistsError(
            "refusing to overwrite prospective score output directory: "
            f"{output_dir}"
        )


def write_reports(
    reports: dict[str, pd.DataFrame],
    output_dir: Path,
    provenance: dict[str, object],
) -> None:
    """Publish one complete result directory, refusing every overwrite."""
    output_dir = Path(output_dir)
    if set(reports) != set(OUTPUT_FILES):
        raise ValueError(
            f"report keys must be exactly {sorted(OUTPUT_FILES)}"
        )
    _refuse_existing_output(output_dir)
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(
        prefix=f".{output_dir.name}.staging-", dir=output_dir.parent
    ))
    try:
        for key, name in OUTPUT_FILES.items():
            reports[key].to_csv(staging / name, index=False)
        (staging / PROVENANCE_FILE).write_text(
            json.dumps(provenance, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        staging.replace(output_dir)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--truth", type=Path, required=True)
    parser.add_argument("--v47-prediction", type=Path, required=True)
    parser.add_argument("--v48-prediction", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    # Refuse before opening truth or prediction files, so a repeated command
    # cannot even recompute an already-published prospective evaluation.
    _refuse_existing_output(args.output_dir)
    truth = pd.read_csv(args.truth, dtype={"ID": str})
    v47_prediction = pd.read_csv(args.v47_prediction, dtype={"ID": str})
    v48_prediction = pd.read_csv(args.v48_prediction, dtype={"ID": str})
    reports = score_frozen_v47_v48(truth, v47_prediction, v48_prediction)
    provenance = build_provenance(
        args.truth,
        args.v47_prediction,
        args.v48_prediction,
        truth_rows=len(truth),
        v47_rows=len(v47_prediction),
        v48_rows=len(v48_prediction),
    )
    write_reports(reports, args.output_dir, provenance)
    print(reports["overall"].to_string(index=False))


if __name__ == "__main__":
    main()
