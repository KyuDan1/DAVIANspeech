"""Original-audio WPT File expert with an explicit logit mixture.

This module deliberately owns only the File decision.  It is small enough to
be added after any existing submission pipeline, and it leaves both component
scores and both presence scores bit-exact.  The expensive WPT inference is
delegated to :mod:`wpt_spectra_inference`, so a package that already contains
the Spectra-AASIST checkpoint can reuse that exact asset.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

try:  # package import in tests; flat imports inside a submission archive
    from .pipeline import find_audio_files, order_by_submission
    from .wpt_spectra_inference import predict_wpt_tasks
except ImportError:  # pragma: no cover - exercised by script.py
    from pipeline import find_audio_files, order_by_submission
    from wpt_spectra_inference import predict_wpt_tasks


def _logit(values) -> np.ndarray:
    values = np.clip(np.asarray(values, dtype=np.float64), 1e-5, 1 - 1e-5)
    return np.log(values) - np.log1p(-values)


def _sigmoid(values) -> np.ndarray:
    return np.exp(-np.logaddexp(0, -np.asarray(values, dtype=np.float64)))


def mix_file_logits(
    anchor: np.ndarray,
    wpt: np.ndarray,
    *,
    wpt_weight: float,
    unified: np.ndarray | None = None,
    unified_weight: float = 0.0,
) -> np.ndarray:
    """Return a convex File-only mixture in logit space.

    If ``unified`` is omitted, ``unified_weight`` must be zero.  The anchor
    receives the residual weight ``1 - wpt_weight - unified_weight``.  Keeping
    the three coefficients explicit makes the v41-surrogate algebra auditable
    and avoids silently applying another pipeline's outer fusion twice.
    """
    weights = (float(wpt_weight), float(unified_weight))
    if any(not 0 <= value <= 1 for value in weights) or sum(weights) > 1:
        raise ValueError("WPT and unified weights must be non-negative and sum <= 1")
    anchor = np.asarray(anchor, dtype=np.float64)
    wpt = np.asarray(wpt, dtype=np.float64)
    if anchor.shape != wpt.shape:
        raise ValueError("anchor and WPT File probabilities must have equal shape")
    if unified is None:
        if unified_weight != 0:
            raise ValueError("unified probabilities are required for non-zero weight")
        unified_logit = 0.0
    else:
        unified = np.asarray(unified, dtype=np.float64)
        if unified.shape != anchor.shape:
            raise ValueError("unified and anchor File probabilities must have equal shape")
        unified_logit = unified_weight * _logit(unified)
    anchor_weight = 1.0 - wpt_weight - unified_weight
    return _sigmoid(
        anchor_weight * _logit(anchor)
        + wpt_weight * _logit(wpt)
        + unified_logit
    )


def apply_file_logit_mixture(
    submission_path: Path,
    expert_ids: np.ndarray,
    wpt_probabilities: np.ndarray,
    *,
    wpt_weight: float,
    unified_probabilities: np.ndarray | None = None,
    unified_weight: float = 0.0,
) -> None:
    """Update only ``FILE_FAKE_PROB`` in a submission CSV.

    Expert probability matrices use the repository's Voice/Music/File order.
    All IDs must match exactly; this prevents accidental cross-file alignment,
    which would also violate the competition's independent-sample rule.
    """
    expert_ids = np.asarray(expert_ids).astype(str)
    if len(set(expert_ids)) != len(expert_ids):
        raise ValueError("expert predictions contain duplicate IDs")
    wpt_probabilities = np.asarray(wpt_probabilities, dtype=np.float64)
    if wpt_probabilities.shape != (len(expert_ids), 3):
        raise ValueError("WPT probabilities must have shape [files, 3]")
    if unified_probabilities is not None:
        unified_probabilities = np.asarray(
            unified_probabilities, dtype=np.float64
        )
        if unified_probabilities.shape != (len(expert_ids), 3):
            raise ValueError("unified probabilities must have shape [files, 3]")

    submission_path = Path(submission_path)
    with submission_path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        columns = list(reader.fieldnames or [])
        rows = list(reader)
    row_ids = [row["ID"] for row in rows]
    if len(set(row_ids)) != len(row_ids):
        raise ValueError("submission contains duplicate IDs")
    if set(row_ids) != set(expert_ids):
        raise ValueError("submission and expert prediction IDs differ")

    position = {item: index for index, item in enumerate(expert_ids)}
    anchor = np.asarray(
        [float(row["FILE_FAKE_PROB"]) for row in rows], dtype=np.float64
    )
    order = np.asarray([position[item] for item in row_ids], dtype=np.int64)
    unified_file = (
        None if unified_probabilities is None
        else unified_probabilities[order, 2]
    )
    mixed = mix_file_logits(
        anchor,
        wpt_probabilities[order, 2],
        wpt_weight=wpt_weight,
        unified=unified_file,
        unified_weight=unified_weight,
    )
    for row, value in zip(rows, mixed):
        row["FILE_FAKE_PROB"] = round(float(value), 10)

    temporary = submission_path.with_suffix(".tmp.csv")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(submission_path)


def apply_file_submission_blend(
    submission_path: Path,
    branch_submission_path: Path,
    *,
    branch_weight: float,
) -> None:
    """Blend only File from a fully evaluated branch submission.

    This is the exact operation used when the complementary evidence is the
    final v41 File rank rather than raw WPT.  The branch is aligned by ID and
    every non-File value in ``submission_path`` remains untouched.
    """
    if not 0 <= branch_weight <= 1:
        raise ValueError("branch weight must lie in [0, 1]")
    submission_path = Path(submission_path)
    branch_submission_path = Path(branch_submission_path)

    def read(path: Path) -> tuple[list[str], list[dict[str, str]]]:
        with path.open(encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            return list(reader.fieldnames or []), list(reader)

    columns, rows = read(submission_path)
    _, branch_rows = read(branch_submission_path)
    row_ids = [row["ID"] for row in rows]
    branch_ids = [row["ID"] for row in branch_rows]
    if len(set(row_ids)) != len(row_ids) or len(set(branch_ids)) != len(branch_ids):
        raise ValueError("submission branches contain duplicate IDs")
    if set(row_ids) != set(branch_ids):
        raise ValueError("submission branch IDs differ")
    branch_by_id = {row["ID"]: row for row in branch_rows}
    anchor = np.asarray(
        [float(row["FILE_FAKE_PROB"]) for row in rows], dtype=np.float64
    )
    expert = np.asarray([
        float(branch_by_id[item]["FILE_FAKE_PROB"]) for item in row_ids
    ], dtype=np.float64)
    mixed = _sigmoid(
        (1 - branch_weight) * _logit(anchor)
        + branch_weight * _logit(expert)
    )
    for row, value in zip(rows, mixed):
        row["FILE_FAKE_PROB"] = round(float(value), 10)
    temporary = submission_path.with_suffix(".tmp.csv")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(submission_path)


def apply_wpt_file_expert(
    test_dir: Path,
    submission_path: Path,
    model_dir: Path,
    checkpoint_path: Path,
    *,
    device: str = "cuda",
    file_batch_size: int = 6,
    file_views: int = 5,
    file_temperature: float = 2.0,
    wpt_weight: float,
    unified_expert_path: Path | None = None,
    unified_weight: float = 0.0,
) -> None:
    """Score original audio once and apply a File-only WPT mixture."""
    submission_path = Path(submission_path)
    with submission_path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    audio_paths = order_by_submission(find_audio_files(Path(test_dir)), rows)
    probabilities = predict_wpt_tasks(
        audio_paths,
        model_dir,
        checkpoint_path,
        device=device,
        file_batch_size=file_batch_size,
        file_views=file_views,
        file_temperature=file_temperature,
    )
    unified = None
    if unified_expert_path is not None:
        with np.load(unified_expert_path, allow_pickle=False) as archive:
            unified_ids = archive["ids"].astype(str)
            unified_values = archive["probabilities"].astype(np.float64)
        ids = np.asarray([row["ID"] for row in rows])
        if len(set(unified_ids)) != len(unified_ids) or set(unified_ids) != set(ids):
            raise ValueError("unified and submission IDs differ")
        position = {item: index for index, item in enumerate(unified_ids)}
        unified = unified_values[[position[item] for item in ids]]
    apply_file_logit_mixture(
        submission_path,
        np.asarray([row["ID"] for row in rows]),
        probabilities,
        wpt_weight=wpt_weight,
        unified_probabilities=unified,
        unified_weight=unified_weight,
    )
