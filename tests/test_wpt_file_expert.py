import csv
from pathlib import Path

import numpy as np
import pytest

from src.wpt_file_expert import (
    apply_file_logit_mixture,
    apply_file_submission_blend,
    mix_file_logits,
)


COLUMNS = [
    "ID", "FILE_FAKE_PROB", "VOICE_FAKE_PROB", "MUSIC_FAKE_PROB",
    "VOICE_PRESENT_PROB", "MUSIC_PRESENT_PROB",
]


def _logit(value):
    value = np.clip(np.asarray(value, dtype=np.float64), 1e-5, 1 - 1e-5)
    return np.log(value) - np.log1p(-value)


def _write(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def test_v41_surrogate_coefficients_are_exact():
    anchor = np.asarray([.13, .87])
    unified = np.asarray([.31, .69])
    wpt = np.asarray([.77, .23])
    # 70% of v41, after substituting the v47 anchor for v41's pre-WPT
    # pipeline: .30*A + .70*(.25*A + .15*U + .60*W).
    actual = mix_file_logits(
        anchor, wpt, wpt_weight=.42,
        unified=unified, unified_weight=.105,
    )
    expected_logit = .475 * _logit(anchor) + .105 * _logit(unified) + (
        .42 * _logit(wpt)
    )
    np.testing.assert_allclose(_logit(actual), expected_logit, atol=1e-12)


def test_file_mixture_preserves_other_four_outputs_and_aligns_ids(tmp_path: Path):
    rows = [
        dict(zip(COLUMNS, ["b", .20, .31, .41, .51, .61])),
        dict(zip(COLUMNS, ["a", .80, .32, .42, .52, .62])),
    ]
    submission = tmp_path / "submission.csv"
    _write(submission, rows)
    ids = np.asarray(["a", "b"])
    wpt = np.asarray([[.1, .2, .9], [.3, .4, .1]])
    apply_file_logit_mixture(
        submission, ids, wpt, wpt_weight=.525,
    )
    with submission.open(newline="") as handle:
        updated = list(csv.DictReader(handle))
    assert float(updated[0]["FILE_FAKE_PROB"]) < .20
    assert float(updated[1]["FILE_FAKE_PROB"]) > .80
    for before, after in zip(rows, updated):
        for column in COLUMNS[2:]:
            assert float(after[column]) == before[column]


def test_file_mixture_rejects_missing_unified_values():
    with pytest.raises(ValueError, match="unified probabilities"):
        mix_file_logits(
            np.asarray([.2]), np.asarray([.8]),
            wpt_weight=.4, unified_weight=.1,
        )


def test_file_mixture_rejects_non_convex_weights():
    with pytest.raises(ValueError, match="sum <= 1"):
        mix_file_logits(
            np.asarray([.2]), np.asarray([.8]),
            wpt_weight=.8, unified=np.asarray([.3]), unified_weight=.3,
        )


def test_branch_blend_changes_only_file_and_aligns_reverse_order(tmp_path: Path):
    main = tmp_path / "main.csv"
    branch = tmp_path / "branch.csv"
    rows = [
        dict(zip(COLUMNS, ["a", .20, .31, .41, .51, .61])),
        dict(zip(COLUMNS, ["b", .80, .32, .42, .52, .62])),
    ]
    branch_rows = [
        dict(zip(COLUMNS, ["b", .10, .92, .82, .72, .62])),
        dict(zip(COLUMNS, ["a", .90, .91, .81, .71, .61])),
    ]
    _write(main, rows)
    _write(branch, branch_rows)
    apply_file_submission_blend(main, branch, branch_weight=.70)
    with main.open(newline="") as handle:
        updated = list(csv.DictReader(handle))
    assert float(updated[0]["FILE_FAKE_PROB"]) > .20
    assert float(updated[1]["FILE_FAKE_PROB"]) < .80
    for before, after in zip(rows, updated):
        for column in COLUMNS[2:]:
            assert float(after[column]) == before[column]
