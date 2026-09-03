import csv
from pathlib import Path

import numpy as np

from src.component_consistency import (
    apply_mixed_component_consistency,
    save_layout_scores,
    stem_layout_score,
)


COLUMNS = [
    "ID", "FILE_FAKE_PROB", "VOICE_FAKE_PROB", "MUSIC_FAKE_PROB",
    "VOICE_PRESENT_PROB", "MUSIC_PRESENT_PROB",
]


def write_submission(path: Path) -> None:
    rows = [
        ["sequential", .1, .9, .2, .9, .9],
        ["concurrent", .1, .9, .2, .9, .9],
        ["telephone", .1, .9, .2, .9, .9],
        ["voice_only", .1, .9, .2, .9, .1],
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(COLUMNS)
        writer.writerows(rows)


def test_layout_score_is_larger_for_sequential_activity():
    original = np.ones(32_000, dtype=np.float32)
    concurrent = np.full(32_000, .5, dtype=np.float32)
    localized = np.concatenate((
        np.zeros(16_000, dtype=np.float32),
        np.ones(16_000, dtype=np.float32),
    ))
    concurrent_score = stem_layout_score(original, concurrent, concurrent)
    localized_score = stem_layout_score(original, localized, 1 - localized)
    assert localized_score > concurrent_score


def test_consistency_changes_only_routed_nontelephone_mixture(tmp_path):
    submission = tmp_path / "submission.csv"
    layout = tmp_path / "layout.npz"
    telephone = tmp_path / "telephone.npz"
    write_submission(submission)
    save_layout_scores(
        layout,
        ["sequential", "concurrent", "telephone", "voice_only"],
        [.6, .1, .6, .6],
    )
    np.savez(telephone, ids=np.asarray(["telephone"]))

    apply_mixed_component_consistency(
        submission, layout, telephone, layout_threshold=.35, file_weight=.7,
    )
    with submission.open(newline="", encoding="utf-8") as handle:
        rows = {row["ID"]: row for row in csv.DictReader(handle)}
    assert float(rows["sequential"]["FILE_FAKE_PROB"]) > .1
    assert float(rows["concurrent"]["FILE_FAKE_PROB"]) == .1
    assert float(rows["telephone"]["FILE_FAKE_PROB"]) == .1
    assert float(rows["voice_only"]["FILE_FAKE_PROB"]) == .1
    for row in rows.values():
        assert float(row["VOICE_FAKE_PROB"]) == .9
        assert float(row["MUSIC_FAKE_PROB"]) == .2


def test_layout_score_archive_rejects_duplicate_ids(tmp_path):
    with np.testing.assert_raises(ValueError):
        save_layout_scores(tmp_path / "bad.npz", ["same", "same"], [.1, .2])
