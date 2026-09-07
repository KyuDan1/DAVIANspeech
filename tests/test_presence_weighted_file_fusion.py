import csv

import numpy as np
import pytest

from src.presence_weighted_file_fusion import (
    apply_presence_weighted_file_fusion,
    component_file_evidence,
)


COLUMNS = [
    "ID", "FILE_FAKE_PROB", "VOICE_FAKE_PROB", "MUSIC_FAKE_PROB",
    "VOICE_PRESENT_PROB", "MUSIC_PRESENT_PROB",
]


def test_presence_adjusts_component_before_soft_or():
    score = component_file_evidence(.9, .9, .9, .1)
    voice_only = component_file_evidence(.9, .1, .9, .9)

    assert np.isfinite(score)
    assert score > .9
    assert voice_only > .9


def test_file_fusion_changes_only_file_column(tmp_path):
    path = tmp_path / "submission.csv"
    original = ["one", .1, .9, .2, .9, .9]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(COLUMNS)
        writer.writerow(original)

    apply_presence_weighted_file_fusion(path, file_weight=.5)

    with path.open(newline="", encoding="utf-8") as handle:
        row = next(csv.DictReader(handle))
    assert float(row["FILE_FAKE_PROB"]) > original[1]
    for column, value in zip(COLUMNS[2:], original[2:]):
        assert float(row[column]) == value


def test_file_fusion_rejects_invalid_weight(tmp_path):
    with pytest.raises(ValueError, match="file_weight"):
        apply_presence_weighted_file_fusion(tmp_path / "missing.csv", 1.1)
