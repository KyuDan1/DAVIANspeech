import csv

import numpy as np
import pytest

from src.component_consistent_file_fusion import (
    apply_component_consistent_file_fusion,
    component_consistent_file_probability,
)


COLUMNS = [
    "ID", "FILE_FAKE_PROB", "VOICE_FAKE_PROB", "MUSIC_FAKE_PROB",
    "VOICE_PRESENT_PROB", "MUSIC_PRESENT_PROB",
]


def test_component_residual_is_gated_and_monotone():
    result = component_consistent_file_probability(
        [.2, .2, .2], [.9, .9, .1], [.1, .1, .9],
        [.9, .05, .9], [.9, .9, .1], file_weight=.2,
    )
    assert result[0] > .2
    assert result[1] == pytest.approx(.2)
    assert result[2] == pytest.approx(.2)


def test_file_fusion_reads_final_components_and_preserves_other_columns(tmp_path):
    path = tmp_path / "submission.csv"
    original = ["mixed", .2, .85, .3, .8, .9]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(COLUMNS)
        writer.writerow(original)

    apply_component_consistent_file_fusion(path)

    with path.open(newline="", encoding="utf-8") as handle:
        row = next(csv.DictReader(handle))
    assert float(row["FILE_FAKE_PROB"]) > original[1]
    for column, value in zip(COLUMNS[2:], original[2:]):
        assert float(row[column]) == value


def test_component_residual_rejects_invalid_inputs():
    with pytest.raises(ValueError, match="file_weight"):
        component_consistent_file_probability(.2, .3, .4, .5, .6, file_weight=1.1)
    with pytest.raises(ValueError, match="probabilities"):
        component_consistent_file_probability(.2, 1.1, .4, .5, .6)
    with pytest.raises(ValueError, match="finite"):
        component_consistent_file_probability(.2, np.nan, .4, .5, .6)
