from __future__ import annotations

import csv

import numpy as np
import pytest
import torch

from src.pipeline import (
    combine,
    fake_probability,
    find_audio_files,
    order_by_submission,
    read_sample_submission,
)
from src.presence import extract_segment, segment_starts


def test_segment_starts_covers_tail_without_duplicate():
    assert segment_starts(10, 10) == [0]
    assert segment_starts(25, 10) == [0, 10, 15]


def test_extract_segment_repeats_short_audio():
    audio = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    np.testing.assert_array_equal(extract_segment(audio, 0, 8), [1, 2, 3, 1, 2, 3, 1, 2])


def test_submission_order_and_validation(tmp_path):
    test_dir = tmp_path / "test"
    test_dir.mkdir()
    (test_dir / "b.wav").touch()
    (test_dir / "a.flac").touch()
    (test_dir / "ignored.txt").touch()

    sample = tmp_path / "sample.csv"
    columns = [
        "ID", "FILE_FAKE_PROB", "VOICE_FAKE_PROB", "MUSIC_FAKE_PROB",
        "VOICE_PRESENT_PROB", "MUSIC_PRESENT_PROB",
    ]
    with sample.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows([{"ID": "b"}, {"ID": "a"}])

    files = find_audio_files(test_dir)
    parsed_columns, rows = read_sample_submission(sample)
    assert parsed_columns == columns
    assert [p.stem for p in order_by_submission(files, rows)] == ["b", "a"]

    rows.append({"ID": "missing"})
    with pytest.raises(ValueError, match="disagree"):
        order_by_submission(files, rows)


class _Detector:
    def fake_probability(self, batch):
        return torch.arange(1, len(batch) + 1, device=batch.device) / 10


def test_fake_probability_batches_and_silence():
    audio = np.ones(25, dtype=np.float32)
    assert fake_probability(_Detector(), audio, "cpu", window=10, batch_size=2) == pytest.approx(0.2)
    assert fake_probability(_Detector(), np.zeros(10, dtype=np.float32), "cpu", 10, 2) == 0.0


def test_combine_uses_max_fake_among_present_components():
    assert combine(0.8, 0.4, 0.7, 0.9) == pytest.approx(0.8)
    assert combine(0.8, 0.4, 0.1, 0.9) == pytest.approx(0.4)
    assert combine(0.8, 0.4, 0.1, 0.2) == pytest.approx(0.4)
