from __future__ import annotations

import csv

import numpy as np
import pytest
import torch

import src.pipeline as pipeline
from src.pipeline import (
    XLSR_EMBEDDING_DIM,
    combine,
    fake_probability,
    fake_probability_and_embedding,
    find_audio_files,
    order_by_submission,
    read_sample_submission,
    save_xlsr_window_embeddings,
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


class _EmbeddingDetector:
    def __init__(self):
        self.encoder_batches = []

    def normalize(self, batch):
        return batch

    def embedding(self, batch):
        self.encoder_batches.append(batch.clone())
        return batch[:, :1].repeat(1, XLSR_EMBEDDING_DIM)

    def proj_fc(self, pooled):
        return torch.stack((pooled[:, 0], -pooled[:, 0]), dim=1)


def test_xlsr_windows_are_returned_in_order_without_an_extra_encoder_pass():
    audio = np.arange(25, dtype=np.float32)
    detector = _EmbeddingDetector()
    score, mean, embeddings, starts = fake_probability_and_embedding(
        detector, audio, "cpu", window=10, batch_size=2,
        return_windows=True,
    )

    assert score > 0.5
    assert [len(batch) for batch in detector.encoder_batches] == [2, 1]
    assert embeddings.shape == (3, XLSR_EMBEDDING_DIM)
    assert embeddings[:, 0].tolist() == [0.0, 10.0, 15.0]
    assert starts.tolist() == [0, 10, 15]
    assert mean[0].item() == pytest.approx(25 / 3)

    legacy_result = fake_probability_and_embedding(
        _EmbeddingDetector(), audio, "cpu", window=10, batch_size=2,
    )
    assert len(legacy_result) == 2
    assert legacy_result[0] == pytest.approx(score)
    torch.testing.assert_close(legacy_result[1], mean)


def test_xlsr_window_export_is_padded_ordered_and_label_blind(tmp_path):
    first = np.full((1, XLSR_EMBEDDING_DIM), 1.25, dtype=np.float32)
    second = np.stack([
        np.full(XLSR_EMBEDDING_DIM, value, dtype=np.float32)
        for value in (2.0, 3.0, 4.0)
    ])
    first_path = tmp_path / "first.npz"
    second_path = tmp_path / "second.npz"
    arguments = (
        ["submission-b", "submission-a"], [first, second],
        [np.asarray([0]), np.asarray([0, 10, 15])], 10,
    )
    save_xlsr_window_embeddings(first_path, *arguments)
    save_xlsr_window_embeddings(second_path, *arguments)

    assert first_path.read_bytes() == second_path.read_bytes()
    with np.load(first_path, allow_pickle=False) as archive:
        assert set(archive.files) == {
            "ids", "embeddings", "mask", "starts", "window", "sample_rate",
        }
        assert archive["ids"].tolist() == ["submission-b", "submission-a"]
        assert archive["embeddings"].shape == (2, 3, XLSR_EMBEDDING_DIM)
        assert archive["embeddings"].dtype == np.float32
        np.testing.assert_array_equal(
            archive["mask"], [[True, False, False], [True, True, True]]
        )
        np.testing.assert_array_equal(
            archive["starts"], [[0, -1, -1], [0, 10, 15]]
        )
        assert not archive["embeddings"][0, 1:].any()
        assert archive["window"].item() == 10
        assert archive["sample_rate"].item() == 16_000


def test_xlsr_window_export_cleans_temp_and_preserves_target_on_error(
    tmp_path, monkeypatch,
):
    output = tmp_path / "windows.npz"
    output.write_bytes(b"existing")

    def fail_after_partial_write(handle, arrays):
        handle.write(b"partial")
        raise OSError("disk failure")

    monkeypatch.setattr(pipeline, "_write_deterministic_npz", fail_after_partial_write)
    with pytest.raises(OSError, match="disk failure"):
        save_xlsr_window_embeddings(
            output, ["a"], [np.zeros((1, XLSR_EMBEDDING_DIM))], [[0]], 10,
        )
    assert output.read_bytes() == b"existing"
    assert list(tmp_path.glob(f".{output.name}.*.tmp")) == []


def test_xlsr_window_export_rejects_wrong_embedding_shape(tmp_path):
    output = tmp_path / "windows.npz"
    with pytest.raises(ValueError, match=r"\[windows, 1920\]"):
        save_xlsr_window_embeddings(
            output, ["a"], [np.zeros((1, XLSR_EMBEDDING_DIM - 1))], [[0]], 10,
        )
    assert not output.exists()


def test_combine_uses_max_fake_among_present_components():
    assert combine(0.8, 0.4, 0.7, 0.9) == pytest.approx(0.8)
    assert combine(0.8, 0.4, 0.1, 0.9) == pytest.approx(0.4)
    assert combine(0.8, 0.4, 0.1, 0.2) == pytest.approx(0.4)
