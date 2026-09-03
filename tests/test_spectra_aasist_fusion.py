from __future__ import annotations

import csv

import numpy as np
import torch

from src.spectra_aasist_detector import (
    SpectraStemScorer,
    _fixed_windows,
    apply_spectra_voice_fusion,
)


COLUMNS = [
    "ID", "FILE_FAKE_PROB", "VOICE_FAKE_PROB", "MUSIC_FAKE_PROB",
    "VOICE_PRESENT_PROB", "MUSIC_PRESENT_PROB",
]


def test_fixed_windows_are_deterministic_and_fixed_length():
    short = np.arange(10, dtype=np.float32)
    assert _fixed_windows(short, 16, 3).shape == (3, 16)
    np.testing.assert_array_equal(
        _fixed_windows(short, 16, 3)[0], _fixed_windows(short, 16, 3)[2]
    )
    long = np.arange(100, dtype=np.float32)
    windows = _fixed_windows(long, 20, 3)
    np.testing.assert_array_equal(windows[:, 0], [0, 40, 80])


def test_stem_scorer_can_save_aligned_sliding_windows(tmp_path, monkeypatch):
    class DummyModel(torch.nn.Module):
        def forward(self, waveforms):
            means = waveforms.mean(dim=1)
            return torch.stack((means, -means), dim=1)

    monkeypatch.setattr(
        "src.spectra_aasist_detector._load_model",
        lambda _path, _device: DummyModel(),
    )
    scorer = SpectraStemScorer(
        tmp_path, device="cpu", windows=3, file_batch_size=1,
        window_samples=4, collect_sliding=True,
    )
    scorer.add("sample", np.arange(10, dtype=np.float32))
    fixed_path = tmp_path / "fixed.npz"
    sliding_path = tmp_path / "sliding.npz"
    scorer.save(fixed_path, sliding_path)

    fixed = np.load(fixed_path)
    sliding = np.load(sliding_path)
    assert fixed["ids"].tolist() == ["sample"]
    assert fixed["fake_margin"].shape == (1,)
    assert sliding["offsets"].tolist() == [0, 3]
    assert sliding["starts"].tolist() == [0, 4, 6]
    assert sliding["fake_margins"].shape == (3,)
    assert sliding["valid"].tolist() == [True]


def test_spectra_fusion_changes_only_intended_columns(tmp_path):
    submission = tmp_path / "submission.csv"
    rows = [
        # Valid and Voice dominates: both File and Voice must change.
        dict(zip(COLUMNS, ["voice", .6, .7, .2, .9, .1])),
        # Valid but Music dominates: Voice changes, File is preserved.
        dict(zip(COLUMNS, ["music", .8, .2, .9, .8, .9])),
        # Silent/invalid stem: neither prediction changes.
        dict(zip(COLUMNS, ["silent", .3, .4, .1, .9, .1])),
    ]
    with submission.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    statistics = tmp_path / "spectra.npz"
    np.savez(
        statistics,
        ids=np.asarray(["voice", "music", "silent"]),
        fake_margin=np.asarray([2.0, -2.0, 9.0], dtype=np.float32),
        valid=np.asarray([True, True, False]),
    )

    apply_spectra_voice_fusion(
        submission, statistics, voice_weight=.1, file_weight=.05
    )
    with submission.open(encoding="utf-8", newline="") as handle:
        result = {row["ID"]: row for row in csv.DictReader(handle)}

    assert float(result["voice"]["FILE_FAKE_PROB"]) != .6
    assert float(result["voice"]["VOICE_FAKE_PROB"]) != .7
    assert float(result["music"]["FILE_FAKE_PROB"]) == .8
    assert float(result["music"]["VOICE_FAKE_PROB"]) != .2
    for column, original in zip(COLUMNS[1:], [.3, .4, .1, .9, .1]):
        assert float(result["silent"][column]) == original
    for item, original in zip(("voice", "music"), rows[:2]):
        for column in (
            "MUSIC_FAKE_PROB", "VOICE_PRESENT_PROB", "MUSIC_PRESENT_PROB"
        ):
            assert float(result[item][column]) == float(original[column])
