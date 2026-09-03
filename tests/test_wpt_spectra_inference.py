import csv
from pathlib import Path

import numpy as np

from src.wpt_spectra_inference import (
    apply_fixed_task_moe_fusion, fixed_windows,
)


def test_fixed_windows_are_deterministic_and_repeat_short_audio():
    short = np.arange(3, dtype=np.float32)
    expected = np.asarray([0, 1, 2, 0, 1], dtype=np.float32)
    np.testing.assert_array_equal(fixed_windows(short, 5, 2)[0], expected)
    long = np.arange(10, dtype=np.float32)
    windows = fixed_windows(long, 4, 3)
    np.testing.assert_array_equal(windows[:, 0], [0, 3, 6])


def test_fixed_moe_changes_only_voice_and_file(tmp_path: Path):
    ids = np.asarray(["a", "b"])
    unified = tmp_path / "unified.npz"
    np.savez(
        unified, ids=ids,
        probabilities=np.asarray([[.2, .3, .4], [.8, .7, .6]], np.float32),
    )
    columns = [
        "ID", "FILE_FAKE_PROB", "VOICE_FAKE_PROB", "MUSIC_FAKE_PROB",
        "VOICE_PRESENT_PROB", "MUSIC_PRESENT_PROB",
    ]
    original = [{
        "ID": item, "FILE_FAKE_PROB": .45, "VOICE_FAKE_PROB": .35,
        "MUSIC_FAKE_PROB": .55, "VOICE_PRESENT_PROB": .65,
        "MUSIC_PRESENT_PROB": .75,
    } for item in ids]
    submission = tmp_path / "submission.csv"
    with submission.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(original)
    wpt = np.asarray([[.9, .1, .8], [.1, .9, .2]], np.float32)
    apply_fixed_task_moe_fusion(submission, unified, ids, wpt)
    with submission.open(newline="") as handle:
        updated = list(csv.DictReader(handle))
    for before, after in zip(original, updated):
        assert float(after["FILE_FAKE_PROB"]) != before["FILE_FAKE_PROB"]
        assert float(after["VOICE_FAKE_PROB"]) != before["VOICE_FAKE_PROB"]
        for column in (
            "MUSIC_FAKE_PROB", "VOICE_PRESENT_PROB", "MUSIC_PRESENT_PROB",
        ):
            assert float(after[column]) == before[column]
