import csv
from pathlib import Path

import numpy as np

from src.wpt_spectra_inference import (
    apply_standalone_task_fusion,
    aggregate_view_logits, apply_fixed_task_moe_fusion, fixed_windows,
    nested_view_indices,
)


def test_fixed_windows_are_deterministic_and_repeat_short_audio():
    short = np.arange(3, dtype=np.float32)
    expected = np.asarray([0, 1, 2, 0, 1], dtype=np.float32)
    np.testing.assert_array_equal(fixed_windows(short, 5, 2)[0], expected)
    long = np.arange(10, dtype=np.float32)
    windows = fixed_windows(long, 4, 3)
    np.testing.assert_array_equal(windows[:, 0], [0, 3, 6])


def test_five_file_views_nest_original_three_component_views():
    np.testing.assert_array_equal(nested_view_indices(5, 3), [0, 2, 4])
    np.testing.assert_array_equal(nested_view_indices(3, 3), [0, 1, 2])


def test_view_logmeanexp_is_length_normalized():
    one = np.asarray([[[.2, -.4, 1.3]]], dtype=np.float32)
    import torch
    logits = torch.from_numpy(one)
    repeated = logits.repeat(1, 5, 1)
    torch.testing.assert_close(
        aggregate_view_logits(logits, 2.0),
        aggregate_view_logits(repeated, 2.0),
    )


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


def test_standalone_wpt_preserves_music_and_presence(tmp_path: Path):
    ids = np.asarray(["a", "b"])
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
    apply_standalone_task_fusion(
        submission, ids, wpt, voice_weight=.075, file_weight=.01
    )
    with submission.open(newline="") as handle:
        updated = list(csv.DictReader(handle))
    for before, after in zip(original, updated):
        assert float(after["FILE_FAKE_PROB"]) != before["FILE_FAKE_PROB"]
        assert float(after["VOICE_FAKE_PROB"]) != before["VOICE_FAKE_PROB"]
        for column in (
            "MUSIC_FAKE_PROB", "VOICE_PRESENT_PROB", "MUSIC_PRESENT_PROB",
        ):
            assert float(after[column]) == before[column]


def test_standalone_wpt_safe_defaults_change_voice_only(tmp_path: Path):
    ids = np.asarray(["a"])
    columns = [
        "ID", "FILE_FAKE_PROB", "VOICE_FAKE_PROB", "MUSIC_FAKE_PROB",
        "VOICE_PRESENT_PROB", "MUSIC_PRESENT_PROB",
    ]
    original = {
        "ID": "a", "FILE_FAKE_PROB": .45, "VOICE_FAKE_PROB": .35,
        "MUSIC_FAKE_PROB": .55, "VOICE_PRESENT_PROB": .65,
        "MUSIC_PRESENT_PROB": .75,
    }
    submission = tmp_path / "submission.csv"
    with submission.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerow(original)
    apply_standalone_task_fusion(
        submission, ids, np.asarray([[.9, .1, .8]], np.float32)
    )
    with submission.open(newline="") as handle:
        updated = next(csv.DictReader(handle))
    assert float(updated["VOICE_FAKE_PROB"]) != original["VOICE_FAKE_PROB"]
    for column in (
        "FILE_FAKE_PROB", "MUSIC_FAKE_PROB", "VOICE_PRESENT_PROB",
        "MUSIC_PRESENT_PROB",
    ):
        assert float(updated[column]) == original[column]
