from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts.score_all_type_blind_one_shot import (
    metrics,
    official_eer,
    validate_and_align,
)


def test_official_eer_perfect_and_reversed():
    assert official_eer([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9]) == 0
    assert official_eer([0, 0, 1, 1], [0.9, 0.8, 0.2, 0.1]) == 1


def test_validate_and_score_all_types(tmp_path: Path):
    truth = pd.DataFrame({
        "ID": ["a", "b", "c", "d"],
        "CHANNEL": ["clean", "g711_ulaw", "clean", "g722_wb"],
        "MIX_MODE": ["voice_only", "music_only", "concurrent", "sequential"],
        "FILE_FAKE": [0, 1, 0, 1],
        "VOICE_FAKE": [0, np.nan, 0, 1],
        "MUSIC_FAKE": [np.nan, 1, 0, 0],
        "VOICE_PRESENT": [1, 0, 1, 1],
        "MUSIC_PRESENT": [0, 1, 1, 1],
    })
    prediction = pd.DataFrame({
        "ID": ["d", "b", "a", "c"],
        "FILE_FAKE_PROB": [.9, .8, .1, .2],
        "VOICE_FAKE_PROB": [.9, .5, .1, .2],
        "MUSIC_FAKE_PROB": [.1, .9, .5, .2],
        "VOICE_PRESENT_PROB": [.9, .1, .9, .9],
        "MUSIC_PRESENT_PROB": [.9, .9, .1, .9],
    })
    truth_path, prediction_path = tmp_path / "truth.csv", tmp_path / "pred.csv"
    truth.to_csv(truth_path, index=False)
    prediction.to_csv(prediction_path, index=False)
    frame = validate_and_align(truth_path, prediction_path)
    assert frame.AUDIO_TYPE.tolist() == [
        "voice_only", "music_only", "mixed", "mixed",
    ]
    score = metrics(frame)
    assert score["AVAILABLE_ADS"] == pytest.approx(1.0)
    assert score["CPS"] == pytest.approx(1.0)


def test_validate_rejects_invalid_file_or(tmp_path: Path):
    truth = pd.DataFrame({
        "ID": ["x"], "CHANNEL": ["clean"], "MIX_MODE": ["voice_only"],
        "FILE_FAKE": [0], "VOICE_FAKE": [1], "MUSIC_FAKE": [0],
        "VOICE_PRESENT": [1], "MUSIC_PRESENT": [0],
    })
    prediction = pd.DataFrame({
        "ID": ["x"],
        **{column: [0.5] for column in (
            "FILE_FAKE_PROB", "VOICE_FAKE_PROB", "MUSIC_FAKE_PROB",
            "VOICE_PRESENT_PROB", "MUSIC_PRESENT_PROB",
        )},
    })
    truth_path, prediction_path = tmp_path / "truth.csv", tmp_path / "pred.csv"
    truth.to_csv(truth_path, index=False)
    prediction.to_csv(prediction_path, index=False)
    with pytest.raises(ValueError, match="OR"):
        validate_and_align(truth_path, prediction_path)
