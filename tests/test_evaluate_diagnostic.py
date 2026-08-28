from __future__ import annotations

import pandas as pd

from src.evaluate_diagnostic import evaluate_diagnostic, official_eer


def test_official_eer():
    assert official_eer([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9]) == 0.0
    assert official_eer([0, 0, 1, 1], [0.9, 0.8, 0.2, 0.1]) == 1.0


def test_official_score_and_presence_conditioning(tmp_path):
    truth = pd.DataFrame({
        "ID": ["a", "b", "c", "d"],
        "FILE_FAKE": [0, 0, 1, 1],
        "VOICE_FAKE": [0, 1, 1, 0],
        "MUSIC_FAKE": [1, 0, 0, 1],
        "VOICE_PRESENT": [1, 0, 1, 0],
        "MUSIC_PRESENT": [0, 1, 0, 1],
        "AUDIO_TYPE": ["voice", "music", "voice", "music"],
    })
    prediction = pd.DataFrame({
        "ID": ["d", "c", "b", "a"],
        "FILE_FAKE_PROB": [0.9, 0.8, 0.2, 0.1],
        "VOICE_FAKE_PROB": [0.9, 0.8, 0.9, 0.1],
        "MUSIC_FAKE_PROB": [0.9, 0.1, 0.1, 0.9],
        "VOICE_PRESENT_PROB": [0.1, 0.9, 0.1, 0.9],
        "MUSIC_PRESENT_PROB": [0.9, 0.1, 0.9, 0.1],
    })
    truth_path, prediction_path = tmp_path / "truth.csv", tmp_path / "prediction.csv"
    truth.to_csv(truth_path, index=False)
    prediction.to_csv(prediction_path, index=False)

    table = evaluate_diagnostic(prediction_path, truth_path, min_group_size=2)
    overall = table.iloc[0]
    assert overall["VOICE_N"] == 2
    assert overall["MUSIC_N"] == 2
    assert overall["ADS"] == 1.0
    assert overall["CPS"] == 1.0
    assert overall["SCORE"] == 1.0
    assert set(table[table["GROUP"] == "AUDIO_TYPE"]["VALUE"]) == {"music", "voice"}
