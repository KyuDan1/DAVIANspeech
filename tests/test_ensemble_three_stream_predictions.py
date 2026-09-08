from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts.ensemble_three_stream_predictions import ensemble, logit, sigmoid


def write_prediction(path: Path, rows: list[dict]) -> None:
    pd.DataFrame(rows).to_csv(path, index=False)


def rows(probability: float) -> list[dict]:
    return [
        {
            "DATASET": "bank", "ID": sample_id,
            "FILE_FAKE_PROB": probability + offset,
            "VOICE_FAKE_PROB": probability + offset / 2,
            "MUSIC_FAKE_PROB": probability + offset / 3,
            "VOICE_PRESENT_PROB": 0.8,
            "MUSIC_PRESENT_PROB": 0.7,
        }
        for sample_id, offset in (("b", 0.1), ("a", 0.0))
    ]


def test_uniform_logit_ensemble_aligns_ids_and_preserves_presence(tmp_path: Path):
    first, second = tmp_path / "first.csv", tmp_path / "second.csv"
    write_prediction(first, rows(0.2))
    write_prediction(second, list(reversed(rows(0.6))))
    result = ensemble([first, second])
    assert result.ID.tolist() == ["a", "b"]
    assert np.array_equal(result.VOICE_PRESENT_PROB, [0.8, 0.8])
    expected = sigmoid(np.mean(logit(np.asarray([0.2, 0.6]))))
    assert result.loc[result.ID.eq("a"), "FILE_FAKE_PROB"].item() == pytest.approx(
        expected
    )


def test_uniform_logit_ensemble_rejects_presence_drift(tmp_path: Path):
    first, second = tmp_path / "first.csv", tmp_path / "second.csv"
    write_prediction(first, rows(0.2))
    changed = rows(0.6)
    changed[0]["VOICE_PRESENT_PROB"] = 0.81
    write_prediction(second, changed)
    with pytest.raises(ValueError, match="presence changed"):
        ensemble([first, second])
