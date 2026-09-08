import numpy as np
import pandas as pd
import pytest

from scripts.train_mert_generator_robust_music_v53 import (
    codec_pairs,
    normalized_generator,
    source_balanced_weights,
)


def test_generator_aliases_collapse_model_families():
    row = pd.Series({"MUSIC_FAKE": 1, "MUSIC_GENERATOR": "MusicGen_medium"})
    assert normalized_generator(row) == "musicgen"
    row.MUSIC_GENERATOR = "audioldm2"
    assert normalized_generator(row) == "audioldm"


def test_source_balancing_removes_codec_repeat_mass():
    frame = pd.DataFrame({
        "DATASET": ["d"] * 5, "ID": list("abcde"),
        "MUSIC_FAKE": [0, 0, 0, 1, 1],
        "MUSIC_SOURCE_ID": ["r1", "r1", "r2", "f1", "f2"],
    })
    weights = source_balanced_weights(frame)
    weighted = frame.assign(W=weights).groupby("MUSIC_SOURCE_ID").W.sum()
    assert weighted.r1 == pytest.approx(weighted.r2)
    assert weighted.f1 == pytest.approx(weighted.f2)


def test_codec_pairs_are_directed_to_clean_parent_and_validated():
    frame = pd.DataFrame({
        "DATASET": ["d", "d"], "ID": ["x_clean", "x_codec"],
        "PARENT_ID": [np.nan, "x_clean"], "MIXTURE_ID": ["x", "x"],
        "CHANNEL": ["clean", "opus"], "MUSIC_FAKE": [1, 1],
        "MUSIC_SOURCE_ID": ["song", "song"],
    })
    np.testing.assert_array_equal(codec_pairs(frame), [[1, 0]])
    frame.loc[1, "MUSIC_SOURCE_ID"] = "different"
    with pytest.raises(ValueError):
        codec_pairs(frame)
