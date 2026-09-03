from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.spear_temporal_bin_head import SpearTemporalBinHead
from src.spear_temporal_bin_inference import (
    apply_spear_temporal_bin_fusion,
    apply_spear_temporal_bin_music_only,
    predict_temporal_bin_music,
)


def test_temporal_bin_checkpoint_updates_music_and_file_only(tmp_path: Path):
    model = SpearTemporalBinHead(4, torch.zeros(4), torch.ones(4))
    checkpoint = tmp_path / "head.pt"
    torch.save({
        "model": model.state_dict(),
        "config": {
            "feature_dimension": 4, "hidden": 0, "dropout": .1,
            "temperature": 5.0, "minimum_presence_weight": .05,
        },
        "projection": np.zeros((1280, 2), np.float32),
        "layers": np.asarray([0]), "bins": 2,
    }, checkpoint)
    statistics = tmp_path / "statistics.npz"
    np.savez(
        statistics, ids=np.asarray(["a", "b"]),
        features=np.zeros((2, 1, 2, 1, 2, 2), np.float16),
        mask=np.ones((2, 1, 2), bool),
    )
    ids, scores = predict_temporal_bin_music(statistics, checkpoint, device="cpu")
    assert ids.tolist() == ["a", "b"]
    assert scores.shape == (2,)
    submission = tmp_path / "submission.csv"
    original = pd.DataFrame({
        "ID": ["a", "b"], "FILE_FAKE_PROB": [.2, .8],
        "VOICE_FAKE_PROB": [.3, .7], "MUSIC_FAKE_PROB": [.1, .9],
        "VOICE_PRESENT_PROB": [.4, .6], "MUSIC_PRESENT_PROB": [.5, .5],
    })
    original.to_csv(submission, index=False)
    apply_spear_temporal_bin_fusion(
        submission, statistics, checkpoint, device="cpu", music_weight=.1
    )
    changed = pd.read_csv(submission)
    assert not np.allclose(changed.MUSIC_FAKE_PROB, original.MUSIC_FAKE_PROB)
    for column in original.columns.drop(["ID", "MUSIC_FAKE_PROB", "FILE_FAKE_PROB"]):
        np.testing.assert_allclose(changed[column], original[column])
    assert not np.allclose(changed.FILE_FAKE_PROB, original.FILE_FAKE_PROB)


def test_temporal_bin_music_only_preserves_every_other_column(tmp_path: Path):
    model = SpearTemporalBinHead(4, torch.zeros(4), torch.ones(4))
    checkpoint = tmp_path / "head.pt"
    torch.save({
        "model": model.state_dict(),
        "config": {
            "feature_dimension": 4, "hidden": 0, "dropout": .1,
            "temperature": 5.0, "minimum_presence_weight": .05,
        },
    }, checkpoint)
    statistics = tmp_path / "statistics.npz"
    np.savez(
        statistics, ids=np.asarray(["a", "b"]),
        features=np.zeros((2, 1, 2, 1, 2, 2), np.float16),
        mask=np.ones((2, 1, 2), bool),
    )
    submission = tmp_path / "submission.csv"
    original = pd.DataFrame({
        "ID": ["a", "b"], "FILE_FAKE_PROB": [.2, .8],
        "VOICE_FAKE_PROB": [.3, .7], "MUSIC_FAKE_PROB": [.1, .9],
        "VOICE_PRESENT_PROB": [.4, .6], "MUSIC_PRESENT_PROB": [.5, .5],
    })
    original.to_csv(submission, index=False)
    apply_spear_temporal_bin_music_only(
        submission, statistics, checkpoint, device="cpu", music_weight=.5
    )
    changed = pd.read_csv(submission)
    assert not np.allclose(changed.MUSIC_FAKE_PROB, original.MUSIC_FAKE_PROB)
    for column in original.columns.drop(["ID", "MUSIC_FAKE_PROB"]):
        np.testing.assert_allclose(changed[column], original[column])
