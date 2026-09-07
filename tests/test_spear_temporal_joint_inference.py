from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from src.spear_temporal_joint_head import SpearTemporalJointHead
from src.spear_temporal_joint_inference import (
    apply_spear_temporal_joint_fusion,
    predict_temporal_joint,
)


def test_joint_checkpoint_prediction_shape(tmp_path: Path):
    model = SpearTemporalJointHead(
        4, torch.zeros(4), torch.ones(4), hidden=3, dropout=0
    )
    checkpoint = tmp_path / "head.pt"
    torch.save({
        "model": model.state_dict(),
        "config": {
            "feature_dimension": 4, "hidden": 3, "dropout": 0,
            "temperature": 5.0, "minimum_presence_weight": .05,
        },
    }, checkpoint)
    statistics = tmp_path / "statistics.npz"
    np.savez(
        statistics,
        ids=np.asarray(["a", "b"]),
        features=np.zeros((2, 1, 2, 1, 2, 2), np.float16),
        mask=np.ones((2, 1, 2), bool),
    )
    ids, probabilities = predict_temporal_joint(
        statistics, checkpoint, device="cpu"
    )
    assert ids.tolist() == ["a", "b"]
    assert probabilities.shape == (2, 5)
    assert np.all((probabilities > 0) & (probabilities < 1))


def test_joint_fusion_changes_file_voice_only(tmp_path: Path, monkeypatch):
    submission = tmp_path / "submission.csv"
    original = pd.DataFrame({
        "ID": ["a", "b"], "FILE_FAKE_PROB": [.2, .8],
        "VOICE_FAKE_PROB": [.3, .7], "MUSIC_FAKE_PROB": [.1, .9],
        "VOICE_PRESENT_PROB": [.4, .6], "MUSIC_PRESENT_PROB": [.5, .5],
    })
    original.to_csv(submission, index=False)
    monkeypatch.setattr(
        "src.spear_temporal_joint_inference.predict_temporal_joint",
        lambda *args, **kwargs: (
            np.asarray(["b", "a"]),
            np.asarray([[.1, .2, .3, .4, .5], [.9, .8, .7, .6, .5]]),
        ),
    )
    apply_spear_temporal_joint_fusion(
        submission, tmp_path / "stats.npz", tmp_path / "head.pt", device="cpu"
    )
    changed = pd.read_csv(submission)
    assert not np.allclose(changed.FILE_FAKE_PROB, original.FILE_FAKE_PROB)
    assert not np.allclose(changed.VOICE_FAKE_PROB, original.VOICE_FAKE_PROB)
    for column in ("MUSIC_FAKE_PROB", "VOICE_PRESENT_PROB", "MUSIC_PRESENT_PROB"):
        np.testing.assert_allclose(changed[column], original[column])


def test_joint_fusion_rejects_invalid_weight(tmp_path: Path):
    with pytest.raises(ValueError, match="file_weight"):
        apply_spear_temporal_joint_fusion(
            tmp_path / "missing.csv", tmp_path / "missing.npz",
            tmp_path / "missing.pt", file_weight=1.1,
        )


def test_joint_fusion_routes_phone_weights(tmp_path: Path, monkeypatch):
    submission = tmp_path / "submission.csv"
    pd.DataFrame({
        "ID": ["clean", "phone"], "FILE_FAKE_PROB": [.5, .5],
        "VOICE_FAKE_PROB": [.5, .5], "MUSIC_FAKE_PROB": [.5, .5],
        "VOICE_PRESENT_PROB": [.5, .5], "MUSIC_PRESENT_PROB": [.5, .5],
    }).to_csv(submission, index=False)
    monkeypatch.setattr(
        "src.spear_temporal_joint_inference.predict_temporal_joint",
        lambda *args, **kwargs: (
            np.asarray(["clean", "phone"]),
            np.full((2, 5), .9, dtype=np.float32),
        ),
    )
    routed = tmp_path / "phone_ids.npz"
    np.savez(routed, ids=np.asarray(["phone"]))
    apply_spear_temporal_joint_fusion(
        submission, tmp_path / "stats.npz", tmp_path / "head.pt",
        device="cpu", file_weight=.1, voice_weight=.2,
        telephone_ids_path=routed, phone_file_weight=.4,
        phone_voice_weight=.35,
    )
    result = pd.read_csv(submission).set_index("ID")
    assert result.loc["phone", "FILE_FAKE_PROB"] > result.loc["clean", "FILE_FAKE_PROB"]
    assert result.loc["phone", "VOICE_FAKE_PROB"] > result.loc["clean", "VOICE_FAKE_PROB"]
