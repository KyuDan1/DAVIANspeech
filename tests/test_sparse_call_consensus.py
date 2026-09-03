from pathlib import Path

import numpy as np
import pandas as pd

from src.sparse_call_consensus import (
    apply_sparse_call_consensus,
    local_features,
    score_statistics,
)


def test_local_features_encode_agreement_and_disagreement():
    result = local_features(np.array([0.5, 0.8]), np.array([0.0, 2.0]))
    assert result.shape == (2, 4)
    np.testing.assert_allclose(result[0], 0.0)
    assert result[1, 0] > 0 and result[1, 1] > 0
    assert result[1, 2] > 0


def _write_statistics(tmp_path: Path):
    xlsr = tmp_path / "xlsr.npz"
    spectra = tmp_path / "spectra.npz"
    head = tmp_path / "head.npz"
    np.savez(
        xlsr, ids=np.array(["a"]), offsets=np.array([0, 2]),
        starts=np.array([0, 64_000]), scores=np.array([0.2, 0.8]),
        durations=np.array([8.0]), window=np.array(64_000),
    )
    np.savez(
        spectra, ids=np.array(["a"]), offsets=np.array([0, 2]),
        starts=np.array([0, 64_600]), fake_margins=np.array([-1.0, 2.0]),
        durations=np.array([8.0]), window=np.array(64_600),
    )
    np.savez(
        head, feature_mean=np.zeros(4), feature_scale=np.ones(4),
        local_weight=np.array([1.0, 1.0, 0.0, 0.0]),
        local_bias=np.array(0.0), pool_temperature=np.array(2.0),
        bag_weight=np.array(1.0), bag_bias=np.array(0.0),
    )
    return xlsr, spectra, head


def test_consensus_updates_only_voice_by_default(tmp_path: Path):
    xlsr, spectra, head = _write_statistics(tmp_path)
    ids, scores = score_statistics(xlsr, spectra, head)
    assert ids.tolist() == ["a"]
    assert 0 < scores[0] < 1
    submission = tmp_path / "submission.csv"
    original = pd.DataFrame({
        "ID": ["a"], "FILE_FAKE_PROB": [0.4], "VOICE_FAKE_PROB": [0.4],
        "MUSIC_FAKE_PROB": [0.3], "VOICE_PRESENT_PROB": [0.9],
        "MUSIC_PRESENT_PROB": [0.1],
    })
    original.to_csv(submission, index=False)
    apply_sparse_call_consensus(submission, xlsr, spectra, head, voice_weight=.5)
    updated = pd.read_csv(submission)
    assert updated.VOICE_FAKE_PROB.iloc[0] != original.VOICE_FAKE_PROB.iloc[0]
    for column in original.columns.drop(["ID", "VOICE_FAKE_PROB"]):
        assert updated[column].iloc[0] == original[column].iloc[0]


def test_consensus_can_restore_voice_only_file_consistency(tmp_path: Path):
    xlsr, spectra, head = _write_statistics(tmp_path)
    submission = tmp_path / "submission.csv"
    original = pd.DataFrame({
        "ID": ["a"], "FILE_FAKE_PROB": [0.4], "VOICE_FAKE_PROB": [0.4],
        "MUSIC_FAKE_PROB": [0.3], "VOICE_PRESENT_PROB": [0.9],
        "MUSIC_PRESENT_PROB": [0.1],
    })
    original.to_csv(submission, index=False)
    apply_sparse_call_consensus(
        submission, xlsr, spectra, head, voice_weight=.5,
        file_voice_only_weight=.4,
    )
    updated = pd.read_csv(submission)
    assert updated.VOICE_FAKE_PROB.iloc[0] != original.VOICE_FAKE_PROB.iloc[0]
    assert updated.FILE_FAKE_PROB.iloc[0] != original.FILE_FAKE_PROB.iloc[0]
    for column in original.columns.drop(
        ["ID", "VOICE_FAKE_PROB", "FILE_FAKE_PROB"]
    ):
        assert updated[column].iloc[0] == original[column].iloc[0]


def test_file_consistency_does_not_touch_mixed_audio(tmp_path: Path):
    xlsr, spectra, head = _write_statistics(tmp_path)
    submission = tmp_path / "submission.csv"
    original = pd.DataFrame({
        "ID": ["a"], "FILE_FAKE_PROB": [0.4], "VOICE_FAKE_PROB": [0.4],
        "MUSIC_FAKE_PROB": [0.3], "VOICE_PRESENT_PROB": [0.9],
        "MUSIC_PRESENT_PROB": [0.9],
    })
    original.to_csv(submission, index=False)
    apply_sparse_call_consensus(
        submission, xlsr, spectra, head, voice_weight=.5,
        file_voice_only_weight=.4,
    )
    updated = pd.read_csv(submission)
    assert updated.FILE_FAKE_PROB.iloc[0] == original.FILE_FAKE_PROB.iloc[0]
