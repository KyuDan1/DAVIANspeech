import numpy as np
import pandas as pd

from scripts.scale_three_stream_residual import (
    logit, noisy_or_logit, scale_residual, sigmoid,
)


def prediction() -> pd.DataFrame:
    anchor = np.asarray([
        [-1.0, 0.2, -0.4],
        [0.4, -0.7, 0.8],
    ])
    residual = np.asarray([
        [0.3, -0.2, 0.5],
        [-0.1, 0.6, -0.4],
    ])
    voice = anchor[:, 0] + residual[:, 0]
    music = anchor[:, 1] + residual[:, 1]
    file_logit = (
        anchor[:, 2] + 0.7 * residual[:, 2]
        + 0.3 * (
            noisy_or_logit(voice, music)
            - noisy_or_logit(anchor[:, 0], anchor[:, 1])
        )
    )
    return pd.DataFrame({
        "ID": ["a", "b"],
        "VOICE_FAKE_PROB": sigmoid(voice),
        "MUSIC_FAKE_PROB": sigmoid(music),
        "FILE_FAKE_PROB": sigmoid(file_logit),
        "VOICE_LOGIT_RESIDUAL": residual[:, 0],
        "MUSIC_LOGIT_RESIDUAL": residual[:, 1],
        "DIRECT_FILE_LOGIT_RESIDUAL": residual[:, 2],
    })


def test_unit_scale_reproduces_prediction_and_zero_scale_recovers_anchor():
    frame = prediction()
    unit = scale_residual(frame, 1.0)
    for column in ("VOICE_FAKE_PROB", "MUSIC_FAKE_PROB", "FILE_FAKE_PROB"):
        np.testing.assert_allclose(unit[column], frame[column], rtol=0, atol=1e-12)
    zero = scale_residual(frame, 0.0)
    np.testing.assert_allclose(logit(zero.VOICE_FAKE_PROB), [-1.0, 0.4])
    np.testing.assert_allclose(logit(zero.MUSIC_FAKE_PROB), [0.2, -0.7])
    np.testing.assert_allclose(logit(zero.FILE_FAKE_PROB), [-0.4, 0.8])
    assert not zero[list((
        "VOICE_LOGIT_RESIDUAL", "MUSIC_LOGIT_RESIDUAL",
        "DIRECT_FILE_LOGIT_RESIDUAL",
    ))].to_numpy().any()
