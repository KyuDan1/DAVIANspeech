import numpy as np
import pytest

from src.channel_invariant_music_inference import residual_music_fusion


def test_zero_and_one_residual_weights_are_endpoints():
    anchor = np.array([.1, .7], dtype=np.float32)
    expert = np.array([.8, .2], dtype=np.float32)
    np.testing.assert_allclose(residual_music_fusion(anchor, expert, 0), anchor, atol=1e-6)
    np.testing.assert_allclose(residual_music_fusion(anchor, expert, 1), expert, atol=1e-6)


def test_equal_probabilities_are_fixed_point():
    score = np.array([.02, .3, .91], dtype=np.float32)
    np.testing.assert_allclose(residual_music_fusion(score, score, .2), score, atol=1e-6)


def test_bad_weight_fails_closed():
    with pytest.raises(ValueError):
        residual_music_fusion(np.array([.5]), np.array([.5]), 1.1)
