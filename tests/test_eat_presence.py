import numpy as np

from src.eat_presence import EatPresence, fuse_music_probe, fuse_presence
from src.eat_presence_fusion import (
    combine_with_gate, latent_linear_probability, logit_fuse_presence,
)


def test_temporal_views_cover_start_middle_and_end():
    audio = np.arange(20, dtype=np.float32)
    original = EatPresence.SAMPLES
    EatPresence.SAMPLES = 6
    try:
        views = EatPresence.temporal_views(audio)
    finally:
        EatPresence.SAMPLES = original
    assert [view.tolist() for view in views] == [
        list(range(6)), list(range(7, 13)), list(range(14, 20))
    ]


def test_presence_fusion_and_file_gate_are_independent():
    voice, music = fuse_presence(0.8, 0.2, 0.4, 1.0)
    assert np.isclose(voice, 0.66)
    assert np.isclose(music, 0.92)
    assert combine_with_gate(0.1, 0.9, voice, music, gate=0.6) == 0.9


def test_music_probe_fusion_is_bounded_and_weighted():
    assert np.isclose(fuse_music_probe(0.2, 0.7), 0.4)
    assert fuse_music_probe(2.0, 2.0) == 1.0


def test_logit_presence_fusion_has_exact_endpoints():
    assert np.isclose(logit_fuse_presence(0.2, 0.8, 0.0), 0.2)
    assert np.isclose(logit_fuse_presence(0.2, 0.8, 1.0), 0.8)
    assert np.isclose(logit_fuse_presence(0.2, 0.8, 0.5), 0.5)


def test_latent_linear_probability_ignores_padded_views():
    matrix = np.asarray([[[1.0, 3.0]], [[99.0, 99.0]]], dtype=np.float32)
    mask = np.asarray([True, False])
    checkpoint = {
        "mean": np.zeros(4, dtype=np.float32),
        "std": np.ones(4, dtype=np.float32),
        "coefficient": np.ones((1, 4), dtype=np.float32),
        "intercept": np.asarray([0.0], dtype=np.float32),
    }
    # Valid view contributes [mean=(1,3), max=(1,3)] -> decision 8 / 2.
    assert np.isclose(
        latent_linear_probability(matrix, mask, checkpoint, temperature=2.0),
        1 / (1 + np.exp(-4)),
    )
