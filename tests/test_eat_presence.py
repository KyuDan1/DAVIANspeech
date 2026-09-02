import numpy as np

from src.eat_presence import EatPresence, fuse_presence
from src.eat_presence_fusion import combine_with_gate


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
    assert np.isclose(voice, 0.68)
    assert np.isclose(music, 0.92)
    assert combine_with_gate(0.1, 0.9, voice, music, gate=0.6) == 0.9
