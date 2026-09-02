import numpy as np

from scripts.build_temporal_mixed_train import MODES, SR, mix_with_intervals, plan_rows


def test_temporal_plan_is_balanced():
    rows = list(plan_rows(3, 1))
    assert len(rows) == len(MODES) * 4 * 4
    assert sum(row[0] == "train" for row in rows) == len(MODES) * 4 * 3
    assert {(row[2], row[3]) for row in rows} == {(0, 0), (0, 1), (1, 0), (1, 1)}


def test_temporal_mixers_return_valid_intervals_and_audio():
    voice = np.linspace(-0.2, 0.2, 7 * SR, dtype=np.float32)
    music = np.sin(np.arange(35 * SR, dtype=np.float32) / 17) * 0.1
    for mode in MODES:
        audio, voice_interval, music_interval = mix_with_intervals(
            voice, music, mode, 0, f"test-{mode}"
        )
        assert 4 * SR <= audio.size <= 60 * SR
        assert np.isfinite(audio).all()
        assert np.abs(audio).max() <= 0.98 + 1e-6
        for start, end in (voice_interval, music_interval):
            assert 0 <= start < end <= audio.size


def test_sparse_modes_really_localize_one_component():
    voice = np.ones(9 * SR, dtype=np.float32) * 0.1
    music = np.ones(40 * SR, dtype=np.float32) * 0.1
    audio, voice_interval, music_interval = mix_with_intervals(
        voice, music, "sparse_voice", 0, "voice"
    )
    assert voice_interval[1] - voice_interval[0] < audio.size
    assert music_interval == (0, audio.size)
    audio, voice_interval, music_interval = mix_with_intervals(
        voice, music, "sparse_music", 0, "music"
    )
    assert music_interval[1] - music_interval[0] < audio.size
    assert voice_interval == (0, audio.size)
