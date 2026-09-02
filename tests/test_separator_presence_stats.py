import numpy as np

from scripts.extract_separator_presence_stats import statistics


def test_statistics_are_finite_and_component_ordered():
    time = np.arange(16_000, dtype=np.float32) / 16_000
    voice = 0.2 * np.sin(2 * np.pi * 220 * time)
    music = 0.05 * np.sin(2 * np.pi * 880 * time)
    result = statistics(voice + music, voice, music)

    assert result["VOICE_STEM_SHARE"] > result["MUSIC_STEM_SHARE"]
    assert result["VOICE_ENERGY_RATIO"] > result["MUSIC_ENERGY_RATIO"]
    assert result["VOICE_DOMINANT_FRACTION"] == 1.0
    assert result["RECONSTRUCTION_ERROR"] < 1e-5
    assert all(np.isfinite(value) for value in result.values())


def test_statistics_support_short_audio():
    original = np.zeros(200, dtype=np.float32)
    result = statistics(original, original.copy(), original.copy())

    assert result["VOICE_FRAME_RATIO_Q100"] >= 0
    assert result["MUSIC_FRAME_RATIO_Q100"] >= 0
    assert all(np.isfinite(value) for value in result.values())
