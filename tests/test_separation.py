from __future__ import annotations

import numpy as np
import pytest
import soundfile as sf

from src.separation import PrecomputedSeparator


def test_precomputed_separator_loads_mono_16k_stems(tmp_path):
    stereo = np.column_stack([np.ones(32), np.zeros(32)]).astype(np.float32)
    sf.write(tmp_path / "clip_voice.wav", stereo, 16_000, subtype="FLOAT")
    sf.write(tmp_path / "clip_music.wav", -stereo, 16_000, subtype="FLOAT")

    voice, music = PrecomputedSeparator(tmp_path).separate("clip.mp3")
    np.testing.assert_allclose(voice, 0.5)
    np.testing.assert_allclose(music, -0.5)


def test_precomputed_separator_rejects_wrong_sample_rate(tmp_path):
    sf.write(tmp_path / "clip_voice.wav", np.zeros(32), 8_000)
    sf.write(tmp_path / "clip_music.wav", np.zeros(32), 8_000)
    with pytest.raises(ValueError, match="expected 16000"):
        PrecomputedSeparator(tmp_path).separate("clip.wav")
