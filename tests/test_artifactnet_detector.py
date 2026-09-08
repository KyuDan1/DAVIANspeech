import numpy as np

from src.artifactnet_detector import ArtifactNetMusicDetector


def test_artifactnet_windows_include_nonoverlap_and_tail():
    window = ArtifactNetMusicDetector.WINDOW
    audio = np.arange(2 * window + 17, dtype=np.float32)
    chunks = ArtifactNetMusicDetector._windows(audio)
    assert chunks.shape == (3, window)
    np.testing.assert_array_equal(chunks[0], audio[:window])
    np.testing.assert_array_equal(chunks[1], audio[window:2 * window])
    np.testing.assert_array_equal(chunks[-1], audio[-window:])


def test_artifactnet_short_audio_is_zero_padded():
    audio = np.ones(31, dtype=np.float32)
    chunks = ArtifactNetMusicDetector._windows(audio)
    assert chunks.shape == (1, ArtifactNetMusicDetector.WINDOW)
    np.testing.assert_array_equal(chunks[0, :31], audio)
    assert np.count_nonzero(chunks[0, 31:]) == 0
