import numpy as np

from src.xcodec_tokens import CROP_SAMPLES, center_crop_or_pad


def test_center_crop_is_deterministic():
    audio = np.arange(CROP_SAMPLES + 10, dtype=np.float32)
    crop = center_crop_or_pad(audio)
    np.testing.assert_array_equal(crop, audio[5:5 + CROP_SAMPLES])


def test_short_audio_is_center_padded():
    audio = np.ones(10, dtype=np.float32)
    crop = center_crop_or_pad(audio)
    assert crop.shape == (CROP_SAMPLES,)
    assert crop.sum() == 10
    first = (CROP_SAMPLES - 10) // 2
    np.testing.assert_array_equal(crop[first:first + 10], audio)
