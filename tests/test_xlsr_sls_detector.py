import numpy as np
import pytest

from src.xlsr_sls_detector import (
    fixed_window, logmeanexp_probability, segment_starts,
)


def test_segment_starts_covers_tail_without_duplicate():
    assert segment_starts(64_600) == [0]
    assert segment_starts(129_300) == [0, 64_600, 64_700]


def test_fixed_window_tiles_short_audio_and_crops_long_audio():
    short = np.asarray([1, 2, 3], dtype=np.float32)
    assert fixed_window(short, 0, 8).tolist() == [1, 2, 3, 1, 2, 3, 1, 2]
    long = np.arange(12, dtype=np.float32)
    assert fixed_window(long, 3, 4).tolist() == [3, 4, 5, 6]


def test_logmeanexp_is_length_invariant_for_equal_scores():
    assert logmeanexp_probability(np.asarray([0.3])) == pytest.approx(0.3)
    assert logmeanexp_probability(np.full(20, 0.3)) == pytest.approx(0.3)
    value = logmeanexp_probability(np.asarray([0.1, 0.9]))
    assert 0.5 < value < 0.9
