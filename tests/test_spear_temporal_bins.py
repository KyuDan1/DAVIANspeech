import numpy as np
import pytest
import torch

from src.spear_temporal_bins import (
    audio_bin_ranges,
    projected_temporal_bins,
    random_projection,
    temporal_bin_boundaries,
)


def test_temporal_bins_are_contiguous_and_nonempty():
    assert temporal_bin_boundaries(17, 4) == [(0, 4), (4, 8), (8, 12), (12, 17)]
    with pytest.raises(ValueError):
        temporal_bin_boundaries(3, 4)


def test_projected_bins_preserve_batch_bin_layer_and_stat_axes():
    hidden = [torch.randn(2, 16, 6) for _ in range(5)]
    projection = torch.from_numpy(random_projection(6, 3, seed=7))
    result = projected_temporal_bins(hidden, projection, (0, 2, 4), bins=4)
    assert result.shape == (2, 4, 3, 4, 3)
    assert torch.isfinite(result).all()


def test_audio_bin_mask_excludes_zero_padding():
    ranges, mask = audio_bin_ranges(5, 0, 8, bins=4)
    np.testing.assert_array_equal(ranges, [[0, 2], [2, 4], [4, 5], [5, 5]])
    np.testing.assert_array_equal(mask, [True, True, True, False])
