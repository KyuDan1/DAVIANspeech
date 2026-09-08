import numpy as np
import pytest
import torch

from src.spear_temporal_bins import (
    audio_bin_ranges,
    projected_temporal_bins,
    random_projection,
    temporal_bin_boundaries,
)
from src.spear_detector import SpearCrossComponentDetector


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


def test_multiple_temporal_projections_share_one_encoder_forward():
    class FakeModel:
        def __init__(self):
            self.calls = 0

        def __call__(self, waveform, lengths):
            self.calls += 1
            batch = len(waveform)
            base = torch.arange(
                batch * 16 * 6, dtype=torch.float32
            ).reshape(batch, 16, 6)
            return {"hidden_states": [base + layer for layer in range(13)]}

    detector = object.__new__(SpearCrossComponentDetector)
    detector.device = torch.device("cpu")
    detector.model = FakeModel()
    detector.window = 32
    detector.max_windows = 3
    detector.dimension = 6
    configurations = [
        (np.eye(6, 3, dtype=np.float32), (0, 2), 4),
        (np.eye(6, 2, dtype=np.float32), (1, 3, 5), 8),
    ]
    statistics, outputs = (
        detector.dual_domain_statistics_and_multiple_temporal_bins_batch(
            [np.ones(64, dtype=np.float32)], configurations
        )
    )
    assert detector.model.calls == 1
    assert statistics[0][0].shape == (3, 13, 4, 6)
    assert outputs[0][0][0].shape == (3, 4, 2, 4, 3)
    assert outputs[1][0][0].shape == (3, 8, 3, 4, 2)
    assert outputs[0][0][1].shape == (3, 4)
    assert outputs[1][0][1].shape == (3, 8)
