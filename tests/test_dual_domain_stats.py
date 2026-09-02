import numpy as np
import torch

from src.dual_domain_stats import (
    crop_or_pad,
    interval_view_targets,
    sequence_statistics,
    temporal_starts,
)


def test_temporal_starts_cover_start_middle_end():
    assert temporal_starts(100, 40, 3) == [0, 30, 60]
    assert temporal_starts(20, 40, 3) == [0]


def test_interval_view_targets_localize_fake_components():
    targets, mask = interval_view_targets(
        num_samples=100, crop_samples=40,
        voice_interval=(0, 30), music_interval=(70, 100),
        voice_fake=1, music_fake=0, max_views=3,
        minimum_overlap_samples=1,
    )
    assert mask.tolist() == [True, True, True]
    assert targets.tolist() == [
        [1.0, 0.0, 1.0],
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
    ]


def test_crop_or_pad_does_not_repeat_short_audio():
    result = crop_or_pad(np.array([1, 2], dtype=np.float32), 0, 5)
    np.testing.assert_array_equal(result, [1, 2, 0, 0, 0])


def test_sequence_statistics_shape_and_constant_values():
    sequence = torch.ones(2, 5, 3)
    result = sequence_statistics(sequence)
    assert result.shape == (2, 4, 3)
    torch.testing.assert_close(result[:, 0], torch.ones(2, 3))
    torch.testing.assert_close(result[:, 1:], torch.zeros(2, 3, 3))
