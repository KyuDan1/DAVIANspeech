import pytest
import torch

from src.sofia_mert_detector import SofiaMertDetector


def test_temporal_statistics_preserve_layers_and_three_views():
    hidden = torch.arange(13 * 12 * 768, dtype=torch.float32).reshape(13, 12, 768)

    statistics = SofiaMertDetector._temporal_statistics(hidden)

    assert statistics.shape == (3, 13, 2, 768)
    assert torch.allclose(statistics[0, :, 0], hidden[:, :4].mean(dim=1))
    assert torch.allclose(
        statistics[2, :, 1], hidden[:, 8:].std(dim=1, unbiased=False)
    )


def test_temporal_statistics_reject_too_short_sequence():
    hidden = torch.zeros(13, 2, 768)

    with pytest.raises(ValueError, match="too short"):
        SofiaMertDetector._temporal_statistics(hidden)
