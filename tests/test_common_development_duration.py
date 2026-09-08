import pytest

from scripts.audit_common_development_duration import duration_bin


@pytest.mark.parametrize('seconds,expected', [
    (.764, 'below_4s'), (3.9999, 'below_4s'), (4., '4_to_10.24s'),
    (10.24, '4_to_10.24s'), (10.2401, '10.24_to_30s'),
    (30., '10.24_to_30s'), (30.001, '30_to_60s'),
    (60., '30_to_60s'), (60.001, 'above_60s'),
])
def test_duration_boundaries(seconds, expected):
    assert duration_bin(seconds) == expected


@pytest.mark.parametrize('seconds', [0, -1, float('nan'), float('inf')])
def test_invalid_durations_rejected(seconds):
    with pytest.raises(ValueError):
        duration_bin(seconds)
