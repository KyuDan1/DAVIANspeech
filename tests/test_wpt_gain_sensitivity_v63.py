import torch
from scripts.audit_wpt_gain_sensitivity_v63 import peak_normalize_windows


def test_peak_normalization_is_window_local_and_handles_silence():
    values = torch.tensor([[[1., -2., 1.], [.1, -.2, .1]], [[0., 0., 0.], [2., -4., 2.]]])
    result = peak_normalize_windows(values)
    assert torch.isfinite(result).all()
    torch.testing.assert_close(result[0, 0], result[0, 1])
    torch.testing.assert_close(result[0, 0], result[1, 1])
    assert result[1, 0].count_nonzero() == 0
    torch.testing.assert_close(result[:1], peak_normalize_windows(values[:1]))
