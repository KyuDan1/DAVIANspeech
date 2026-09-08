from pathlib import Path
import shutil
import numpy as np
import pytest
import torch
from src.full_coverage_wpt import coverage_windows, codec_pair, masked_lme, forward_bags, resolve_ffmpeg


@pytest.mark.parametrize('length', [1, 64599, 64600, 64601, 160000, 960000])
def test_every_original_sample_is_covered(length):
    windows, starts = coverage_windows(np.ones(length, dtype=np.float32))
    covered = np.zeros(length, dtype=bool)
    for start in starts:
        covered[start:min(start + windows.shape[1], length)] = True
    assert covered.all()
    assert np.isfinite(windows).all()
    if length == 960000:
        assert len(windows) == 15


def test_padding_and_other_files_do_not_affect_pooling():
    values = torch.tensor([[[1., 2.], [3., 4.], [999., 999.]], [[5., 6.], [999., 999.], [999., 999.]]])
    mask = torch.tensor([[1, 1, 0], [1, 0, 0]], dtype=torch.bool)
    result = masked_lme(values, mask)
    torch.testing.assert_close(result[1], torch.tensor([5., 6.]))
    torch.testing.assert_close(result[:1], masked_lme(values[:1, :2], mask[:1, :2]))


def test_forward_excludes_padding_and_preserves_gradient():
    class FakeModel:
        def forward_windows(self, x):
            assert x.abs().max() < 10
            return x.mean(dim=-1, keepdim=True).expand(-1, -1, 3)
    values = torch.ones(2, 2, 100, requires_grad=True)
    mask = torch.tensor([[1, 1], [1, 0]], dtype=torch.bool)
    with torch.no_grad():
        values[1, 1] = 999
    result = forward_bags(FakeModel(), values, mask)
    result.sum().backward()
    assert torch.isfinite(values.grad).all()
    assert values.grad[1, 1].count_nonzero() == 0


@pytest.mark.parametrize('variant', ['g711_ulaw', 'g722_wb', 'opus_nb_8k'])
def test_actual_codec_keeps_pair_shape_and_is_deterministic(variant):
    try:
        executable = resolve_ffmpeg()
    except FileNotFoundError:
        pytest.skip('ffmpeg unavailable')
    audio = .1 * np.sin(np.arange(64600, dtype=np.float32) * .05)
    windows, _ = coverage_windows(audio)
    first = codec_pair(windows, variant, Path(executable), 42)
    second = codec_pair(windows, variant, Path(executable), 42)
    assert first.shape == windows.shape
    np.testing.assert_array_equal(first, second)
    assert np.isfinite(first).all()
