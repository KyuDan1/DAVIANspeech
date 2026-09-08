import numpy as np
import pytest
import torch

from src.mert_generator_robust_music import (
    MertGeneratorRobustMusicHead,
    asymmetric_codec_kl,
    dual_scale_features,
    logit_residual,
    make_projection,
)


def test_dual_scale_feature_shapes_and_temporal_sensitivity():
    statistics = np.zeros((2, 3, 13, 2, 768), dtype=np.float32)
    statistics[1, 2, 8:, 0, 0] = 2.0
    projection = make_projection(8, 4)
    local, long = dual_scale_features(statistics, projection)
    assert local.shape == (2, 192)
    assert long.shape == (2, 336)
    assert not torch.equal(long[0], long[1])


def test_compressed_feature_version_remains_deployable():
    statistics = np.random.default_rng(3).normal(size=(1, 3, 13, 2, 768)).astype(np.float32)
    local, long = dual_scale_features(
        statistics, make_projection(8, 4), "compressed_v1",
    )
    assert local.shape == (1, 40)
    assert long.shape == (1, 56)


@pytest.mark.parametrize("mode", ["local", "long", "dual"])
def test_head_shapes_and_gradients(mode):
    model = MertGeneratorRobustMusicHead(40, 56, hidden=8, dropout=0)
    output = model(torch.randn(3, 40), torch.randn(3, 56), mode)
    assert all(value.shape[0] == 3 for value in output)
    output[0].sum().backward()
    assert any(parameter.grad is not None for parameter in model.parameters())


def test_codec_kl_is_finite_and_prefers_agreement():
    teacher = torch.tensor([20.0, -20.0], dtype=torch.bfloat16)
    same = asymmetric_codec_kl(teacher, teacher)
    wrong = asymmetric_codec_kl(-teacher, teacher)
    assert torch.isfinite(same) and abs(float(same)) < 1e-4
    assert wrong > same


def test_logit_residual_endpoints_and_guard():
    anchor = np.array([.1, .7])
    expert = np.array([.8, .2])
    np.testing.assert_allclose(logit_residual(anchor, expert, 0), anchor, atol=1e-6)
    np.testing.assert_allclose(logit_residual(anchor, expert, 1), expert, atol=1e-6)
    with pytest.raises(ValueError):
        logit_residual(anchor, expert, 2)
