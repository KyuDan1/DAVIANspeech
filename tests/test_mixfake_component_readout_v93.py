import math

import torch

from src.mixfake_component_readout_v93 import ComponentReadoutV93, lme


def test_readout_starts_as_exact_base_lme_and_backpropagates():
    torch.manual_seed(3)
    model = ComponentReadoutV93(embedding_dim=8, width=12, dropout=0.)
    embeddings = torch.randn(5, 3, 8)
    base = torch.randn(5, 3, 3)
    output, residual = model(embeddings, base)
    torch.testing.assert_close(residual, torch.zeros_like(residual))
    torch.testing.assert_close(output, lme(base, 5.0))
    output.square().mean().backward()
    assert model.heads[0][-1].weight.grad is not None


def test_lme_is_view_count_normalized():
    value = torch.tensor([[[1., -2., 3.]]])
    repeated = value.repeat(1, 7, 1)
    torch.testing.assert_close(lme(value), lme(repeated))
    torch.testing.assert_close(lme(value), value[:, 0])


def test_normalization_validation_and_shapes():
    model = ComponentReadoutV93(embedding_dim=8, width=12)
    model.set_normalization(torch.arange(8.), torch.ones(8))
    with torch.no_grad():
        output, residual = model(torch.randn(2, 4, 8), torch.randn(2, 4, 3))
    assert output.shape == residual.shape == (2, 3)
    assert math.isfinite(float(output.sum()))
