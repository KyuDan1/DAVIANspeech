import torch
import numpy as np

from src.component_query_mhfa import ComponentQueryMHFA, component_query_loss
from src.component_query_mhfa_inference import (
    _component_or_file, _logit, _sigmoid,
)


def inputs(batch=3):
    temporal = torch.randn(batch, 3, 6, 2, 38, 16)
    spectral = torch.randn(batch, 3, 6, 2, 8, 16)
    eat_mask = torch.tensor([[1, 1, 1], [1, 0, 0], [1, 1, 0]], dtype=torch.bool)
    spear = torch.randn(batch, 3, 8, 4, 4, 8)
    spear_mask = eat_mask[:, :, None].expand(-1, -1, 8).clone()
    return temporal, spectral, eat_mask, spear, spear_mask


def test_component_query_shapes_and_gradients():
    model = ComponentQueryMHFA(
        eat_dimension=16, spear_dimension=8, width=24, heads=4, depth=1,
    )
    fake, presence, joint = model(*inputs())
    assert fake.shape == (3, 3)
    assert presence.shape == (3, 2)
    assert joint.shape == (3, 4)
    targets = torch.tensor([[0, 0, 0], [1, 0, 1], [0, 1, 1.]], dtype=torch.float)
    present = torch.tensor([[1, 1], [1, 0], [1, 1.]], dtype=torch.float)
    loss, terms = component_query_loss(
        model, fake, presence, joint, targets, present, torch.ones(3)
    )
    loss.backward()
    assert torch.isfinite(loss)
    assert set(terms) == {"voice", "music", "file", "presence", "joint", "ranking"}
    assert model.eat_projection.weight.grad is not None
    assert model.spear_projection.weight.grad is not None


def test_signed_differences_keep_direction():
    values = torch.tensor([0., 2., 1., 4.]).reshape(1, 1, 4, 1, 1, 1)
    first, second = ComponentQueryMHFA.signed_differences(values)
    assert first.flatten().tolist() == [0., 2., -1., 3.]
    assert second.flatten().tolist() == [0., 0., -3., 4.]


def test_file_probability_is_bounded():
    model = ComponentQueryMHFA(
        eat_dimension=16, spear_dimension=8, width=24, heads=4, depth=1,
    )
    probability = model.probabilities(torch.randn(9, 3))
    assert probability.shape == (9, 3)
    assert torch.all((probability >= 0) & (probability <= 1))


def test_probability_logit_round_trip():
    probability = np.asarray([.01, .2, .5, .8, .99])
    np.testing.assert_allclose(_sigmoid(_logit(probability)), probability)


def test_component_or_file_has_exact_endpoints():
    np.testing.assert_allclose(_component_or_file(.2, .8, .5, 0), .2)
    np.testing.assert_allclose(_component_or_file(.2, .8, .5, 1), .9)


def test_masked_views_do_not_change_prediction():
    model = ComponentQueryMHFA(
        eat_dimension=16, spear_dimension=8, width=24, heads=4, depth=1,
        dropout=0,
    ).eval()
    values = list(inputs())
    with torch.no_grad():
        reference = model(*values)[0]
        # Item one has only its first view enabled in both streams.
        values[0][1, 1:] = 1000
        values[1][1, 1:] = -1000
        values[3][1, 1:] = 500
        changed = model(*values)[0]
    torch.testing.assert_close(reference[1], changed[1])
