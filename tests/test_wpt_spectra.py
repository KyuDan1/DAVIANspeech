import torch
from torch import nn

from src.wpt_spectra import (
    DeepWaveletPrompts, WPTSpectraMultitask, component_loss, haar_2d_tokens,
    component_ranking_loss,
)


def test_haar_prompts_preserve_shape_energy_and_gradient():
    values = torch.randn(3, 4, 8, requires_grad=True)
    transformed = haar_2d_tokens(values)
    assert transformed.shape == values.shape
    torch.testing.assert_close(
        transformed.square().sum(), values.square().sum(), rtol=1e-5, atol=1e-5
    )
    transformed.sum().backward()
    assert values.grad is not None and torch.isfinite(values.grad).all()


def test_deep_prompts_are_layer_specific_and_differentiable():
    prompts = DeepWaveletPrompts(3, 8, prompt_tokens=2, wavelet_tokens=4)
    first = prompts(0, 5)
    second = prompts(1, 5)
    assert first.shape == second.shape == (5, 6, 8)
    assert not torch.equal(first, second)
    first.sum().backward()
    assert prompts.standard.grad is not None
    assert prompts.wavelet_initial.grad is not None


def test_component_loss_masks_absent_components():
    logits = torch.randn(4, 3, requires_grad=True)
    targets = torch.tensor([
        [0., 0., 0.], [1., 0., 1.], [0., 1., 1.], [1., 1., 1.]
    ])
    presence = torch.tensor([
        [1., 0.], [1., 0.], [0., 1.], [1., 1.]
    ])
    loss, terms = component_loss(logits, targets, presence, torch.ones(4))
    assert set(terms) == {"voice", "music", "file"}
    assert torch.isfinite(loss)
    loss.backward()
    # Row 0 has no Music, row 2 has no Voice.
    assert logits.grad[0, 1] == 0
    assert logits.grad[2, 0] == 0


def test_logmeanexp_view_aggregation_is_length_normalized():
    # Repeating the same view must not change a file score.  This is important
    # because evaluation clips have different durations and view counts.
    instance = object.__new__(WPTSpectraMultitask)
    nn.Module.__init__(instance)
    instance.temperature = 5.0
    one = torch.tensor([[[1.2, -0.3, 0.7]]])
    repeated = one.repeat(1, 4, 1)
    torch.testing.assert_close(instance.aggregate(one), instance.aggregate(repeated))


def test_component_ranking_loss_prefers_correct_ordering():
    target = torch.tensor([[0., 0., 0.], [1., 1., 1.]])
    presence = torch.ones(2, 2)
    good = torch.tensor([[-2., -2., -2.], [2., 2., 2.]])
    bad = -good
    assert component_ranking_loss(good, target, presence) < component_ranking_loss(
        bad, target, presence
    )
