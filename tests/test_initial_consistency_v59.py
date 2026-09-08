import torch
import pytest

from src.three_stream_anchor_residual import initial_model_consistency_loss


def test_only_present_music_changes_and_teacher_is_frozen():
    logits = torch.ones(2, 3, requires_grad=True)
    teacher = torch.zeros(2, 3, requires_grad=True)
    presence = torch.tensor([[1., 0.], [0., 1.]])
    loss = initial_model_consistency_loss(logits, teacher, presence, [0, 1, 0])
    assert loss.item() == .5
    loss.backward()
    assert teacher.grad is None
    torch.testing.assert_close(logits.grad, torch.tensor([[0., 0., 0.], [0., 1., 0.]]))


def test_identical_models_have_zero_retention_loss():
    logits = torch.randn(4, 3, requires_grad=True)
    loss = initial_model_consistency_loss(logits, logits.detach(), torch.ones(4, 2), [.2, .3, .5])
    assert loss.item() == 0
    loss.backward()
    assert logits.grad.count_nonzero() == 0


def test_no_music_is_finite_zero():
    logits = torch.ones(2, 3, requires_grad=True)
    loss = initial_model_consistency_loss(logits, torch.zeros_like(logits), torch.zeros(2, 2), [0, 1, 0])
    loss.backward()
    assert loss.item() == 0
    assert torch.isfinite(logits.grad).all()


@pytest.mark.parametrize('weights', [[0, 0, 0], [-1, 1, 1], [float('nan'), 1, 0]])
def test_bad_weights_fail(weights):
    with pytest.raises(ValueError):
        initial_model_consistency_loss(torch.zeros(1, 3), torch.zeros(1, 3), torch.ones(1, 2), weights)
