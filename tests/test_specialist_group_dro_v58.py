import torch

from src.three_stream_anchor_residual import group_dro_loss


def test_music_specialist_skips_speech_only_environment():
    logits = torch.zeros(4, 3, requires_grad=True)
    targets = torch.tensor([[0., 0., 0.], [1., 0., 1.],
                            [0., 0., 0.], [0., 1., 1.]])
    presence = torch.tensor([[1., 0.], [1., 0.], [0., 1.], [0., 1.]])
    weights = torch.tensor([0., 1., 0.])
    loss = group_dro_loss(logits, targets, presence, torch.ones(4),
                          torch.tensor([0, 0, 1, 1]), weights, .1)
    torch.testing.assert_close(loss, torch.tensor(2.).log())
    loss.backward()
    assert torch.isfinite(logits.grad).all()
    assert torch.count_nonzero(logits.grad[:2]) == 0
    assert torch.count_nonzero(logits.grad[:, [0, 2]]) == 0


def test_music_specialist_no_music_returns_differentiable_zero():
    logits = torch.zeros(2, 3, requires_grad=True)
    loss = group_dro_loss(logits, torch.zeros(2, 3),
                          torch.tensor([[1., 0.], [1., 0.]]),
                          torch.ones(2), torch.zeros(2, dtype=torch.long),
                          torch.tensor([0., 1., 0.]), .1)
    assert loss.item() == 0
    loss.backward()
    assert torch.isfinite(logits.grad).all()
