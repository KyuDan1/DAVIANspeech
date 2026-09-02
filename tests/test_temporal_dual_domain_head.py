import torch

from src.temporal_dual_domain_head import TemporalDualDomainHead, temporal_multitask_loss


def test_temporal_head_shapes_loss_and_backward():
    model = TemporalDualDomainHead(width=32, heads=4)
    eat = torch.randn(4, 3, 4, 768)
    spear = torch.randn(4, 3, 13, 4, 1280)
    mask = torch.tensor(
        [[1, 1, 1], [1, 0, 0], [1, 1, 0], [1, 1, 1]], dtype=torch.bool
    )
    outputs = model(eat, spear, mask, mask)
    task, joint, eat_view, spear_view = outputs
    assert task.shape == (4, 3)
    assert joint.shape == (4, 4)
    assert eat_view.shape == spear_view.shape == (4, 3, 3)
    targets = torch.tensor(
        [[0, 0, 0], [0, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=torch.float32
    )
    view_targets = targets[:, None].expand(-1, 3, -1).clone()
    loss = temporal_multitask_loss(
        *outputs, targets, torch.tensor([0, 1, 2, 3]),
        view_targets, view_targets, mask, mask,
    )
    assert torch.isfinite(loss)
    loss.backward()


def test_masked_lme_ignores_padding_and_tracks_high_instance():
    logits = torch.tensor([[[0.0, 0.0, 0.0], [4.0, 4.0, 4.0], [99.0, 99.0, 99.0]]])
    mask = torch.tensor([[True, True, False]])
    pooled = TemporalDualDomainHead._masked_lme(logits, mask, torch.ones(3) * 5)
    assert torch.all(pooled > 3.0)
    assert torch.all(pooled < 4.1)
