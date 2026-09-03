import torch

from src.dual_domain_head import DualDomainHead, multitask_loss


def test_dual_domain_head_shapes_and_finite_loss():
    model = DualDomainHead(width=32, heads=4)
    eat = torch.randn(3, 3, 4, 768)
    spear = torch.randn(3, 3, 13, 4, 1280)
    mask = torch.tensor([[1, 1, 1], [1, 0, 0], [1, 1, 0]], dtype=torch.bool)
    task_logits, joint_logits = model(eat, spear, mask, mask)
    assert task_logits.shape == (3, 3)
    assert joint_logits.shape == (3, 4)
    probabilities = model.probabilities(task_logits, joint_logits)
    assert torch.all((probabilities >= 0) & (probabilities <= 1))
    targets = torch.tensor([[0, 0, 0], [0, 1, 1], [1, 1, 1]], dtype=torch.float32)
    labels = torch.tensor([0, 1, 3])
    loss = multitask_loss(task_logits, joint_logits, targets, labels)
    assert torch.isfinite(loss)
    loss.backward()


def test_stream_dropout_never_removes_every_token():
    model = DualDomainHead(width=32, heads=4, stream_dropout=1.0).train()
    mask = torch.ones(8, 3, dtype=torch.bool)
    eat_mask, spear_mask = model._stream_masks(mask, mask)
    assert torch.all(eat_mask.any(1) | spear_mask.any(1))
