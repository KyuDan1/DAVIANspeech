import pytest
import torch

from src.unified_dual_ssl_head import UnifiedDualSSLHead, unified_multitask_loss


def inputs(batch=3):
    eat = torch.randn(batch, 3, 12, 5, 128)
    spear = torch.randn(batch, 3, 8, 4, 4, 64)
    eat_mask = torch.tensor([[1, 1, 1], [1, 0, 0], [1, 1, 0]], dtype=torch.bool)[:batch]
    spear_mask = eat_mask[:, :, None].expand(-1, -1, 8).clone()
    return eat, spear, eat_mask, spear_mask


def test_unified_head_shapes_probabilities_and_gradients():
    model = UnifiedDualSSLHead(width=32, heads=4, dropout=0)
    logits, presence_logits, joint_logits = model(*inputs())
    assert logits.shape == (3, 3)
    assert presence_logits.shape == (3, 2)
    assert joint_logits.shape == (3, 4)
    probability = model.probabilities(logits)
    assert probability.shape == (3, 3)
    assert torch.all((probability > 0) & (probability < 1))

    fake = torch.tensor([[0, 0, 0], [1, 0, 1], [0, 1, 1]], dtype=torch.float32)
    present = torch.tensor([[1, 0], [1, 1], [0, 1]], dtype=torch.float32)
    loss, terms = unified_multitask_loss(
        model, logits, presence_logits, joint_logits, fake, present, torch.ones(3)
    )
    assert torch.isfinite(loss)
    assert set(terms) == {
        "component", "voice", "music", "file", "presence", "joint",
        "consistency",
    }
    loss.backward()
    assert model.eat_layer_logits.grad is not None
    assert model.spear_layer_logits.grad is not None


def test_unified_head_rejects_incompatible_masks():
    model = UnifiedDualSSLHead(width=32, heads=4)
    eat, spear, eat_mask, spear_mask = inputs(batch=2)
    with pytest.raises(ValueError, match="SPEAR mask"):
        model(eat, spear, eat_mask, spear_mask[:, :, :-1])
