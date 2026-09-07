import torch

from src.attention_expert_router import BoundedAttentionRouter


def test_bounded_attention_router_shapes_and_bounds():
    router = BoundedAttentionRouter(
        experts=4, tasks=3, expert_width=8, model_width=16,
        heads=4, strength=0.5,
    ).eval()
    hidden = torch.randn(5, 4, 3, 8)
    probability = torch.rand(5, 4, 3)
    logits, weights = router(hidden, probability)
    assert logits.shape == (5, 3)
    assert weights.shape == (5, 3, 4)
    assert torch.allclose(weights.sum(-1), torch.ones(5, 3), atol=1e-6)
    assert weights.min() >= 0.125 - 1e-6
    assert weights.max() <= 0.625 + 1e-6


def test_uniform_router_matches_logit_of_mean_probability():
    router = BoundedAttentionRouter(
        experts=4, tasks=3, expert_width=8, model_width=16,
        heads=4, strength=0.0,
    ).eval()
    hidden = torch.randn(2, 4, 3, 8)
    probability = torch.rand(2, 4, 3).clamp(1e-4, 1 - 1e-4)
    logits, weights = router(hidden, probability)
    expected = torch.logit(probability.mean(dim=1))
    assert torch.allclose(logits, expected, atol=1e-6)
    assert torch.allclose(weights, torch.full_like(weights, 0.25), atol=1e-6)
