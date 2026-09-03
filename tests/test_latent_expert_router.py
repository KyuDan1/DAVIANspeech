import torch

from src.latent_expert_router import (
    BoundedLatentExpertRouter, pairwise_ranking_loss,
)


def make_router(strength=.25):
    prior = torch.tensor([[.7, .3], [.8, .2], [.6, .4]])
    return BoundedLatentExpertRouter(12, prior, dropout=0, strength=strength)


def test_router_starts_at_validated_fixed_moe():
    model = make_router().eval()
    latent = torch.randn(5, 12)
    logits = torch.randn(5, 2, 3)
    fused, weights = model(latent, logits)
    expected = (logits.permute(0, 2, 1) * model.prior).sum(-1)
    torch.testing.assert_close(weights, model.prior.expand_as(weights))
    torch.testing.assert_close(fused, expected)


def test_router_weights_are_bounded_and_differentiable():
    model = make_router(strength=.2).train()
    with torch.no_grad():
        model.score[-1].weight.normal_()
    latent = torch.randn(7, 12, requires_grad=True)
    logits = torch.randn(7, 2, 3)
    fused, weights = model(latent, logits)
    assert torch.all(weights >= .8 * model.prior[None] - 1e-7)
    torch.testing.assert_close(weights.sum(-1), torch.ones(7, 3))
    fused.sum().backward()
    assert latent.grad is not None and torch.isfinite(latent.grad).all()


def test_router_accepts_separate_expert_and_global_latents():
    prior = torch.tensor([[.7, .3], [.8, .2], [.6, .4]])
    model = BoundedLatentExpertRouter(
        15, prior, dropout=0, strength=.25,
        expert_latent_widths=(5, 7), global_latent_width=3,
    ).eval()
    latent = torch.randn(4, 15, requires_grad=True)
    logits = torch.randn(4, 2, 3)
    fused, weights = model(latent, logits)
    torch.testing.assert_close(weights, prior.expand_as(weights))
    fused.sum().backward()
    assert latent.grad is not None and torch.isfinite(latent.grad).all()


def test_pairwise_loss_rewards_correct_ordering_and_masks_absence():
    targets = torch.tensor([[0., 0.], [1., 0.], [0., 1.], [1., 1.]])
    masks = torch.tensor([[1., 0.], [1., 0.], [0., 1.], [0., 1.]])
    good = torch.tensor([[-2., 0.], [2., 0.], [0., -2.], [0., 2.]])
    bad = -good
    assert pairwise_ranking_loss(good, targets, masks) < pairwise_ranking_loss(
        bad, targets, masks
    )
