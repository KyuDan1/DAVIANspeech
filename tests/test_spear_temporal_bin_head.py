import torch

from src.spear_temporal_bin_head import SpearTemporalBinHead, temporal_bin_loss


def test_temporal_bin_head_shapes_and_loss_are_finite():
    model = SpearTemporalBinHead(12, torch.zeros(12), torch.ones(12))
    features = torch.randn(4, 3, 5, 12)
    mask = torch.ones(4, 3, 5, dtype=torch.bool)
    logits, probability = model(features, mask)
    assert logits.shape == (4, 15, 2)
    assert probability.shape == (4,)
    assert torch.all((probability > 0) & (probability < 1))
    presence = torch.ones(4, 15)
    fake = torch.tensor([[0.0] * 15, [1.0] * 15] * 2)
    loss, terms = temporal_bin_loss(
        logits, probability, mask.reshape(4, -1), presence, fake,
        torch.tensor([0.0, 1.0, 0.0, 1.0]), torch.ones(4), torch.ones(4),
    )
    assert torch.isfinite(loss)
    assert set(terms) == {"file", "local_fake", "presence", "consistency"}


def test_presence_weights_suppress_a_speech_only_false_alarm():
    model = SpearTemporalBinHead(1, torch.zeros(1), torch.ones(1), temperature=5)
    logits = torch.tensor([[[-8.0, 8.0], [8.0, -2.0]]])
    mask = torch.ones(1, 2, dtype=torch.bool)
    weighted = model.aggregate(logits, mask)
    logits[..., 0] = 8.0
    unweighted = model.aggregate(logits, mask)
    assert weighted < unweighted
