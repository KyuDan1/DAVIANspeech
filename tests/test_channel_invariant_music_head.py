import numpy as np
import pandas as pd
import pytest
import torch

from scripts.train_channel_invariant_music_head import balanced_sampling_weights
from src.channel_invariant_music_head import (
    ChannelInvariantMusicHead,
    asymmetric_bernoulli_consistency,
    dedicated_music_loss,
    pairwise_music_rank_loss,
)


def test_dedicated_head_shapes_and_music_gradients():
    model = ChannelInvariantMusicHead(width=16, heads=4, dropout=0)
    eat = torch.randn(2, 3, 4, 768)
    spear = torch.randn(2, 3, 13, 4, 1280)
    mask = torch.ones(2, 3, dtype=torch.bool)
    music, auxiliary = model(eat, spear, mask, mask)
    assert music.shape == (2,)
    assert auxiliary.shape == (2, 2)
    music.sum().backward()
    assert model.music_pool.query.grad is not None


def test_rank_loss_prefers_ordered_fake_scores():
    labels = torch.tensor([0.0, 0.0, 1.0, 1.0])
    ordered = pairwise_music_rank_loss(
        torch.tensor([-2.0, -1.0, 1.0, 2.0]), labels,
    )
    reversed_loss = pairwise_music_rank_loss(
        torch.tensor([2.0, 1.0, -1.0, -2.0]), labels,
    )
    assert ordered < reversed_loss


def test_asymmetric_consistency_is_zero_at_teacher_and_masked():
    teacher = torch.tensor([-2.0, 0.5, 3.0])
    exact = asymmetric_bernoulli_consistency(
        teacher.clone(), teacher, torch.tensor([1.0, 1.0, 0.0]),
    )
    wrong = asymmetric_bernoulli_consistency(
        -teacher, teacher, torch.tensor([1.0, 1.0, 0.0]),
    )
    assert abs(float(exact)) < 1e-6
    assert wrong > exact


def test_asymmetric_consistency_stays_finite_for_confident_bfloat16_teacher():
    codec = torch.tensor([20.0, -20.0], dtype=torch.bfloat16)
    teacher = torch.tensor([20.0, -20.0], dtype=torch.bfloat16)
    loss = asymmetric_bernoulli_consistency(codec, teacher, torch.ones(2))
    assert torch.isfinite(loss)


def test_auxiliary_tasks_are_low_weight_stabilizers():
    music = torch.tensor([-1.0, 1.0], requires_grad=True)
    auxiliary = torch.zeros(2, 2, requires_grad=True)
    total, pieces = dedicated_music_loss(
        music, auxiliary, torch.tensor([0.0, 1.0]),
        torch.tensor([0.0, 1.0]), torch.tensor([0.0, 1.0]),
        torch.ones(2), auxiliary_weight=.05,
    )
    assert total > pieces["music_bce"]
    total.backward()
    assert music.grad is not None and auxiliary.grad is not None


def test_sampling_equalizes_source_repeats_within_hierarchy():
    metadata = pd.DataFrame({
        "BANK": ["a"] * 5 + ["b"] * 2,
        "ID": [str(index) for index in range(7)],
        "MUSIC_FAKE": [0, 0, 0, 1, 1, 0, 1],
        "MUSIC_GENERATOR": ["real", "real", "real", "g", "g", "real", "g"],
        "MUSIC_SOURCE_ID": ["repeat", "repeat", "single", "f1", "f2", "r", "f"],
    })
    weights = balanced_sampling_weights(metadata)
    assert np.isfinite(weights).all() and np.all(weights > 0)
    weighted = metadata.assign(W=weights)
    bank_mass = weighted.groupby("BANK").W.sum()
    assert bank_mass.a == pytest.approx(bank_mass.b)
    source_mass = weighted[weighted.BANK.eq("a") & weighted.MUSIC_FAKE.eq(0)].groupby(
        "MUSIC_SOURCE_ID"
    ).W.sum()
    assert source_mass["repeat"] == pytest.approx(source_mass["single"])
