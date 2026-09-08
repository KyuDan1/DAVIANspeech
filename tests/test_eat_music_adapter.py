import torch
import pytest
from torch import nn

from src.eat_music_adapter import EatMusicAdapter, pairwise_ranking_loss
from scripts.score_eat_music_adapter_v55 import score_dataset


class TupleBlock(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.linear = nn.Linear(width, width)

    def forward(self, values):
        result = self.linear(values)
        return result, result


class LocalEncoder(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.proj = nn.Conv2d(1, width, kernel_size=2, stride=2)

    def forward(self, features):
        return self.proj(features).flatten(2).transpose(1, 2)


class FakeEat(nn.Module):
    def __init__(self, width=8, depth=4):
        super().__init__()
        self.model = nn.Module()
        self.model.local_encoder = LocalEncoder(width)
        self.model.extra_tokens = nn.Parameter(torch.zeros(1, 1, width))
        self.model.pre_norm = nn.LayerNorm(width)
        self.model.pos_drop = nn.Identity()
        self.model.fixed_positional_encoder = None
        self.model.blocks = nn.ModuleList([TupleBlock(width) for _ in range(depth)])


def test_adapter_only_updates_small_residual_and_head():
    base = FakeEat()
    model = EatMusicAdapter(
        base, adapter_blocks=(2, 3), bottleneck=3, pooling_heads=2, hidden=6
    )
    assert all(not parameter.requires_grad for parameter in base.parameters())
    trainable = [name for name, value in model.named_parameters() if value.requires_grad]
    assert trainable
    assert all(not name.startswith("eat.") for name in trainable)

    features = torch.randn(3, 1, 8, 8)
    target = torch.tensor([0.0, 1.0, 1.0])
    loss = torch.nn.functional.binary_cross_entropy_with_logits(model(features), target)
    loss.backward()
    assert model.adapters["2"].up.weight.grad is not None
    assert model.adapters["2"].up.weight.grad.abs().sum() > 0
    assert model.adapters["3"].up.weight.grad is not None
    assert all(parameter.grad is None for parameter in base.parameters())


def test_multiview_mask_and_ranking_loss_are_finite():
    model = EatMusicAdapter(
        FakeEat(), adapter_blocks=(2, 3), bottleneck=3, pooling_heads=2, hidden=6
    )
    features = torch.randn(2, 3, 1, 8, 8)
    mask = torch.tensor([[True, True, False], [True, False, False]])
    logits = model(features, mask)
    assert logits.shape == (2,)
    rank = pairwise_ranking_loss(logits, torch.tensor([0.0, 1.0]))
    assert torch.isfinite(rank)


def test_actual_last_block_indices_receive_gradient_with_padding():
    model = EatMusicAdapter(
        FakeEat(depth=12), adapter_blocks=(10, 11), bottleneck=3,
        pooling_heads=2, hidden=6, pooling_width=4,
    )
    features = torch.randn(2, 2, 1, 8, 8)
    mask = torch.tensor([[True, False], [True, True]])
    model(features, mask).sum().backward()
    assert model.adapters["10"].up.weight.grad.abs().sum() > 0


def test_v55_scorer_refuses_reserved_blind_v8():
    with pytest.raises(ValueError, match="blind v8"):
        score_dataset(
            None, "codec_mixed_blind_v8", None, None, None, 1, .15
        )
