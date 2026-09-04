import pytest
import torch
from torch import nn

from src.eat_patch_graph import (
    EatPatchGraphHead, gaussian_patch_projection,
    hierarchical_patch_graph_features, patch_graph_features, patch_graph_loss,
)
from src.eat_hierarchical import hierarchical_statistics


class _Block(nn.Module):
    def forward(self, values):
        return values + .01, values


class _Patch(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Conv2d(1, 8, kernel_size=2, stride=2)

    def forward(self, values):
        return self.proj(values).flatten(2).transpose(1, 2)


class _Core(nn.Module):
    def __init__(self):
        super().__init__()
        self.local_encoder = _Patch()
        self.extra_tokens = nn.Parameter(torch.zeros(1, 1, 8))
        self.pre_norm = nn.LayerNorm(8)
        self.pos_drop = nn.Identity()
        self.blocks = nn.ModuleList([_Block(), _Block(), _Block()])


def test_patch_graph_features_preserve_axes():
    model = _Core()
    features = torch.randn(2, 1, 8, 6)
    projection = gaussian_patch_projection(8, 4, seed=7)
    temporal, spectral = patch_graph_features(
        model, features, projection, layers=(0, 2)
    )
    assert temporal.shape == (2, 2, 2, 4, 4)
    assert spectral.shape == (2, 2, 2, 3, 4)
    assert torch.isfinite(temporal).all()
    assert torch.isfinite(spectral).all()


def test_joint_extraction_preserves_hierarchical_anchor():
    model = _Core()
    features = torch.randn(2, 1, 8, 6)
    projection = gaussian_patch_projection(8, 4, seed=9)
    expected = hierarchical_statistics(model, features)
    actual, _, _ = hierarchical_patch_graph_features(
        model, features, projection, layers=(0, 2)
    )
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_patch_graph_head_and_conditional_loss():
    model = EatPatchGraphHead(
        layers=2, dimension=4, width=8, heads=2, depth=1,
        maximum_views=3, maximum_time_nodes=4,
        maximum_frequency_nodes=3, dropout=0,
    )
    temporal = torch.randn(5, 3, 2, 2, 4, 4)
    spectral = torch.randn(5, 3, 2, 2, 3, 4)
    mask = torch.tensor([
        [1, 1, 1], [1, 1, 0], [1, 0, 0], [1, 1, 0], [1, 1, 1],
    ], dtype=torch.bool)
    logits = model(temporal, spectral, mask)
    assert logits.shape == (5, 3)
    probability = model.probabilities(logits)
    assert torch.all((probability >= 0) & (probability <= 1))
    targets = torch.tensor([
        [0, 0, 0], [1, 0, 1], [0, 1, 1], [1, 1, 1], [0, 0, 0],
    ], dtype=torch.float32)
    presence = torch.tensor([
        [1, 1], [1, 0], [0, 1], [1, 1], [1, 1],
    ], dtype=torch.float32)
    loss, terms = patch_graph_loss(
        model, logits, targets, presence, torch.ones(5)
    )
    assert torch.isfinite(loss)
    assert set(terms) == {"bce", "ranking", "voice", "music", "file"}
    loss.backward()


def test_patch_graph_head_rejects_empty_view_mask():
    model = EatPatchGraphHead(
        layers=1, dimension=4, width=8, heads=2, depth=1,
        maximum_views=1, maximum_time_nodes=2,
        maximum_frequency_nodes=2,
    )
    temporal = torch.randn(1, 1, 1, 2, 2, 4)
    spectral = torch.randn(1, 1, 1, 2, 2, 4)
    with pytest.raises(ValueError, match="valid view"):
        model(temporal, spectral, torch.zeros(1, 1, dtype=torch.bool))
