import torch
from torch import nn

from src.eat_hierarchical import gaussian_projection, hierarchical_statistics
from src.hierarchical_eat_music import HierarchicalEatMusicHead


class _Position(nn.Module):
    def forward(self, values, mask):
        del mask
        return torch.zeros_like(values)


class _Block(nn.Module):
    def forward(self, values):
        return values + 0.1, values


class _Patch(nn.Module):
    def __init__(self):
        super().__init__()
        self.projection = nn.Conv2d(1, 8, kernel_size=2, stride=2)

    def forward(self, values):
        return self.projection(values).flatten(2).transpose(1, 2)


class _Eat(nn.Module):
    def __init__(self):
        super().__init__()
        self.local_encoder = _Patch()
        self.extra_tokens = nn.Parameter(torch.zeros(1, 1, 8))
        self.fixed_positional_encoder = _Position()
        self.pre_norm = nn.LayerNorm(8)
        self.pos_drop = nn.Identity()
        self.blocks = nn.ModuleList([_Block(), _Block(), _Block()])

    def encode(self, values):
        values = self.local_encoder(values)
        values = values + self.fixed_positional_encoder(values, None)
        values = torch.cat((self.extra_tokens.expand(len(values), -1, -1), values), 1)
        values = self.pos_drop(self.pre_norm(values))
        for block in self.blocks:
            values, _ = block(values)
        return values


def test_hierarchical_statistics_shape_and_determinism():
    model = _Eat()
    features = torch.randn(2, 1, 8, 8)
    projection = gaussian_projection(8, 4, seed=7)
    first = hierarchical_statistics(model, features, projection)
    second = hierarchical_statistics(model, features, projection)
    assert first.shape == (2, 3, 5, 4)
    assert torch.equal(first, second)
    assert torch.equal(projection, gaussian_projection(8, 4, seed=7))


def test_hierarchical_music_head_masks_padded_views_and_is_finite():
    mean = torch.zeros(3, 5, 8)
    std = torch.ones_like(mean)
    model = HierarchicalEatMusicHead(
        mean, std, max_views=3, heads=2, pool_heads=2,
        dropout=0, dsu_probability=0,
    ).eval()
    values = torch.randn(2, 3, 3, 5, 8)
    mask = torch.tensor([[True, False, False], [True, True, True]])
    first = model(values, mask)
    values[0, 1:] = 1_000
    second = model(values, mask)
    assert first.shape == (2,)
    assert torch.isfinite(first).all()
    assert torch.allclose(first[0], second[0], atol=1e-6)


def test_dsu_changes_training_values_but_not_eval_predictions():
    mean = torch.zeros(2, 5, 8)
    std = torch.ones_like(mean)
    model = HierarchicalEatMusicHead(
        mean, std, max_views=2, heads=2, pool_heads=2,
        dropout=0, dsu_probability=1, dsu_scale=1,
    )
    values = torch.randn(4, 2, 2, 5, 8)
    mask = torch.ones(4, 2, dtype=torch.bool)
    model.eval()
    first = model(values, mask)
    second = model(values, mask)
    assert torch.equal(first, second)
    model.train()
    torch.manual_seed(1)
    train_first = model(values, mask)
    torch.manual_seed(2)
    train_second = model(values, mask)
    assert not torch.equal(train_first, train_second)
