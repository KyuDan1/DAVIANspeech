from types import SimpleNamespace

import pytest
import torch
from torch import nn

from src.common_eat_adaptation import PairedEatRepresentation, paired_file_logits


class TinyBase(nn.Module):
    def __init__(self):
        super().__init__()
        core = nn.Module()
        core.local_encoder = nn.Module()
        core.local_encoder.proj = nn.Conv2d(1, 8, 2, 2)
        core.extra_tokens = nn.Parameter(torch.randn(1, 1, 8))
        core.pre_norm = core.pos_drop = nn.Identity()
        core.blocks = nn.ModuleList([nn.Sequential(nn.Linear(8, 8), nn.Tanh()) for _ in range(6)])
        self.model = core


def test_zero_init_control_rng_and_adapter_only_gradients():
    torch.manual_seed(7)
    base = TinyBase()
    state = torch.random.get_rng_state().clone()
    model = PairedEatRepresentation(base, (3, 4, 5), bottleneck=4)
    assert torch.equal(state, torch.random.get_rng_state())
    model.train()
    assert not base.training and model.adapters.training
    features = torch.randn(2, 1, 4, 2)
    outputs, mask = model.from_features(features, torch.tensor([16, 32]))
    assert mask.sum(-1).tolist() == [1, 2]
    torch.testing.assert_close(outputs['control'], outputs['adapted'], rtol=0, atol=0)
    before = outputs['control'].detach().clone()
    loss = outputs['adapted'].square().mean()
    loss.backward()
    assert all(p.grad is None and not p.requires_grad for p in base.parameters())
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.adapters.parameters())
    assert any(p.grad.abs().sum() > 0 for p in model.adapters.parameters())
    optimizer = torch.optim.AdamW(model.adapters.parameters(), lr=.01)
    optimizer.step()
    after, _ = model.from_features(features, torch.tensor([16, 32]))
    torch.testing.assert_close(after['control'], before, rtol=0, atol=0)
    assert not torch.allclose(after['adapted'], before)


@pytest.mark.parametrize('blocks', [(), (2, 1), (1, 1), (-1,), (6,)])
def test_invalid_adapter_blocks_rejected(blocks):
    with pytest.raises(ValueError):
        PairedEatRepresentation(TinyBase(), blocks)


def test_paired_file_bag_aggregation_preserves_file_gradients():
    class Encoder:
        def __call__(self, windows, lengths):
            tokens = windows[:, None, None, :].repeat(1, 3, 1, 1)
            return {'control': tokens, 'adapted': tokens * 2}, torch.ones(len(windows), 1, dtype=torch.bool)

    class Head(nn.Module):
        def forward(self, tokens, mask):
            return tokens[:, 0, 0]

    windows = torch.randn(3, 5, requires_grad=True)
    result = paired_file_logits(Encoder(), {'control': Head(), 'adapted': Head()}, windows,
                                torch.ones(3), [1, 2])
    torch.testing.assert_close(result['control'][0], windows[0])
    torch.testing.assert_close(result['adapted'][0], windows[0] * 2)
    result['adapted'].sum().backward()
    assert torch.isfinite(windows.grad).all() and windows.grad.abs().sum() > 0
