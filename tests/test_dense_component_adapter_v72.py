import torch
from torch import nn

from src.dense_component_adapter_v72 import adapted_dense_logits
from src.dense_component_v71 import DenseComponentHead, dense_component_loss, paired_dense_logits


class TinyRepresentation(nn.Module):
    def __init__(self):
        super().__init__()
        self.base = nn.Linear(4, 4).requires_grad_(False)
        self.adapter = nn.Linear(4, 4)

    def forward(self, windows, lengths):
        values = self.base(windows[:, :4])
        adapted = self.adapter(values)
        return {'adapted': adapted[:, None, None].expand(-1, 3, 512, -1)}, torch.ones(len(windows), 512, dtype=torch.bool)


def test_dense_loss_reaches_adapter_but_not_frozen_base():
    torch.manual_seed(72)
    encoder = TinyRepresentation()
    head = DenseComponentHead(dimension=4, width=3)
    windows, lengths = torch.randn(3, 8), torch.tensor([8, 8, 8])
    outputs, dense, valid = adapted_dense_logits(encoder, head, windows, lengths, [1, 2], 2)
    target = torch.zeros_like(dense)
    target[:, 10:20, :3] = 1
    loss = outputs.square().mean() + dense_component_loss(dense, target, valid[..., None].expand_as(target))
    loss.backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in encoder.adapter.parameters())
    assert all(p.grad is None for p in encoder.base.parameters())
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in head.parameters())


def test_initial_forward_matches_frozen_v71_path():
    torch.manual_seed(72)
    encoder, head = TinyRepresentation(), DenseComponentHead(dimension=4, width=3)
    windows, lengths = torch.randn(3, 8), torch.tensor([8, 8, 8])
    reference, _, _ = paired_dense_logits(encoder, {'dense_supervised': head}, windows, lengths, [1, 2], 2)
    outputs, _, _ = adapted_dense_logits(encoder, head, windows, lengths, [1, 2], 2)
    torch.testing.assert_close(reference['dense_supervised'], outputs, rtol=0, atol=0)
