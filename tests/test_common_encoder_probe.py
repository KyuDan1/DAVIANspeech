import numpy as np
import pytest
import torch

from src.common_encoder_probe import CommonTokenHead, complete_windows, component_loss, depth_indices


def test_complete_windows_cover_tail_without_repeating_short_audio():
    short, lengths, starts = complete_windows(np.arange(4, dtype=np.float32), 6)
    assert short.tolist() == [[0, 1, 2, 3, 0, 0]]
    assert lengths.tolist() == [4]
    long, lengths, starts = complete_windows(np.arange(17, dtype=np.float32), 6)
    covered = set()
    for start, view in zip(starts, long):
        covered.update(range(start, start + len(view)))
    assert covered == set(range(17))
    assert long[-1, -1] == 16


@pytest.mark.parametrize('pooling', ['mean', 'attention'])
def test_head_padding_and_file_independence(pooling):
    torch.manual_seed(4)
    head = CommonTokenHead(12, width=8, pooling=pooling).eval()
    tokens = torch.randn(2, 3, 7, 12)
    mask = torch.tensor([[1, 1, 1, 0, 0, 0, 0], [1, 1, 1, 1, 1, 1, 1]], dtype=torch.bool)
    expected = head(tokens, mask)
    poisoned = tokens.masked_fill(~mask[:, None, :, None], float('nan'))
    torch.testing.assert_close(head(poisoned, mask), expected)
    torch.testing.assert_close(head(tokens[:1, :, :3], mask[:1, :3]), expected[:1])
    torch.testing.assert_close(head(tokens.flip(0), mask.flip(0)).flip(0), expected)


def test_pooling_conditions_share_schema_and_parameter_count():
    mean = CommonTokenHead(12, pooling='mean')
    attention = CommonTokenHead(12, pooling='attention')
    attention.load_state_dict(mean.state_dict(), strict=True)
    assert sum(p.numel() for p in mean.parameters()) == sum(p.numel() for p in attention.parameters())
    with torch.no_grad():
        attention.queries.zero_()
    x, mask = torch.randn(2, 3, 6, 12), torch.ones(2, 6, dtype=torch.bool)
    torch.testing.assert_close(mean(x, mask), attention(x, mask))


def test_absent_component_is_not_a_real_training_target():
    logits = torch.zeros(2, 5, requires_grad=True)
    targets = torch.tensor([[0, 0, float('nan'), 1, 0], [1, float('nan'), 1, 0, 1]])
    loss = component_loss(logits, targets)
    loss.backward()
    assert torch.isfinite(loss)
    assert logits.grad[0, 2] == 0 and logits.grad[1, 1] == 0
    assert logits.grad[0, 1] != 0 and logits.grad[1, 2] != 0


def test_invalid_active_label_and_empty_mask_rejected():
    with pytest.raises(ValueError):
        component_loss(torch.zeros(1, 5), torch.tensor([[float('nan'), 0, 0, 1, 1]]))
    with pytest.raises(ValueError):
        CommonTokenHead(12)(torch.zeros(1, 3, 2, 12), torch.zeros(1, 2, dtype=torch.bool))


def test_depth_indices_have_three_distinct_stages():
    assert depth_indices(48) == (11, 23, 47)
    assert depth_indices(24) == (5, 11, 23)
    assert depth_indices(13) == (3, 6, 12)
    with pytest.raises(ValueError):
        depth_indices(2)


def test_finite_head_gradients():
    head = CommonTokenHead(12, width=8)
    output = head(torch.randn(4, 3, 10, 12), torch.ones(4, 10, dtype=torch.bool))
    target = torch.tensor([[0, 0, 0, 1, 1], [1, 1, 0, 1, 1], [1, 0, 1, 1, 1], [1, 1, 1, 1, 1]]).float()
    component_loss(output, target).backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in head.parameters())
