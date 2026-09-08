import numpy as np
import pandas as pd
import pytest
import torch

from src.common_encoder_probe import CommonTokenHead
from src.presence_repair_v76 import head_statistics, presence_loss, training_review_mask


@pytest.mark.parametrize('pooling', ['mean', 'attention'])
def test_statistics_reconstruct_native_head_exactly(pooling):
    torch.manual_seed(10)
    head = CommonTokenHead(8, width=4, pooling=pooling).eval()
    tokens = torch.randn(2, 3, 6, 8)
    mask = torch.tensor([[True] * 6, [True] * 3 + [False] * 3])
    statistics = head_statistics(head, tokens, mask)
    rebuilt = (statistics * head.output_weight).sum(-1) + head.output_bias
    torch.testing.assert_close(rebuilt, head(tokens, mask), rtol=0, atol=0)


def test_mask_only_explicit_negative_training_rows_without_relabeling():
    train = pd.DataFrame(dict(DATASET=['a'] * 3, ID=['x', 'y', 'z'], VOICE_PRESENT=[0, 0, 1]))
    review = pd.DataFrame(dict(DATASET=['a'] * 2, ID=['x', 'y'], VOICE_PRESENT=[0, 0],
        EAT_VOICE=[.8, .1], PANNS_VOICE=[.9, .9], BOTH_VOICE_REVIEW=[True, False]))
    original = train.copy(deep=True)
    assert training_review_mask(train, review).tolist() == [True, False, False]
    pd.testing.assert_frame_equal(train, original)
    review.loc[0, 'ID'] = 'heldout'
    with pytest.raises(ValueError, match='non-TRAIN'):
        training_review_mask(train, review)


def test_reject_positive_or_unfaithful_flag():
    train = pd.DataFrame(dict(DATASET=['a'], ID=['x'], VOICE_PRESENT=[1]))
    review = pd.DataFrame(dict(DATASET=['a'], ID=['x'], VOICE_PRESENT=[0],
        EAT_VOICE=[.8], PANNS_VOICE=[.9], BOTH_VOICE_REVIEW=[True]))
    with pytest.raises(ValueError, match='negative'):
        training_review_mask(train, review)
    train.VOICE_PRESENT = 0
    review.BOTH_VOICE_REVIEW = False
    with pytest.raises(ValueError, match='threshold'):
        training_review_mask(train, review)


def test_masked_gradient_is_zero_and_all_masked_is_finite():
    logits = torch.tensor([.1, .2], requires_grad=True)
    labels = torch.tensor([0., 1.])
    loss = presence_loss(logits, labels, torch.tensor([True, False]))
    loss.backward()
    assert logits.grad[0] == 0 and logits.grad[1] != 0
    logits.grad.zero_()
    loss = presence_loss(logits, labels, torch.tensor([True, True]))
    loss.backward()
    assert loss == 0 and torch.equal(logits.grad, torch.zeros(2))
