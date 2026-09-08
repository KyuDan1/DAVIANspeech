import sys
from pathlib import Path

import torch
from torch.nn import functional as F


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from train_mixfake_multistream_joint_v97 import (  # noqa: E402
    ADS_WEIGHTS, weighted_task_loss,
)
from train_mixfake_multistream_prompt_v94 import task_rank  # noqa: E402


def test_joint_loss_matches_taskwise_ads_weighting():
    clean = torch.tensor([
        [-1.0, 0.2, 0.7], [0.3, -0.6, 1.1], [1.2, 0.8, -0.4],
    ], requires_grad=True)
    phone = torch.tensor([
        [-0.8, 0.1, 0.4], [0.1, -0.2, 0.9], [0.8, 1.0, -0.1],
    ], requires_grad=True)
    target = torch.tensor([
        [0.0, 1.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 0.0],
    ])
    ranking_weight, consistency_weight = .13, .17
    actual = weighted_task_loss(
        clean, phone, target, ranking_weight, consistency_weight
    )
    expected = clean.new_zeros(())
    for task, weight in enumerate(ADS_WEIGHTS):
        bce = .5 * (
            F.binary_cross_entropy_with_logits(clean[:, task], target[:, task])
            + F.binary_cross_entropy_with_logits(phone[:, task], target[:, task])
        )
        rank = .5 * (
            task_rank(clean[:, task], target[:, task])
            + task_rank(phone[:, task], target[:, task])
        )
        consistency = F.smooth_l1_loss(phone[:, task], clean[:, task].detach())
        expected = expected + float(weight) * (
            bce + ranking_weight * rank + consistency_weight * consistency
        )
    torch.testing.assert_close(actual, expected)
    actual.backward()
    assert torch.isfinite(clean.grad).all()
    assert torch.isfinite(phone.grad).all()


def test_ads_weights_match_checkpoint_task_order():
    assert ADS_WEIGHTS.tolist() == [.2, .3, .5]
