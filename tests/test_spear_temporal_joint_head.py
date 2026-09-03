import torch

from src.spear_temporal_joint_head import (
    SpearTemporalJointHead,
    joint_temporal_loss,
)


def test_joint_temporal_head_shapes_and_loss():
    model = SpearTemporalJointHead(8, torch.zeros(8), torch.ones(8), hidden=12)
    features = torch.randn(4, 3, 2, 8)
    mask = torch.ones(4, 3, 2, dtype=torch.bool)
    logits, outputs = model(features, mask)
    assert logits.shape == (4, 6, 4)
    assert len(outputs) == 5
    assert all(value.shape == (4,) for value in outputs)
    local_presence = torch.ones(4, 6)
    local_fake = torch.tensor([[0.0] * 6, [1.0] * 6] * 2)
    labels = torch.tensor([0.0, 1.0, 0.0, 1.0])
    loss, terms = joint_temporal_loss(
        logits, outputs, mask.reshape(4, -1),
        local_presence, local_presence, local_fake, local_fake,
        labels, labels, labels, torch.ones(4), torch.ones(4), torch.ones(4),
    )
    assert torch.isfinite(loss)
    assert set(terms) == {
        "component", "file", "local", "presence", "global_presence",
    }


def test_component_presence_suppresses_file_evidence():
    model = SpearTemporalJointHead(2, torch.zeros(2), torch.ones(2), hidden=4)
    mask = torch.ones(1, 2, dtype=torch.bool)
    logits = torch.tensor([[[-8.0, -8.0, 8.0, 8.0], [-8.0, -8.0, 8.0, 8.0]]])
    absent_file = model.aggregate(logits, mask)[0]
    logits[..., :2] = 8.0
    present_file = model.aggregate(logits, mask)[0]
    assert absent_file < present_file
