"""Component-decoupled variant of the EAT/SPEAR authenticity head."""

from __future__ import annotations

import torch
from torch.nn import functional as F

try:
    from .dual_domain_head import DualDomainHead
except ImportError:  # pragma: no cover - offline flat-module package
    from dual_domain_head import DualDomainHead


class InvariantDualDomainHead(DualDomainHead):
    """Keep component marginals independent from the RR/RF/FR/FF head.

    The joint classifier remains an auxiliary training target, but it no longer
    leaks voice evidence into Music or music evidence into Voice.  File uses a
    mostly-direct score with a small differentiable OR prior.
    """

    def __init__(self, *args, file_component_weight: float = 0.10, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.file_component_weight = float(file_component_weight)

    def probabilities(
        self, task_logits: torch.Tensor, joint_logits: torch.Tensor
    ) -> torch.Tensor:
        del joint_logits
        direct = task_logits.sigmoid()
        component_or = 1 - (1 - direct[:, 0]) * (1 - direct[:, 1])
        file_score = (
            (1 - self.file_component_weight) * direct[:, 2]
            + self.file_component_weight * component_or
        )
        return torch.stack((direct[:, 0], direct[:, 1], file_score), dim=-1)


def invariant_multitask_loss(
    model: InvariantDualDomainHead,
    task_logits: torch.Tensor,
    joint_logits: torch.Tensor,
    component_targets: torch.Tensor,
    joint_targets: torch.Tensor,
) -> torch.Tensor:
    """Balanced direct tasks with a low-weight auxiliary factorial objective."""
    direct = F.binary_cross_entropy_with_logits(task_logits, component_targets)
    joint = F.cross_entropy(joint_logits, joint_targets)
    probabilities = model.probabilities(task_logits, joint_logits)
    component_or = 1 - (1 - probabilities[:, 0]) * (1 - probabilities[:, 1])
    consistency = F.smooth_l1_loss(probabilities[:, 2], component_or)
    return direct + 0.25 * joint + 0.05 * consistency
