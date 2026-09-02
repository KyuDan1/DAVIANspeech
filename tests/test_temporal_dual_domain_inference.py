import numpy as np

from src.temporal_dual_domain_inference import _fuse


def test_temporal_fusion_has_identity_endpoints_and_is_monotone():
    anchor = np.array([0.1, 0.5, 0.9])
    expert = np.array([0.9, 0.5, 0.1])
    assert np.allclose(_fuse(anchor, expert, 0), anchor)
    assert np.allclose(_fuse(anchor, expert, 1), expert)
    middle = _fuse(anchor, expert, 0.25)
    assert anchor[0] < middle[0] < expert[0]
    assert expert[2] < middle[2] < anchor[2]
