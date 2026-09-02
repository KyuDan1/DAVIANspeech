import numpy as np

from src.dual_domain_inference import _fuse


def test_logit_fusion_has_exact_endpoints_and_balanced_odds():
    anchor = np.asarray([0.2, 0.8])
    expert = np.asarray([0.8, 0.2])
    assert np.allclose(_fuse(anchor, expert, 0.0), anchor)
    assert np.allclose(_fuse(anchor, expert, 1.0), expert)
    assert np.allclose(_fuse(anchor, expert, 0.5), 0.5)
