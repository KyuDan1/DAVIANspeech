import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from spear_detector import fuse_cross_component_scores  # noqa: E402


def test_cross_component_fusion_uses_conservative_default_weight():
    file_score, music_score = fuse_cross_component_scores(0.2, 0.3, 0.8, 0.9)
    assert file_score == pytest.approx(0.26)
    assert music_score == pytest.approx(0.36)


def test_cross_component_fusion_validates_weight():
    with pytest.raises(ValueError):
        fuse_cross_component_scores(0.2, 0.3, 0.8, 0.9, weight=1.1)
