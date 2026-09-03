import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from simple_pipeline import high_band_energy_ratio, select_route  # noqa: E402


def test_component_routes_are_explicit():
    assert select_route(.99, .9, .9) == "mixed"
    assert select_route(.01, .8, .03) == "voice"
    assert select_route(.01, .3, .9) == "music"
    # Singing music may have a non-trivial voice tag, but is not speech-only.
    assert select_route(.01, .4, .9) == "music"
    assert select_route(.01, .6, .4, is_phone=True) == "voice"
    assert select_route(.01, .4, .6, is_phone=True) == "music"


def test_telephone_band_cue():
    rng = np.random.default_rng(20260830)
    broadband = rng.normal(size=16_000).astype(np.float32)
    telephone = np.sin(2 * np.pi * 1_000 * np.arange(16_000) / 16_000).astype(np.float32)
    assert high_band_energy_ratio(broadband) > 3e-6
    assert high_band_energy_ratio(telephone) < 3e-6
