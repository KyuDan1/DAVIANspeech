import numpy as np

from src.modern_fakeprint_detector import ModernFakeprintDetector, _blend


def test_embedding_shape_and_range():
    time = np.arange(16_000 * 5, dtype=np.float32) / 16_000
    audio = np.sin(2 * np.pi * 440 * time).astype(np.float32)
    feature = ModernFakeprintDetector.embedding(audio)
    assert feature.shape == (3_585,)
    assert np.isfinite(feature).all()
    assert feature.min() >= 0.0
    assert feature.max() <= 1.0


def test_embedding_rejects_empty_audio():
    try:
        ModernFakeprintDetector.embedding(np.array([], dtype=np.float32))
    except ValueError:
        pass
    else:
        raise AssertionError("empty audio must be rejected")


def test_blend_endpoints():
    assert np.isclose(_blend(0.2, 0.8, 0.0), 0.2)
    assert np.isclose(_blend(0.2, 0.8, 1.0), 0.8)
