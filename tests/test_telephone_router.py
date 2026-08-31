import numpy as np

from telephone_channel import apply_channel
from telephone_router import TelephoneRouter, extract_telephone_features


def test_telephone_features_are_finite_and_fixed_size():
    rng = np.random.default_rng(7)
    audio = rng.standard_normal(32_000).astype(np.float32)
    first = extract_telephone_features(audio)
    second = extract_telephone_features(audio)
    assert first.shape == second.shape
    assert first.ndim == 1
    assert first.size > 400
    assert np.isfinite(first).all()
    np.testing.assert_array_equal(first, second)


def test_resample_channel_removes_high_band_energy():
    time = np.arange(32_000) / 16_000
    audio = (
        np.sin(2 * np.pi * 1_000 * time)
        + np.sin(2 * np.pi * 6_000 * time)
    ).astype(np.float32)
    telephone = apply_channel(audio, "resample8k")
    frequency = np.fft.rfftfreq(len(telephone), 1 / 16_000)
    power = np.abs(np.fft.rfft(telephone)) ** 2
    assert power[frequency > 4_200].sum() / power.sum() < 1e-4


def test_portable_router_uses_stored_threshold(tmp_path):
    rng = np.random.default_rng(7)
    audio = rng.standard_normal(16_000).astype(np.float32)
    features = extract_telephone_features(audio)
    checkpoint = tmp_path / "router.npz"
    np.savez(
        checkpoint,
        mean=np.zeros_like(features),
        scale=np.ones_like(features),
        weight=np.zeros_like(features),
        bias=np.asarray(0.0, dtype=np.float32),
        threshold=np.asarray(0.6, dtype=np.float32),
    )
    router = TelephoneRouter(checkpoint)
    assert router.probability(audio) == 0.5
    assert not router.is_narrowband(audio)
