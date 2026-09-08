import numpy as np
import pandas as pd
import pytest
import torch

from src.lowband_music_v67 import LowbandMusic
from scripts.train_lowband_music_v67 import music_weights


@pytest.fixture(autouse=True)
def limit_threads():
    original = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(original)


def test_features_are_lowband_gain_and_polarity_invariant():
    torch.manual_seed(4)
    x = torch.randn(2, 4000)
    model = LowbandMusic(phase=True)
    f = model.features(x)
    assert f.shape[1:3] == (3, 105)
    torch.testing.assert_close(f, model.features(.1 * x), atol=2e-4, rtol=2e-4)
    torch.testing.assert_close(f, model.features(-x), atol=1e-6, rtol=1e-6)
    assert f[:, 1:].abs().sum() > 0
    assert LowbandMusic(phase=False).features(x)[:, 1:].abs().sum() == 0


def test_silent_input_and_parameter_gradients_are_finite():
    model = LowbandMusic(phase=True)
    output = model.forward_windows(torch.zeros(2, 4000))
    assert torch.isfinite(output).all()
    output.square().mean().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())


def test_padding_and_file_order_do_not_change_predictions():
    torch.manual_seed(5)
    model = LowbandMusic(phase=True).eval()
    x = torch.randn(2, 2, 4000)
    mask = torch.tensor([[True, False], [True, True]])
    with torch.inference_mode():
        original = model(x, mask, chunk_size=3)
        x[0, 1] = 99999
        changed = model(x, mask, chunk_size=1)
        reversed_score = model(x.flip(0), mask.flip(0), chunk_size=3).flip(0)
    torch.testing.assert_close(original, changed, atol=2e-6, rtol=2e-6)
    torch.testing.assert_close(original, reversed_score, atol=2e-6, rtol=2e-6)


def test_phase_ablation_has_identical_parameters_and_initialization():
    torch.manual_seed(8)
    a = LowbandMusic(False)
    torch.manual_seed(8)
    b = LowbandMusic(True)
    assert a.state_dict().keys() == b.state_dict().keys()
    for key, value in a.state_dict().items():
        torch.testing.assert_close(value, b.state_dict()[key], atol=0, rtol=0)


def test_music_class_sampler_does_not_balance_file_class_instead():
    rows = []
    for v, m, copies in [(0, 0, 1), (1, 0, 20), (0, 1, 3), (1, 1, 2)]:
        rows.extend([dict(DATASET='train', FILE_FAKE=max(v, m), VOICE_FAKE=v,
            MUSIC_FAKE=m, VOICE_PRESENT=1, MUSIC_PRESENT=1)] * copies)
    frame = pd.DataFrame(rows)
    original = frame.copy()
    frame['weight'] = music_weights(frame).numpy()
    np.testing.assert_allclose(frame.groupby('MUSIC_FAKE').weight.sum().to_numpy(), [.5, .5])
    pd.testing.assert_frame_equal(frame.drop(columns='weight'), original)
    original.loc[0, 'MUSIC_PRESENT'] = 0
    with pytest.raises(ValueError, match='Music-present'):
        music_weights(original)
