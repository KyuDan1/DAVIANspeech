import importlib.util
from pathlib import Path
import torch
import yaml

from src.full_coverage_wpt import forward_bags
from src import full_coverage_wpt_inference as inference

ROOT = Path(__file__).resolve().parents[1]


class RecordingModel:
    def __init__(self):
        self.inputs = []

    def forward_windows(self, x):
        self.inputs.append(x.detach().clone())
        return x.mean(dim=-1, keepdim=True).expand(-1, -1, 3)


def test_default_is_bit_exact_with_preserved_v60_source():
    path = ROOT / 'tests/fixtures/full_coverage_wpt_v60_reference.py'
    spec = importlib.util.spec_from_file_location('src._legacy_v60_snapshot', path)
    legacy = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(legacy)
    x = torch.randn(2, 3, 100)
    mask = torch.tensor([[1, 1, 0], [1, 1, 1]], dtype=torch.bool)
    torch.testing.assert_close(forward_bags(RecordingModel(), x, mask),
                               legacy.forward_bags(RecordingModel(), x, mask), rtol=0, atol=0)


def test_peak_option_is_post_preemphasis_and_window_local():
    x = torch.tensor([[[1., 2., -2., 0.], [.1, .2, -.2, 0.], [999., 999., 999., 999.]]])
    mask = torch.tensor([[1, 1, 0]], dtype=torch.bool)
    model = RecordingModel()
    forward_bags(model, x, mask, peak_normalize=True)
    actual = model.inputs[0][:, 0]
    raw = x[mask]
    expected = raw.clone()
    expected[:, 1:] = raw[:, 1:] - .97 * raw[:, :-1]
    expected /= expected.abs().amax(-1, keepdim=True) + 1e-8
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(actual[0], actual[1])


def test_inference_uses_checkpoint_normalization_flag(monkeypatch):
    import numpy as np
    observed = []
    def fake_forward(*args, **kwargs):
        observed.append(kwargs['peak_normalize'])
        return torch.zeros(1, 3)
    monkeypatch.setattr(inference, 'forward_bags', fake_forward)
    config = dict(window=4, temperature=2., window_chunk_size=16, peak_normalize=True)
    inference.predict_one_audio(None, np.ones(10, dtype=np.float32), config, torch.device('cpu'))
    del config['peak_normalize']
    inference.predict_one_audio(None, np.ones(10, dtype=np.float32), config, torch.device('cpu'))
    assert observed == [True, False]


def test_training_config_matches_bce_control_except_normalization_and_metadata():
    control = yaml.safe_load((ROOT / 'configs/paired_wpt_file_v60.yaml').read_text())
    candidate = yaml.safe_load((ROOT / 'configs/paired_peak_wpt_v64.yaml').read_text())
    metadata = {'schema_version', 'variants', 'decision', 'peak_normalize'}
    assert {k: v for k, v in control.items() if k not in metadata} == {
        k: v for k, v in candidate.items() if k not in metadata}
    assert candidate['peak_normalize'] is True
    assert candidate['variants']['paired_peak'] == control['variants']['paired_bce']
