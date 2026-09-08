from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import yaml

from src.common_encoder_probe import FrozenEncoderTokens


def test_wavlm_extension_preserves_original_protocol():
    root = Path(__file__).resolve().parents[1]
    original = yaml.safe_load((root / 'configs/common_encoder_probe.yaml').read_text())
    extension = yaml.safe_load((root / 'configs/common_encoder_probe_wavlm.yaml').read_text())
    assert extension['encoders'].pop('wavlm') == 'models/wavlm-large'
    assert original == extension


def test_wavlm_local_loading_native_normalization_and_masks(monkeypatch):
    from transformers import WavLMModel, Wav2Vec2FeatureExtractor

    class FakeModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = SimpleNamespace(layers=list(range(24)))
            self.weight = torch.nn.Parameter(torch.ones(1))

        def _get_feat_extract_output_lengths(self, lengths):
            return lengths // 320

        def forward(self, input_values, attention_mask, output_hidden_states):
            self.last_inputs = input_values, attention_mask
            assert output_hidden_states
            states = tuple(torch.full((2, 512, 8), float(i)) for i in range(25))
            return SimpleNamespace(hidden_states=states)

    model = FakeModel()
    calls = []

    def load_model(path, **kwargs):
        calls.append((path, kwargs))
        return model

    def load_processor(path, **kwargs):
        calls.append((path, kwargs))
        return Wav2Vec2FeatureExtractor(do_normalize=True, return_attention_mask=True)

    monkeypatch.setattr(WavLMModel, 'from_pretrained', load_model)
    monkeypatch.setattr(Wav2Vec2FeatureExtractor, 'from_pretrained', load_processor)
    encoder = FrozenEncoderTokens('wavlm', Path('/local/wavlm'), device='cpu')
    windows = torch.full((2, 163840), 987.)  # Padding must never affect normalization.
    windows[0, :640] = torch.linspace(-2, 3, 640)
    windows[1, :960] = torch.linspace(5, 7, 960)
    tokens, mask = encoder(windows, torch.tensor([640, 960]))
    assert all(kwargs == {'local_files_only': True} for _, kwargs in calls)
    assert encoder.indices == (5, 11, 23)
    assert tokens[0, :, 0, 0].tolist() == [6, 12, 24]
    assert mask.sum(-1).tolist() == [2, 3]
    actual, sample_mask = model.last_inputs
    assert sample_mask.sum(-1).tolist() == [640, 960]
    assert not actual[0, 640:].any()
    np.testing.assert_allclose(actual[0, :640].mean(), 0, atol=1e-6)
    assert not model.training and not model.weight.requires_grad
