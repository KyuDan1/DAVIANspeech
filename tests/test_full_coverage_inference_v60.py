import numpy as np
import pytest
import torch
from src.full_coverage_wpt_inference import load_full_coverage_model, predict_one_audio


def test_smoke_checkpoint_is_refused_before_backbone_loading(tmp_path):
    path = tmp_path / 'smoke.pt'
    torch.save(dict(model_type='full_coverage_wpt_file_v60', smoke=True), path)
    with pytest.raises(ValueError, match='smoke'):
        load_full_coverage_model(tmp_path, path, torch.device('cpu'))


def test_one_file_result_does_not_depend_on_prior_files():
    class Model:
        def forward_windows(self, x):
            return x.mean(dim=-1, keepdim=True).expand(-1, -1, 3)
    config = dict(window=100, temperature=2., window_chunk_size=16)
    a = np.ones(1000, dtype=np.float32)
    b = np.zeros(350, dtype=np.float32)
    model = Model()
    device = torch.device('cpu')
    first = predict_one_audio(model, a, config, device)
    predict_one_audio(model, b, config, device)
    assert first == predict_one_audio(model, a, config, device)
