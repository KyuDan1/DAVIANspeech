import numpy as np
import pytest
import torch

from src.lowband_music_v67 import LowbandMusic
from src.lowband_music_inference_v67 import load_model, predict_one_audio


def test_smoke_is_rejected_and_checkpoint_phase_is_checked(tmp_path):
    path = tmp_path / 'smoke.pt'
    config = dict(window=512, temperature=2., window_chunk_size=16, variants={'phase': {'phase': True}})
    checkpoint = dict(model_type='lowband_music_v67', smoke=True, phase=True, config=config,
                      variant='phase', state=LowbandMusic(True).state_dict())
    torch.save(checkpoint, path)
    with pytest.raises(ValueError, match='smoke'):
        load_model(path, torch.device('cpu'))
    model, restored = load_model(path, torch.device('cpu'), allow_smoke=True)
    assert model.phase and restored == config
    checkpoint['phase'] = False
    torch.save(checkpoint, path)
    with pytest.raises(ValueError, match='phase/config'):
        load_model(path, torch.device('cpu'), allow_smoke=True)


def test_inference_preserves_all_windows_and_file_local_mask():
    seen = []
    class Model:
        def __call__(self, windows, mask, temperature, chunk_size):
            seen.append((windows.clone(), mask.clone()))
            return torch.tensor([0.])
    config = dict(window=4, temperature=2., window_chunk_size=16)
    assert predict_one_audio(Model(), np.arange(10, dtype=np.float32), config, torch.device('cpu')) == .5
    assert seen[0][0].shape == (1, 3, 4)
    assert seen[0][1].all()
    assert set(seen[0][0].flatten().tolist()) == set(range(10))
