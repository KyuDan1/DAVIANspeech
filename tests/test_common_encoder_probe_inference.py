import numpy as np
import pandas as pd
import pytest
import torch

from src.common_encoder_probe import CommonTokenHead
from src.common_encoder_probe_inference import CommonProbePredictor
from scripts.verify_common_encoder_probe_inference import select_rows


class FakeEncoder:
    def __call__(self, windows, lengths):
        # Deterministic test encoder; full waveform is not discarded in production.
        return windows[:, None, :8, None].expand(-1, 3, -1, 4), torch.ones(len(windows), 8, dtype=torch.bool)


def test_completed_checkpoint_required(tmp_path):
    with pytest.raises(ValueError, match='completed'):
        CommonProbePredictor(tmp_path / 'attention/head.pt', tmp_path, device='cpu')


def test_single_file_inference_is_stateless_and_five_finite_probabilities():
    predictor = object.__new__(CommonProbePredictor)
    predictor.encoder = FakeEncoder()
    predictor.head = CommonTokenHead(4, width=8).eval()
    predictor.config = {'encoder_chunk': 2, 'temperature': 2.}
    a = np.random.default_rng(4).normal(size=320000).astype('float32')
    b = np.ones(64000, dtype='float32')
    first = predictor(a)
    predictor(b)
    assert np.array_equal(first, predictor(a))
    assert first.shape == (5,) and np.isfinite(first).all()
    assert ((first >= 0) & (first <= 1)).all()


def test_infrastructure_sample_includes_absent_component_cells():
    frame = pd.DataFrame([dict(DATASET='dev', ID=str(i), VOICE_PRESENT=1,
                               MUSIC_PRESENT=0, VOICE_FAKE=i % 2, MUSIC_FAKE=np.nan) for i in range(6)])
    result = select_rows(frame)
    assert len(result) == 4 and result.VOICE_FAKE.nunique() == 2
    pd.testing.assert_frame_equal(select_rows(frame, full=True), frame)
