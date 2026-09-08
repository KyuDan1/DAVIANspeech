import importlib.util
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
spec = importlib.util.spec_from_file_location('paired_wpt_training_v60', ROOT / 'scripts/train_paired_wpt_file_v60.py')
training = importlib.util.module_from_spec(spec)
spec.loader.exec_module(training)


def test_pair_collation_preserves_correspondence_and_masks():
    a = np.ones((1, 8), dtype=np.float32)
    b = np.full((2, 8), 2, dtype=np.float32)
    windows, mask, targets, indices = training.collate_bags([(a, a + 10, 0., 9), (b, b + 10, 1., 3)])
    assert windows.shape == (4, 2, 8)
    torch.testing.assert_close(windows[:, 0, 0], torch.tensor([1., 2., 11., 12.]))
    assert mask.sum(dim=1).tolist() == [1, 2, 1, 2]
    assert targets.tolist() == [0., 1.]
    assert indices.tolist() == [9, 3]


def test_sampling_balances_file_classes_without_dev_priors():
    rows = []
    for dataset, count in [('small', 1), ('large', 7)]:
        for v, m in [(0, 0), (1, 0), (0, 1), (1, 1)]:
            rows.extend([dict(DATASET=dataset, FILE_FAKE=max(v, m), VOICE_PRESENT=1,
                              MUSIC_PRESENT=1, VOICE_FAKE=v, MUSIC_FAKE=m)] * count)
    frame = pd.DataFrame(rows)
    frame['weight'] = training.balanced_weights(frame).numpy()
    totals = frame.groupby(['DATASET', 'FILE_FAKE']).weight.sum().to_numpy()
    np.testing.assert_allclose(totals, np.full(4, .5))
