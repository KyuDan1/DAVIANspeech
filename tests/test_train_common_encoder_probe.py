import numpy as np
import pandas as pd
import torch

from scripts.train_common_encoder_probe import collate, file_logits, metrics
from src.common_encoder_probe import CommonTokenHead, component_loss
from src.evaluate_diagnostic import LABEL_COLUMNS, PREDICTION_COLUMNS
from scripts.smoke_common_encoder_probe import pad_token_batches


class ToyEncoder:
    def __call__(self, windows, lengths):
        return windows[:, None, :, None].expand(-1, 3, -1, 4).clone(), torch.arange(windows.shape[1])[None] < lengths[:, None]


def test_dynamic_encoder_token_counts_are_padded_with_invalid_mask():
    tokens, mask = pad_token_batches([torch.ones(1, 3, 7, 4), torch.ones(1, 3, 3, 4)],
                                    [torch.ones(1, 7, dtype=torch.bool), torch.ones(1, 3, dtype=torch.bool)])
    assert tokens.shape == (2, 3, 7, 4)
    assert mask.sum(1).tolist() == [7, 3]
    assert tokens[1, :, 3:].count_nonzero() == 0


def test_complete_file_loss_not_per_crop_and_two_independent_backwards():
    items = [(np.random.default_rng(1).normal(size=(2, 8)).astype('float32'), np.array([8, 8]),
              np.array([1, 1, 0, 1, 1], dtype='float32'), 0),
             (np.zeros((1, 8), dtype='float32'), np.array([4]),
              np.array([0, 0, np.nan, 1, 0], dtype='float32'), 1)]
    windows, lengths, counts, targets, indices = collate(items)
    heads = {name: CommonTokenHead(4, width=8, pooling=name) for name in ['mean', 'attention']}
    outputs = file_logits(ToyEncoder(), heads, windows, lengths, counts, 2, 2.)
    assert counts == [2, 1] and indices == [0, 1]
    for name, values in outputs.items():
        assert values.shape == (2, 5)  # Not three independently labeled windows.
        component_loss(values, targets).backward()
        assert torch.isfinite(heads[name].projection.weight.grad).all()


def test_macro_selection_includes_pure_voice_and_music_domains():
    labels = [[0, 0, np.nan, 1, 0], [1, 1, np.nan, 1, 0],
              [0, np.nan, 0, 0, 1], [1, np.nan, 1, 0, 1],
              [0, 0, 0, 1, 1], [1, 1, 1, 1, 1]]
    frame = pd.DataFrame(labels, columns=LABEL_COLUMNS)
    frame['DATASET'] = ['voice', 'voice', 'music', 'music', 'mixed', 'mixed']
    frame[PREDICTION_COLUMNS] = frame[LABEL_COLUMNS].fillna(0).to_numpy() * .8 + .1
    summary, slices = metrics(frame)
    assert summary['macro_ads'] == 1 and summary['selection'] == 1
    # Reverse just the pure music detector. Its domain MUST affect selection.
    frame.loc[frame.DATASET.eq('music'), 'MUSIC_FAKE_PROB'] = [.9, .1]
    summary, _ = metrics(frame)
    assert summary['macro_ads'] == .85
    assert len(slices) == 3
