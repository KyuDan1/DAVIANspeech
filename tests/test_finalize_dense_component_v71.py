import copy
import json
from pathlib import Path
import sys

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from finalize_dense_component_v71 import audit_draws, audit_run, require_declared_rows


@pytest.fixture
def draw_bank(tmp_path):
    pd.DataFrame([{'ID': 'train-1', 'DATASET': 'authorized'}]).to_csv(tmp_path / 'train_ids.csv', index=False)
    pd.DataFrame([{'ID': key, 'GROUP_ID': 'group-' + key} for key in ['VR', 'VF', 'MR', 'MF', 'VR2']]).to_csv(
        tmp_path / 'source_catalog.csv', index=False)
    sources = {key: dict(id=key, group='group-' + key, sha256=f'{index + 1:064x}')
               for index, key in enumerate(['VR', 'VF', 'MR', 'MF'])}
    sources['REAL_INSERT'] = dict(id='VR2', group='group-VR2', sha256='f' * 64)
    original = dict(kind='original_train', row=0, dataset='authorized', id='train-1', epoch=1, draw=0)
    synthetic = dict(kind='synthetic_train', channel='clean', layout='sparse_voice', seconds=4,
                     epoch=1, draw=1, sources=sources)
    config = dict(epochs=2, samples_per_epoch=2, synthetic_channels=['clean'],
                  synthetic_layouts=['sparse_voice'], synthetic_durations=[4])
    for epoch in [1, 2]:
        draws = [copy.deepcopy(original), copy.deepcopy(synthetic)]
        for draw in draws:
            draw['epoch'] = epoch
        (tmp_path / f'epoch_{epoch}_draws.json').write_text(json.dumps(draws))
    return tmp_path, config


def mutate(path, function):
    rows = json.loads(path.read_text())
    function(rows)
    path.write_text(json.dumps(rows))


def test_complete_matched_draw_trace_is_accepted(draw_bank):
    run, config = draw_bank
    report = audit_draws(run, config)
    assert report['unique_consumed_raw_sources'] == 5
    assert report['counts']['synthetic_train'] == 2
    assert report['counts']['original_train'] == 2


def test_only_exact_full_development_scope_is_accepted():
    declared = pd.DataFrame({'ID': ['a', 'b'], 'DATASET': ['one', 'two']})
    require_declared_rows(declared.iloc[::-1], declared)
    for bad in [declared.iloc[:1], pd.concat([declared.iloc[:1]] * 2), declared.assign(ID=['a', 'heldout'])]:
        with pytest.raises(ValueError, match='exact declared'):
            require_declared_rows(bad, declared)


@pytest.mark.parametrize('change,match', [
    (lambda rows: rows.pop(), 'incomplete'),
    (lambda rows: rows[1].update(draw=0), 'order'),
    (lambda rows: rows[0].update(id='heldout'), 'identity'),
    (lambda rows: rows[0].update(row=-1), 'outside'),
    (lambda rows: rows[1].update(channel='undeclared'), 'undeclared'),
    (lambda rows: rows[1]['sources']['VF'].update(id='heldout'), 'catalog'),
    (lambda rows: rows[1]['sources']['VF'].update(sha256='0' * 64), 'changed'),
    (lambda rows: rows[1]['sources'].update(REAL_INSERT=copy.deepcopy(rows[1]['sources']['VR'])), 'change source group'),
])
def test_invalid_training_trace_is_rejected(draw_bank, change, match):
    run, config = draw_bank
    mutate(run / 'epoch_2_draws.json', change)
    with pytest.raises(ValueError, match=match):
        audit_draws(run, config)


def test_missing_completion_cannot_be_finalized(tmp_path):
    with pytest.raises(FileNotFoundError):
        audit_run(tmp_path)


def test_smoke_cannot_be_finalized_as_accuracy(tmp_path):
    (tmp_path / 'completed.json').write_text(json.dumps({'smoke': True}))
    (tmp_path / 'manifest.json').write_text(json.dumps({'smoke': True}))
    with pytest.raises(ValueError, match='smoke'):
        audit_run(tmp_path)


def test_pair_scoring_keeps_every_forward_inside_one_file(monkeypatch):
    import numpy as np
    import torch
    import rescore_dense_component_pair_v71 as scorer
    calls = []
    def fake_forward(encoder, heads, windows, lengths, counts, chunk, temperature):
        calls.append((len(windows), counts))
        return {name: torch.zeros(1, 5) for name in heads}, None, None
    monkeypatch.setattr(scorer, 'paired_dense_logits', fake_forward)
    config = {'encoder_chunk': 16, 'temperature': 5.}
    first = scorer.predict_one(None, {'file_only': None, 'dense_supervised': None}, np.zeros(960000, np.float32), config)
    scorer.predict_one(None, {'file_only': None, 'dense_supervised': None}, np.zeros(64000, np.float32), config)
    assert calls == [(6, [6]), (1, [1])]
    assert all(np.array_equal(probabilities, np.full(5, .5)) for probabilities in first.values())


def test_pair_scoring_rejects_nonfinite_output(monkeypatch):
    import numpy as np
    import torch
    import rescore_dense_component_pair_v71 as scorer
    monkeypatch.setattr(scorer, 'paired_dense_logits', lambda *args: ({'file_only': torch.full((1, 5), float('nan'))}, None, None))
    with pytest.raises(ValueError, match='nonfinite'):
        scorer.predict_one(None, {'file_only': None}, np.zeros(64000, np.float32), {'encoder_chunk': 16, 'temperature': 5.})
