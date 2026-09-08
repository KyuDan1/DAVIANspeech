import json
import numpy as np
import pandas as pd
import pytest

from scripts.compare_common_encoder_probe import (align_truth, paired_bootstrap,
                                                 completed_matched_runs, typed_metrics)
from src.evaluate_diagnostic import PREDICTION_COLUMNS


def frames():
    rows = []
    for base, (v, m) in enumerate([(0, 0), (1, 0), (0, 1), (1, 1)]):
        for channel in ['clean', 'phone']:
            rows.append(dict(DATASET='dev', ID=f'{base}_{channel}', BASE_ID=str(base), CHANNEL=channel,
                             COMPONENT_CASE=('F' if v else 'R') + ('F' if m else 'R'),
                             FILE_FAKE=max(v, m), VOICE_FAKE=v, MUSIC_FAKE=m,
                             VOICE_PRESENT=1, MUSIC_PRESENT=1))
    truth = pd.DataFrame(rows)
    prediction = truth[['DATASET', 'ID']].copy()
    prediction[PREDICTION_COLUMNS] = np.array([
        [max(row.VOICE_FAKE, row.MUSIC_FAKE), row.VOICE_FAKE, row.MUSIC_FAKE, 1, 1]
        for row in truth.itertuples()]) * .8 + .1
    return truth, prediction


def test_completion_gate_precedes_all_prediction_reads(tmp_path):
    with pytest.raises(RuntimeError, match='completed'):
        completed_matched_runs(tmp_path)


def closed_run_fixtures(tmp_path):
    for encoder in ['eat_large', 'spear', 'xlsr']:
        path = tmp_path / encoder
        path.mkdir()
        (path / 'completed.json').write_text('{}')
        (path / 'manifest.json').write_text(json.dumps(dict(encoder=encoder,
            config={'schema': 'common_encoder_probe_v1', 'epochs': 1, 'samples_per_epoch': 4},
            audit={'manifests': []}, code_sha256={})))
        for filename in ['train_ids.csv', 'development_ids.csv']:
            (path / filename).write_text('DATASET,ID\nexample,one\n')
        np.save(path / 'epoch_1_train_draws.npy', np.array([0, 0, 0, 0]))


def test_completed_sampler_and_order_must_match(tmp_path):
    closed_run_fixtures(tmp_path)
    _, _, records = completed_matched_runs(tmp_path)
    assert records[0]['count'] == 4
    np.save(tmp_path / 'xlsr/epoch_1_train_draws.npy', np.array([0, 0, 0, 1]))
    with pytest.raises(ValueError, match='sampler draws'):
        completed_matched_runs(tmp_path)


def test_completed_truth_and_order_hashes_must_match(tmp_path):
    closed_run_fixtures(tmp_path)
    (tmp_path / 'spear/development_ids.csv').write_text('DATASET,ID\nexample,two\n')
    with pytest.raises(ValueError, match='ordered'):
        completed_matched_runs(tmp_path)


def test_extended_registry_does_not_relax_training_settings(tmp_path):
    closed_run_fixtures(tmp_path)
    path = tmp_path / 'spear/manifest.json'
    manifest = json.loads(path.read_text())
    manifest['config']['encoders'] = {'wavlm': 'models/wavlm-large'}
    path.write_text(json.dumps(manifest))
    completed_matched_runs(tmp_path)
    manifest['config']['samples_per_epoch'] = 8
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match='configs differ'):
        completed_matched_runs(tmp_path)


def test_changed_checkpoint_registry_rejected(tmp_path):
    closed_run_fixtures(tmp_path)
    for name in ['spear', 'xlsr']:
        path = tmp_path / name / 'manifest.json'
        manifest = json.loads(path.read_text())
        manifest['config']['encoders'] = {'wavlm': f'models/{name}'}
        path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match='checkpoint registry changed'):
        completed_matched_runs(tmp_path)


def test_exact_alignment_reorders_but_rejects_missing_duplicate_or_nan():
    truth, prediction = frames()
    original = align_truth(truth, prediction)
    pd.testing.assert_frame_equal(original, align_truth(truth, prediction.iloc[::-1]))
    with pytest.raises(ValueError, match='keys'):
        align_truth(truth, prediction.iloc[:-1])
    with pytest.raises(ValueError, match='duplicate'):
        align_truth(truth, pd.concat([prediction, prediction.iloc[:1]]))
    prediction.loc[0, 'MUSIC_FAKE_PROB'] = np.nan
    with pytest.raises(ValueError, match='probabilities'):
        align_truth(truth, prediction)


def test_identical_predictions_have_zero_paired_interval():
    truth, prediction = frames()
    frame = align_truth(truth, prediction)
    for row in paired_bootstrap(frame, frame, repetitions=20):
        assert row['base_count'] == 4 and row['rows'] == 8
        assert row['delta_eer_ci_low'] == row['delta_eer_ci_high'] == 0


def test_worse_candidate_interval_and_fixed_seed():
    truth, prediction = frames()
    anchor = align_truth(truth, prediction)
    candidate = anchor.copy()
    candidate['MUSIC_FAKE_PROB'] = 1 - candidate['MUSIC_FAKE_PROB']
    result = paired_bootstrap(anchor, candidate, repetitions=20)
    assert result == paired_bootstrap(anchor, candidate, repetitions=20)
    music = next(row for row in result if row['task'] == 'MUSIC')
    assert music['delta_eer_ci_low'] == music['delta_eer_ci_high'] == 1
    with pytest.raises(ValueError, match='ordered'):
        paired_bootstrap(anchor, candidate.iloc[::-1], repetitions=20)


def test_single_class_type_does_not_invent_component_eer():
    truth, prediction = frames()
    frame = align_truth(truth, prediction)
    result = typed_metrics(frame)
    assert len(result) == 1 and result[0]['type'] == 'mixed'
    assert np.isnan(result[0]['CPS'])


def test_base_label_changes_rejected():
    truth, prediction = frames()
    frame = align_truth(truth, prediction)
    frame.loc[0, 'COMPONENT_CASE'] = 'FF'
    with pytest.raises(ValueError, match='labels change'):
        paired_bootstrap(frame, frame, repetitions=20)
