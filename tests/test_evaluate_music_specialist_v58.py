import importlib.util
from pathlib import Path
import sys

import pandas as pd
import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / 'scripts'
sys.path.insert(0, str(SCRIPTS))
spec = importlib.util.spec_from_file_location('evaluate_music_specialist_v58', SCRIPTS / 'evaluate_music_specialist_v58.py')
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


def test_constant_score_eer_is_half_and_absent_music_is_ignored():
    frame = pd.DataFrame({'MUSIC_PRESENT': [1, 1, 0], 'MUSIC_FAKE': [0, 1, 1],
                          'score': [.5, .5, .99]})
    metrics = audit.music_metrics(frame, 'score')
    assert metrics['eer'] == .5
    assert metrics['n'] == 2


def test_one_class_has_no_eer_but_reports_operating_point():
    frame = pd.DataFrame({'MUSIC_PRESENT': [1, 1], 'MUSIC_FAKE': [1, 1],
                          'score': [.4, .8]})
    metrics = audit.music_metrics(frame, 'score')
    assert metrics['eer'] is None
    assert metrics['fnr_at_05'] == .5


def test_gate_rejects_channel_regression_despite_pooled_gain():
    rows = [dict(axis=axis, gain=gain, incumbent={'eer': .3}, candidate={'eer': .3 - gain})
            for axis, gain in [('pooled', .02), ('DATASET', .02), ('CHANNEL_V57', -.03)]]
    gate = dict(minimum_pooled_music_eer_gain=.01, maximum_domain_music_eer_regression=.025,
                maximum_channel_music_eer_regression=.025, require_mean_domain_improvement=True,
                require_worst_domain_nonworse=True)
    result = audit.decision(rows, gate)
    assert result['pass_gate'] is False
    assert result['checks']['pooled_improves'] is True
    assert result['checks']['channels_nonregressing'] is False


def test_mismatched_rows_are_rejected():
    truth = pd.DataFrame({'DATASET': ['dev'], 'ID': ['a']})
    predictions = pd.DataFrame({'DATASET': ['dev'], 'ID': ['b'], 'MUSIC_FAKE_PROB': [.5]})
    with pytest.raises(ValueError, match='exactly match'):
        audit.compare(truth, predictions, predictions)


def test_float_roundoff_does_not_count_as_domain_improvement():
    rows = [dict(axis=axis, gain=gain, incumbent={'eer': .3}, candidate={'eer': .3 - gain})
            for axis, gain in [('pooled', .02), ('DATASET', 1e-18), ('CHANNEL_V57', .01)]]
    gate = dict(minimum_pooled_music_eer_gain=.01, maximum_domain_music_eer_regression=.025,
                maximum_channel_music_eer_regression=.025, require_mean_domain_improvement=True,
                require_worst_domain_nonworse=True)
    assert audit.decision(rows, gate)['checks']['mean_domain_improves'] is False


def test_file_metrics_include_single_component_rows():
    frame = pd.DataFrame({'FILE_FAKE': [0, 1], 'MUSIC_PRESENT': [0, 0], 'score': [.1, .9]})
    result = audit.component_metrics(frame, 'score', 'file')
    assert result['eer'] == 0
    assert result['n'] == 2
