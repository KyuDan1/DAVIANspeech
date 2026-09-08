import pandas as pd
import pytest
from scripts.score_long_voice_v74 import gate_decision


def fixture():
    rows = [dict(BANK=bank, AXIS=axis, GROUP=str(group), FILE_EER_delta=-.02)
            for bank in ['uniform', 'fixed']
            for axis, groups in [('ALL', ['ALL']), ('DURATION', [30, 45, 60]),
                                  ('CHANNEL', ['clean', 'g711_ulaw', 'opus_nb_8k'])]
            for group in groups]
    return pd.DataFrame(rows), dict(uniform_min_pooled_file_eer_improvement=.01,
        max_length_channel_regression=.025, fixed_max_pooled_regression=.025)


def test_all_predeclared_gates_pass():
    table, gates = fixture()
    assert gate_decision(table, gates)['passed']


@pytest.mark.parametrize('bank,axis,value', [('uniform', 'ALL', -.009),
    ('fixed', 'ALL', .026), ('uniform', 'DURATION', .026), ('fixed', 'CHANNEL', .026)])
def test_one_regression_cannot_hide_behind_pooled_improvement(bank, axis, value):
    table, gates = fixture()
    table.loc[table.BANK.eq(bank) & table.AXIS.eq(axis), 'FILE_EER_delta'] = value
    assert not gate_decision(table, gates)['passed']


def test_missing_gate_cell_is_not_a_pass():
    table, gates = fixture()
    with pytest.raises(ValueError):
        gate_decision(table.iloc[:-1], gates)


def test_wrong_duration_cannot_replace_a_required_gate():
    table, gates = fixture()
    table.loc[table.AXIS.eq('DURATION') & table.GROUP.eq('30'), 'GROUP'] = '20'
    with pytest.raises(ValueError):
        gate_decision(table, gates)
