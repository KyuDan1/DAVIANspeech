import pytest
from scripts.finalize_paired_wpt_v60 import choose_candidate, completed_variant


def test_incomplete_run_rejected_before_any_comparison(tmp_path):
    with pytest.raises(ValueError, match='not complete'):
        completed_variant(tmp_path)


def test_no_candidate_when_both_anchor_checks_fail():
    variants = {name: {'selection': score} for name, score in [('paired_bce', .7), ('paired_consistency', .8)]}
    gates = {name: {'pass_gate': False} for name in variants}
    assert choose_candidate(variants, gates) is None


def test_bce_can_win_without_consistency_gain():
    variants = {'paired_bce': {'selection': .8}, 'paired_consistency': {'selection': .7}}
    gates = {name: {'pass_gate': True} for name in variants}
    assert choose_candidate(variants, gates) == 'paired_bce'


def test_failed_anchor_gate_cannot_be_overridden_by_pooled_selection():
    variants = {'paired_bce': {'selection': .7}, 'paired_consistency': {'selection': .9}}
    gates = {'paired_bce': {'pass_gate': True}, 'paired_consistency': {'pass_gate': False}}
    assert choose_candidate(variants, gates) == 'paired_bce'
