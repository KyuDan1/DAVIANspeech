import numpy as np
import pytest
import pandas as pd
from src.wpt_slot_replacement import logit, replace_wpt_slot
from scripts.evaluate_wpt_slot_v62 import compose


def mix(base, expert):
    return np.exp(-np.logaddexp(0, -(.6 * logit(base) + .4 * logit(expert))))


def test_replacement_matches_fresh_composition_not_an_extra_ensemble():
    base = np.array([.01, .2, .5, .8, .99])
    old = np.array([.8, .4, .6, .2, .1])
    new = np.array([.2, .3, .4, .5, .9])
    result = replace_wpt_slot(mix(base, old), old, new)
    np.testing.assert_allclose(result, mix(base, new), atol=1e-14)


def test_identity_is_bit_exact_even_with_csv_rounding():
    old = np.array([.2, .7, .8])
    final = np.round(mix(np.array([.05, .5, .95]), old), 10)
    np.testing.assert_array_equal(replace_wpt_slot(final, old, old), final)


def test_incompatible_anchor_fails_instead_of_clipping_recovered_branch():
    with pytest.raises(ValueError, match='incompatible'):
        replace_wpt_slot(np.array([.99999]), np.array([.00001]), np.array([.5]))


@pytest.mark.parametrize('bad', [np.nan, -1., 2.])
def test_invalid_probability_rejected(bad):
    with pytest.raises(ValueError):
        replace_wpt_slot([.5], [.5], [bad])


def test_csv_alignment_and_nonfile_text_preserved():
    anchor = pd.DataFrame(dict(DATASET=['dev', 'dev'], ID=['a', 'b'],
        FILE_FAKE_PROB=['0.5', '0.5'], VOICE_FAKE_PROB=['0.1234567890', '0.8888888888'],
        MUSIC_FAKE_PROB=['0.8', '0.2'], VOICE_PRESENT_PROB=['0.99', '0.01'], MUSIC_PRESENT_PROB=['0.1', '0.9']))
    old = anchor[['DATASET', 'ID', 'FILE_FAKE_PROB']].iloc[::-1].copy()
    new = old.copy()
    new['FILE_FAKE_PROB'] = ['0.8', '0.2']
    result = compose(anchor, old, new, .4)
    assert float(result.iloc[0].FILE_FAKE_PROB) < .5 < float(result.iloc[1].FILE_FAKE_PROB)
    pd.testing.assert_frame_equal(result.drop(columns='FILE_FAKE_PROB'), anchor.drop(columns='FILE_FAKE_PROB'))


def test_csv_rebase_rejects_missing_file_instead_of_inner_join():
    frame = pd.DataFrame(dict(DATASET=['dev'], ID=['a'], FILE_FAKE_PROB=['.5']))
    with pytest.raises(ValueError, match='keys differ'):
        compose(frame, frame, frame.iloc[:0], .4)
