import numpy as np
import pandas as pd
import pytest

from scripts.evaluate_lowband_music_v67 import blend_music, completed


def test_only_music_changes_and_reverse_keys_are_aligned():
    anchor = pd.DataFrame(dict(DATASET=['dev', 'dev'], ID=['a', 'b'],
        MUSIC_FAKE_PROB=['0.20', '0.80'], FILE_FAKE_PROB=['0.1', '0.9'],
        VOICE_FAKE_PROB=['0.123400', '0.987600'], VOICE_PRESENT_PROB=['0.77', '0.88'],
        MUSIC_PRESENT_PROB=['0.99', '0.98']))
    candidate = anchor[['DATASET', 'ID', 'MUSIC_FAKE_PROB']].iloc[::-1].copy()
    candidate['MUSIC_FAKE_PROB'] = ['0.2', '0.8']
    result = blend_music(anchor, candidate, .5)
    np.testing.assert_allclose(result.MUSIC_FAKE_PROB.astype(float), [.5, .5])
    pd.testing.assert_frame_equal(result.drop(columns='MUSIC_FAKE_PROB'), anchor.drop(columns='MUSIC_FAKE_PROB'))
    pd.testing.assert_frame_equal(blend_music(anchor, candidate, 0), anchor)
    with pytest.raises(ValueError, match='matching'):
        blend_music(anchor, candidate.iloc[:1], .1)


def test_incomplete_run_cannot_be_used_for_selection(tmp_path):
    with pytest.raises(ValueError, match='incomplete'):
        completed(tmp_path)
