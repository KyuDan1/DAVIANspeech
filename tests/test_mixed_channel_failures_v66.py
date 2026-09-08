import numpy as np
import pandas as pd
import pytest

from scripts.audit_mixed_channel_failures_v66 import align, diagnose, eer_point


def example():
    rows, preds = [], []
    for i, (v, m) in enumerate([(0, 0), (1, 0), (0, 1), (1, 1)]):
        for channel in ['clean', 'phone']:
            rows.append(dict(DATASET='dev', ID=f'{i}_{channel}', BASE_ID=str(i), CHANNEL=channel,
                FILE_FAKE=max(v, m), VOICE_FAKE=v, MUSIC_FAKE=m, VOICE_PRESENT=1,
                MUSIC_PRESENT=1, COMPONENT_CASE=('F' if v else 'R') + ('F' if m else 'R'),
                MIX_MODE='concurrent', SNR_DB=0))
            probs = [(.9 if x else .1) if channel == 'clean' else .5 for x in [max(v, m), v, m]]
            preds.append(dict(DATASET='dev', ID=f'{i}_{channel}',
                              **dict(zip(['FILE_FAKE_PROB', 'VOICE_FAKE_PROB', 'MUSIC_FAKE_PROB'], probs))))
    return pd.DataFrame(rows), pd.DataFrame(preds)


def test_constant_cell_has_no_eer_and_constant_scores_have_half_eer():
    assert eer_point([1, 1], [.1, .2]) == (None, None)
    assert eer_point([0, 1], [.5, .5])[0] == .5


def test_alignment_exact_and_order_independent():
    truth, prediction = example()
    first = align(truth, prediction)
    second = align(truth, prediction.sample(frac=1, random_state=4))
    pd.testing.assert_frame_equal(first, second)
    with pytest.raises(ValueError, match='exactly'):
        align(truth, prediction.iloc[:-1])


def test_paired_metadata_and_missing_channels_rejected():
    truth, prediction = example()
    truth.loc[1, 'SNR_DB'] = 10
    with pytest.raises(ValueError, match='metadata'):
        align(truth, prediction)
    truth, prediction = example()
    with pytest.raises(ValueError, match='incomplete'):
        align(truth.iloc[:-1], prediction.iloc[:-1])


def test_clean_threshold_is_held_fixed_and_pair_sign_is_label_aware():
    truth, prediction = example()
    result = diagnose(align(truth, prediction))
    errors = result['fixed_clean_threshold']
    assert np.allclose(errors.clean_dev_threshold, .9)
    assert errors.query("channel == 'clean'").error_rate.max() == 0
    assert errors.query("channel == 'phone' and cell == 'FF'").error_rate.min() == 1
    assert np.allclose(result['paired_changes'].mean_signed_confidence_change, -.4)
    assert np.allclose(result['paired_changes'].fraction_confidence_worsened, 1)
    assert result['contrasts'].query("axis == 'CHANNEL' and group == 'phone'").eer.eq(.5).all()
