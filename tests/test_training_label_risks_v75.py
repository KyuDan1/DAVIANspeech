import pandas as pd
from scripts.report_training_label_risks_v75 import flag_rows


def test_review_flags_do_not_relabel_or_confuse_real_background():
    train = pd.DataFrame(dict(DATASET=['x', 'x', 'mixfake_music_train_v1'], ID=['a', 'b', 'c'],
        MUSIC_SOURCE_ID=['real', 'fake', 'not_screened'], MUSIC_GENERATOR=['real', 'x', 'suno'],
        VOICE_PRESENT=[0, 1, 1], VOICE_FAKE=[0, 0, 0], MUSIC_FAKE=[0, 1, 1]))
    original = train.copy(deep=True)
    source = pd.DataFrame(dict(SOURCE_ID=['real', 'fake'], LABEL=[0, 1], BOTH_VOICE_REVIEW=[True, True]))
    result = flag_rows(train, source)
    assert result.VOICE_ABSENCE_TARGET_REQUIRES_REVIEW.tolist() == [True, False, False]
    assert result.VOICE_REAL_TARGET_WITH_FLAGGED_FAKE_BACKGROUND.tolist() == [False, True, False]
    assert result.MIXFAKE_RF_SUNO_UDIO_BACKGROUND_UNVERIFIED.tolist() == [False, False, True]
    pd.testing.assert_frame_equal(train, original)
