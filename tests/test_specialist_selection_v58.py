import numpy as np
import pandas as pd
import pytest

from scripts.train_three_stream_anchor_residual import specialist_selection


def test_music_selection_ignores_unrelated_improvement_and_absent_music():
    domains = pd.DataFrame({"MUSIC_EER": [.1, .3, np.nan], "FILE_EER": [.4,.4,.4]})
    pooled = {"MUSIC_EER": .2, "FILE_EER": .4}
    score = specialist_selection(domains, pooled, "music")
    assert score == pytest.approx(.775)
    domains["FILE_EER"] = 0
    pooled["FILE_EER"] = 0
    assert specialist_selection(domains, pooled, "music") == score
    domains["MUSIC_EER"] = [.1, .4, np.nan]
    assert specialist_selection(domains, pooled, "music") < score


def test_specialist_rejects_undefined_task():
    with pytest.raises(ValueError, match="no defined"):
        specialist_selection(pd.DataFrame({"MUSIC_EER": [np.nan]}),
                             {"MUSIC_EER": .2}, "music")
