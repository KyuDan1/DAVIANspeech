import pandas as pd

from scripts.train_spear_temporal_joint_mil import channel_pairs, component_pairs


def test_component_pairs_hold_target_and_change_nuisance():
    frame = pd.DataFrame({
        "VOICE_SOURCE_ID": ["real", "real", "fake", "fake"],
        "VOICE_FAKE": [0, 0, 1, 1],
        "MUSIC_FAKE": [0, 1, 0, 1],
    })
    pairs = component_pairs(
        frame, "VOICE_SOURCE_ID", "VOICE_FAKE", "MUSIC_FAKE"
    )
    assert len(pairs) == 4
    for left, right in pairs:
        assert frame.loc[left, "VOICE_FAKE"] == frame.loc[right, "VOICE_FAKE"]
        assert frame.loc[left, "MUSIC_FAKE"] != frame.loc[right, "MUSIC_FAKE"]


def test_channel_pairs_find_clean_parent():
    frame = pd.DataFrame({
        "ID": ["a", "a_phone", "b"],
        "PARENT_ID": [None, "a", "missing"],
        "FILE_FAKE": [1, 1, 0], "VOICE_FAKE": [1, 1, 0],
        "MUSIC_FAKE": [0, 0, 0],
    })
    assert channel_pairs(frame).tolist() == [[1, 0]]
