import numpy as np
import pandas as pd

from scripts.train_dual_domain_head import Bank
from scripts.train_invariant_dual_domain_head import CounterfactualDataset


def test_single_component_rows_are_kept_but_not_paired_as_counterfactuals():
    truth = pd.DataFrame([
        {
            "ID": "music_only", "VOICE_FAKE": np.nan, "MUSIC_FAKE": 1,
            "FILE_FAKE": 1, "VOICE_SOURCE_ID": "", "MUSIC_SOURCE_ID": "music_a",
        },
        {
            "ID": "voice_only", "VOICE_FAKE": 1, "MUSIC_FAKE": np.nan,
            "FILE_FAKE": 1, "VOICE_SOURCE_ID": "voice_a", "MUSIC_SOURCE_ID": "",
        },
        {
            "ID": "music_pair_real_voice", "VOICE_FAKE": 0, "MUSIC_FAKE": 0,
            "FILE_FAKE": 0, "VOICE_SOURCE_ID": "voice_b", "MUSIC_SOURCE_ID": "music_b",
        },
        {
            "ID": "music_pair_fake_voice", "VOICE_FAKE": 1, "MUSIC_FAKE": 0,
            "FILE_FAKE": 1, "VOICE_SOURCE_ID": "voice_c", "MUSIC_SOURCE_ID": "music_b",
        },
        {
            "ID": "voice_pair_real_music", "VOICE_FAKE": 0, "MUSIC_FAKE": 0,
            "FILE_FAKE": 0, "VOICE_SOURCE_ID": "voice_d", "MUSIC_SOURCE_ID": "music_c",
        },
        {
            "ID": "voice_pair_fake_music", "VOICE_FAKE": 0, "MUSIC_FAKE": 1,
            "FILE_FAKE": 1, "VOICE_SOURCE_ID": "voice_d", "MUSIC_SOURCE_ID": "music_d",
        },
    ])
    count = len(truth)
    bank = Bank(
        name="unit", channel="clean", ids=truth.ID.to_numpy(),
        eat=np.zeros((count, 1), dtype=np.float32),
        spear=np.zeros((count, 1), dtype=np.float32),
        eat_mask=np.ones((count, 1), dtype=bool),
        spear_mask=np.ones((count, 1), dtype=bool),
        targets=np.zeros((count, 3), dtype=np.float32),
        joint=np.zeros(count, dtype=np.int64), truth=truth,
    )

    dataset = CounterfactualDataset([bank])

    assert len(dataset) == count
    assert dataset.music_mask[:2].tolist() == [0.0, 0.0]
    assert dataset.voice_mask[:2].tolist() == [0.0, 0.0]
    assert dataset.music_mask[2:4].tolist() == [1.0, 1.0]
    assert dataset.voice_mask[4:6].tolist() == [1.0, 1.0]


def test_asymmetric_channel_pairing_updates_children_only():
    truth = pd.DataFrame([
        {
            "ID": "clean", "PARENT_ID": "", "VOICE_FAKE": 1,
            "MUSIC_FAKE": 0, "FILE_FAKE": 1,
        },
        {
            "ID": "codec_a", "PARENT_ID": "clean", "VOICE_FAKE": 1,
            "MUSIC_FAKE": 0, "FILE_FAKE": 1,
        },
        {
            "ID": "codec_b", "PARENT_ID": "clean", "VOICE_FAKE": 1,
            "MUSIC_FAKE": 0, "FILE_FAKE": 1,
        },
    ])
    count = len(truth)
    bank = Bank(
        name="unit", channel="clean", ids=truth.ID.to_numpy(),
        eat=np.zeros((count, 1), dtype=np.float32),
        spear=np.zeros((count, 1), dtype=np.float32),
        eat_mask=np.ones((count, 1), dtype=bool),
        spear_mask=np.ones((count, 1), dtype=bool),
        targets=np.zeros((count, 3), dtype=np.float32),
        joint=np.zeros(count, dtype=np.int64), truth=truth,
    )

    symmetric = CounterfactualDataset([bank])
    asymmetric = CounterfactualDataset([bank], asymmetric_channel=True)

    assert symmetric.channel_mask.tolist() == [1.0, 1.0, 1.0]
    assert asymmetric.channel_mask.tolist() == [0.0, 1.0, 1.0]
    assert asymmetric.channel_partner.tolist() == [0, 0, 0]
