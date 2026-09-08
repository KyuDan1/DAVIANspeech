import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from evaluate_mixfake_multistream_prompt_v94 import (  # noqa: E402
    select_label, task_mask,
)


def test_joint_labels_fill_absent_only_for_storage_and_masks_exclude_them():
    frame = pd.DataFrame({
        "VOICE_FAKE": [1.0, np.nan, 0.0],
        "MUSIC_FAKE": [np.nan, 1.0, 0.0],
        "FILE_FAKE": [1, 1, 0],
        "VOICE_PRESENT": [1, 0, 1],
        "MUSIC_PRESENT": [0, 1, 1],
    })
    labels = select_label(frame, "all")
    assert labels.tolist() == [[1, 0, 1], [0, 1, 1], [0, 0, 0]]
    assert task_mask(frame, "voice").tolist() == [True, False, True]
    assert task_mask(frame, "music").tolist() == [False, True, True]
    assert task_mask(frame, "file").tolist() == [True, True, True]
