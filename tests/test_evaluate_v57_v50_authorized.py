import pandas as pd
import pytest

from scripts.evaluate_v57_v50_authorized import output_change_audit


def test_change_audit_rejects_a_non_task_output_change():
    base = pd.DataFrame([{
        "DATASET": "d", "ID": "a", "FILE_FAKE_PROB": .1,
        "VOICE_FAKE_PROB": .2, "MUSIC_FAKE_PROB": .3,
        "VOICE_PRESENT_PROB": .4, "MUSIC_PRESENT_PROB": .5,
    }])
    current = base.assign(MUSIC_FAKE_PROB=.6)
    with pytest.raises(RuntimeError, match="unexpected"):
        output_change_audit(base, current, {"VOICE_FAKE_PROB"})


def test_change_audit_accepts_exact_single_column_change():
    base = pd.DataFrame([{
        "DATASET": "d", "ID": "a", "FILE_FAKE_PROB": .1,
        "VOICE_FAKE_PROB": .2, "MUSIC_FAKE_PROB": .3,
        "VOICE_PRESENT_PROB": .4, "MUSIC_PRESENT_PROB": .5,
    }])
    current = base.assign(VOICE_FAKE_PROB=.6)
    audit = output_change_audit(base, current, {"VOICE_FAKE_PROB"})
    assert audit["changed_rows_by_column"]["VOICE_FAKE_PROB"] == 1
    assert audit["other_columns_bit_exact"]
