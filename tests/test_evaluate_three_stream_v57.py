from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts.evaluate_three_stream_v57 import (
    add_diagnostic_axes,
    candidate_decision,
    cell_operating_rows,
    expert_export,
    full_residual_predictions,
    join_predictions,
    task_isolated_predictions,
)


def test_diagnostic_axes_keep_cell_layout_channel_and_both_generators():
    frame = pd.DataFrame([
        {
            "ID": "a", "VOICE_PRESENT": 1, "MUSIC_PRESENT": 1,
            "VOICE_FAKE": 0, "MUSIC_FAKE": 1,
            "MIX_MODE": "partial_overlap", "CHANNEL": "g711",
            "FIRST_GENERATOR": "tts_a", "SECOND_GENERATOR": "tts_b",
            "MUSIC_GENERATOR": "suno",
        },
    ])
    output = add_diagnostic_axes(frame)
    assert output.loc[0, "CELL_V57"] == "RF"
    assert output.loc[0, "LAYOUT_V57"] == "partial_overlap"
    assert output.loc[0, "CHANNEL_V57"] == "g711"
    assert output.loc[0, "VOICE_GENERATOR_V57"] == "tts_a+tts_b"
    assert output.loc[0, "MUSIC_GENERATOR_V57"] == "suno"


def test_prediction_join_requires_exact_development_keys():
    truth = pd.DataFrame({"DATASET": ["d"], "ID": ["a"]})
    prediction = pd.DataFrame({"DATASET": ["d"], "ID": ["b"]})
    with pytest.raises(ValueError, match="exactly match"):
        join_predictions(truth, prediction)


def test_task_isolation_changes_only_requested_output_column():
    anchor = pd.DataFrame([{
        "DATASET": "d", "ID": "a", "FILE_FAKE_PROB": 0.2,
        "VOICE_FAKE_PROB": 0.3, "MUSIC_FAKE_PROB": 0.4,
        "VOICE_PRESENT_PROB": 0.8, "MUSIC_PRESENT_PROB": 0.7,
    }])
    candidate = anchor.assign(
        VOICE_LOGIT_RESIDUAL=0.5,
        MUSIC_LOGIT_RESIDUAL=-0.25,
        DIRECT_FILE_LOGIT_RESIDUAL=1.0,
    )
    definitions = {
        "voice": "VOICE_FAKE_PROB",
        "music": "MUSIC_FAKE_PROB",
        "file": "FILE_FAKE_PROB",
    }
    for task, changed in definitions.items():
        result = task_isolated_predictions(anchor, candidate, task)
        unchanged = [column for column in anchor if column not in {
            "DATASET", "ID", changed,
        }]
        pd.testing.assert_frame_equal(
            result[unchanged], anchor[unchanged], check_exact=True,
        )
        assert result.loc[0, changed] != anchor.loc[0, changed]


def test_expert_export_keeps_residuals_and_applies_file_coefficient():
    anchor = pd.DataFrame([{
        "DATASET": "d", "ID": "a", "FILE_FAKE_PROB": 0.5,
        "VOICE_FAKE_PROB": 0.5, "MUSIC_FAKE_PROB": 0.5,
        "VOICE_PRESENT_PROB": 0.8, "MUSIC_PRESENT_PROB": 0.7,
    }])
    candidate = anchor.assign(
        VOICE_LOGIT_RESIDUAL=0.5,
        MUSIC_LOGIT_RESIDUAL=-0.25,
        DIRECT_FILE_LOGIT_RESIDUAL=1.0,
    )
    output = expert_export(anchor, candidate)
    assert output.loc[0, "ISOLATED_VOICE_EXPERT_LOGIT"] == pytest.approx(0.5)
    assert output.loc[0, "ISOLATED_MUSIC_EXPERT_LOGIT"] == pytest.approx(-0.25)
    assert output.loc[0, "ISOLATED_FILE_EXPERT_LOGIT"] == pytest.approx(0.7)


def test_full_residual_can_be_rebased_on_a_different_anchor():
    anchor = pd.DataFrame([{
        "DATASET": "d", "ID": "a", "FILE_FAKE_PROB": 0.5,
        "VOICE_FAKE_PROB": 0.5, "MUSIC_FAKE_PROB": 0.5,
        "VOICE_PRESENT_PROB": 0.8, "MUSIC_PRESENT_PROB": 0.7,
    }])
    candidate = anchor.assign(
        VOICE_LOGIT_RESIDUAL=0.0, MUSIC_LOGIT_RESIDUAL=0.0,
        DIRECT_FILE_LOGIT_RESIDUAL=1.0,
    )
    output = full_residual_predictions(anchor, candidate)
    assert output.loc[0, "FILE_FAKE_PROB"] == pytest.approx(
        1 / (1 + np.exp(-0.7))
    )
    assert output.loc[0, "VOICE_FAKE_PROB"] == 0.5
    assert output.loc[0, "MUSIC_FAKE_PROB"] == 0.5


def test_cells_use_global_task_thresholds_instead_of_undefined_cell_eer():
    rows = []
    for cell, labels, scores in (
        ("RR", (0, 0, 0), (.1, .1, .1)),
        ("RF", (1, 0, 1), (.9, .1, .9)),
        ("FR", (1, 1, 0), (.9, .9, .1)),
        ("FF", (1, 1, 1), (.9, .9, .9)),
    ):
        for index in range(2):
            rows.append({
                "CELL_V57": cell, "FILE_FAKE": labels[0],
                "VOICE_FAKE": labels[1], "MUSIC_FAKE": labels[2],
                "VOICE_PRESENT": 1, "MUSIC_PRESENT": 1,
                "FILE_FAKE_PROB": scores[0], "VOICE_FAKE_PROB": scores[1],
                "MUSIC_FAKE_PROB": scores[2],
            })
    output = cell_operating_rows("perfect", pd.DataFrame(rows))
    assert {row["CELL"] for row in output} == {"RR", "RF", "FR", "FF"}
    assert all(row["WEIGHTED_CELL_ERROR"] == 0 for row in output)
