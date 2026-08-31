from collections import Counter

import numpy as np

from scripts.build_factorial_eval_1200 import component_case, plan_rows
from scripts.build_yue_cross_component_audit import (
    SR,
    concurrent,
    partial,
    sequential,
)


def test_factorial_plan_has_exact_competition_size_and_split_balance():
    plan = plan_rows()
    assert len(plan) == 1200
    assert Counter(row[0] for row in plan) == {
        "dev": 400, "holdout": 400, "locked": 400,
    }
    for split in ("dev", "holdout", "locked"):
        modes = Counter(row[1] for row in plan if row[0] == split)
        assert modes == {
            "voice_only": 50,
            "music_only": 50,
            "concurrent": 100,
            "partial_overlap": 100,
            "sequential": 100,
        }


def test_factorial_plan_balances_component_labels():
    plan = plan_rows()
    voice = Counter(row[2] for row in plan if row[2] is not None)
    music = Counter(row[3] for row in plan if row[3] is not None)
    assert voice == {0: 525, 1: 525}
    assert music == {0: 525, 1: 525}
    for split in ("dev", "holdout", "locked"):
        for mode in ("concurrent", "partial_overlap", "sequential"):
            cells = Counter(
                (row[2], row[3]) for row in plan
                if row[0] == split and row[1] == mode
            )
            assert cells == {(0, 0): 25, (0, 1): 25, (1, 0): 25, (1, 1): 25}


def test_component_cases_are_unambiguous():
    assert component_case(0, None) == "voice_real"
    assert component_case(1, None) == "voice_fake"
    assert component_case(None, 0) == "music_real"
    assert component_case(None, 1) == "music_fake"
    assert component_case(0, 1) == "voice_real__music_fake"
    assert component_case(1, 0) == "voice_fake__music_real"


def test_yue_audit_mixers_have_expected_duration_and_safe_peak():
    voice = 0.1 * np.ones(9 * SR, dtype="float32")
    music = 0.2 * np.ones(7 * SR, dtype="float32")
    outputs = {
        "concurrent": concurrent(voice, music, "a"),
        "partial": partial(voice, music, "b"),
        "sequential": sequential(voice, music, "c"),
    }
    assert outputs["concurrent"].size == 9 * SR
    assert outputs["partial"].size == 14 * SR
    assert outputs["sequential"].size == 16 * SR + SR // 4
    assert all(abs(output).max() <= 0.98 for output in outputs.values())
