from pathlib import Path

import pytest

from scripts.build_music_only_v51_submission import (
    CALL_INSERT,
    CALL_MARKER,
    IMPORT_INSERT,
    IMPORT_MARKER,
    inject_script,
)


def test_inject_script_adds_one_music_only_stage():
    source = "prefix\n" + IMPORT_MARKER + "middle\n" + CALL_MARKER + "suffix\n"
    changed = inject_script(source)
    assert changed.count(IMPORT_INSERT) == 1
    assert changed.count(CALL_INSERT) == 1
    assert "music_weight=0.0375" in changed


@pytest.mark.parametrize("source", [IMPORT_MARKER, CALL_MARKER, ""])
def test_inject_script_rejects_incomplete_base(source):
    with pytest.raises(ValueError, match="marker changed"):
        inject_script(source)


def test_frozen_call_is_music_only_and_reuses_existing_caches():
    assert "eat_patch_graph, spear_component_bins" in CALL_INSERT
    assert "FILE_FAKE_PROB" not in CALL_INSERT
    assert "VOICE_FAKE_PROB" not in CALL_INSERT
