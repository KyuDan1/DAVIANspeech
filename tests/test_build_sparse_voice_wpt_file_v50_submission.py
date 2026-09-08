from pathlib import Path

import pytest

from scripts.build_sparse_voice_v49_submission import patch_entrypoint as sparse_patch
from scripts.build_sparse_voice_wpt_file_v50_submission import patch_entrypoint


ROOT = Path(__file__).resolve().parents[1]


def test_v50_adds_one_original_audio_wpt_pass_and_file_only_weight():
    v47 = (ROOT / "component_query_or_v47/script.py").read_text("utf-8")
    source = sparse_patch(v47)
    patched = patch_entrypoint(source)
    compile(patched, "script.py", "exec")
    assert patched.count("apply_wpt_file_expert(") == 1
    assert patched.count("run(args)") == 1
    assert "file_views=5, file_temperature=2.0" in patched
    assert "wpt_weight=0.4" in patched
    assert 'BASE_DIR / "model" / "spectra-aasist"' in patched
    assert "voice_weight=0.60" in patched
    assert "file_voice_only_weight=0.0" in patched


def test_v50_weight_guard_and_double_patch():
    source = sparse_patch(
        (ROOT / "component_query_or_v47/script.py").read_text("utf-8")
    )
    with pytest.raises(ValueError, match="limits WPT File weight"):
        patch_entrypoint(source, file_weight=.51)
    patched = patch_entrypoint(source)
    with pytest.raises(ValueError, match="marker changed"):
        patch_entrypoint(patched)
