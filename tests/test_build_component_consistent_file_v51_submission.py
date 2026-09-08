from pathlib import Path

import pytest

from scripts.build_component_consistent_file_v51_submission import patch_entrypoint


ROOT = Path(__file__).resolve().parents[1]


def test_v51_runs_once_after_final_voice_specialist_and_before_cleanup():
    source = (ROOT / "sparse_voice_wpt_file_v50_frozen/script.py").read_text("utf-8")
    patched = patch_entrypoint(source)
    compile(patched, "script.py", "exec")
    assert patched.count("apply_component_consistent_file_fusion(") == 1
    assert "file_weight=0.2" in patched
    call = patched.index("    apply_component_consistent_file_fusion(")
    voice = patched.index("    apply_sparse_call_consensus(")
    cleanup = patched.index("    for path in (")
    assert voice < call < cleanup


def test_v51_stays_after_a_later_music_specialist():
    source = (ROOT / "sparse_voice_wpt_file_v50_frozen/script.py").read_text("utf-8")
    cleanup = "    for path in (\n"
    source = source.replace(
        cleanup,
        "    apply_future_music_specialist(args.output)\n" + cleanup,
    )
    patched = patch_entrypoint(source)
    music = patched.index("    apply_future_music_specialist(args.output)")
    file_call = patched.index("    apply_component_consistent_file_fusion(")
    assert music < file_call


def test_v51_guards_weight_and_double_patch():
    source = (ROOT / "sparse_voice_wpt_file_v50_frozen/script.py").read_text("utf-8")
    with pytest.raises(ValueError, match="limits File residual"):
        patch_entrypoint(source, file_weight=.31)
    patched = patch_entrypoint(source)
    with pytest.raises(ValueError, match="marker changed"):
        patch_entrypoint(patched)
