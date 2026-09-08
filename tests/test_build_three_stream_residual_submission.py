from pathlib import Path

import pytest

from scripts.build_three_stream_residual_submission import patch_entrypoint


ROOT = Path(__file__).resolve().parents[1]


def test_exact_v47_entrypoint_receives_one_residual_and_cache_cleanup():
    source = (ROOT / "component_query_or_v47/script.py").read_text("utf-8")
    patched = patch_entrypoint(source)
    assert patched.count("from three_stream_anchor_residual_inference import") == 1
    assert patched.count("apply_three_stream_anchor_residual(") == 1
    assert patched.count("args.xlsr_window_embeddings_output = xlsr_windows") == 1
    assert "spear_component_bins, xlsr_windows" in patched


def test_entrypoint_patcher_refuses_already_patched_or_unknown_source():
    source = (ROOT / "component_query_or_v47/script.py").read_text("utf-8")
    with pytest.raises(ValueError, match="marker changed"):
        patch_entrypoint(patch_entrypoint(source))
    with pytest.raises(ValueError, match="marker changed"):
        patch_entrypoint("def main():\n    pass\n")
