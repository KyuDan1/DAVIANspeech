import ast
from pathlib import Path

import pytest

from scripts.build_v50_three_stream_v57 import patch_entrypoint


ROOT = Path(__file__).resolve().parents[1]


def test_v50_patch_runs_one_task_after_all_exact_v50_experts():
    source = (ROOT / "svwpt_v50_pkg" / "script.py").read_text("utf-8")
    patched = patch_entrypoint(source, "file")
    ast.parse(patched)
    assert patched.count("apply_three_stream_anchor_residual(") == 1
    assert 'task_mode="file"' in patched
    assert patched.index("apply_sparse_call_consensus(") < patched.index(
        "apply_three_stream_anchor_residual("
    )
    assert "xlsr_windows, spectra_stats" in patched


def test_v50_patch_rejects_non_isolated_mode():
    source = (ROOT / "svwpt_v50_pkg" / "script.py").read_text("utf-8")
    with pytest.raises(ValueError, match="task_mode"):
        patch_entrypoint(source, "full")
