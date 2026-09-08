from pathlib import Path

from scripts.build_exact_v41_file_branch_submission import patch_entrypoint


ROOT = Path(__file__).resolve().parents[1]


def test_exact_branch_reuses_base_encoders_and_changes_only_file():
    source = (ROOT / "component_query_or_v47/script.py").read_text("utf-8")
    patched = patch_entrypoint(
        source, spectra_subdir="spectra-aasist", branch_weight=.70,
    )
    compile(patched, "script.py", "exec")
    assert patched.count("run(args)") == 1
    assert patched.count("apply_eat_presence_fusion(") == 1
    assert patched.count("apply_fusion_with_stats(") == 1
    assert patched.count("apply_wpt_fixed_moe_fusion(") == 1
    assert patched.count("copy2(args.output, v41_branch)") == 1
    assert "apply_file_submission_blend(" in patched
    assert "branch_weight=0.7" in patched
    assert 'BASE_DIR / "model" / "spectra-aasist"' in patched
    assert "additional_temporal_bin_requests" in patched


def test_exact_branch_patch_is_compatible_with_sparse_voice_entrypoint():
    from scripts.build_sparse_voice_v49_submission import patch_entrypoint as sparse

    source = (ROOT / "component_query_or_v47/script.py").read_text("utf-8")
    patched = patch_entrypoint(
        sparse(source), spectra_subdir="spectra-aasist", branch_weight=.50,
    )
    compile(patched, "script.py", "exec")
    assert patched.count("apply_spectra_voice_fusion(") == 1
    assert patched.count("apply_sparse_call_consensus(") == 1
    assert patched.count("apply_file_submission_blend(") == 1
    assert "file_weight=0.0" in patched
    assert "file_voice_only_weight=0.0" in patched
