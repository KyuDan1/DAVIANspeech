import csv
from pathlib import Path
import zipfile

import numpy as np
import pytest

from scripts.build_sparse_voice_v49_submission import (
    inspect_archive,
    make_zip,
    patch_entrypoint,
    patch_pipeline,
)
from src.sparse_call_consensus import apply_sparse_call_consensus
from src.spectra_aasist_detector import apply_spectra_voice_fusion


ROOT = Path(__file__).resolve().parents[1]
UNTOUCHED_COLUMNS = (
    "FILE_FAKE_PROB",
    "MUSIC_FAKE_PROB",
    "VOICE_PRESENT_PROB",
    "MUSIC_PRESENT_PROB",
)


def test_exact_v47_patch_reuses_one_xlsr_pass_and_zeroes_all_file_updates():
    pipeline_source = (
        ROOT / "component_query_or_v47/model/src/pipeline.py"
    ).read_text("utf-8")
    entry_source = (ROOT / "component_query_or_v47/script.py").read_text("utf-8")
    pipeline = patch_pipeline(pipeline_source)
    entrypoint = patch_entrypoint(entry_source)

    compile(pipeline, "pipeline.py", "exec")
    compile(entrypoint, "script.py", "exec")
    assert pipeline.count("XlsrAntiDeepfake.from_checkpoint") == 1
    assert pipeline.count("profile=xlsr_profile") == 1
    assert pipeline.count("spectra_scorer.add(path.stem, voice_audio)") == 1
    assert pipeline.count("collect_sliding=True") == 1
    assert entrypoint.count("apply_spectra_voice_fusion(") == 1
    assert entrypoint.count("apply_sparse_call_consensus(") == 1
    assert "voice_weight=0.10, file_weight=0.0" in entrypoint
    assert "voice_weight=0.60" in entrypoint
    assert "file_voice_only_weight=0.0" in entrypoint

    # The patch is tied to exact v47 and cannot be accidentally applied twice.
    with pytest.raises(ValueError, match="marker changed"):
        patch_pipeline(pipeline)
    with pytest.raises(ValueError, match="marker changed"):
        patch_entrypoint(entrypoint)


def _read_single_row(path: Path) -> dict[str, str]:
    with path.open(encoding="utf-8", newline="") as handle:
        return next(csv.DictReader(handle))


def test_frozen_spectra_then_sparse_consensus_mutates_voice_only(tmp_path: Path):
    submission = tmp_path / "submission.csv"
    original = {
        "ID": "sample",
        "FILE_FAKE_PROB": "0.4123456789",
        "VOICE_FAKE_PROB": "0.3123456789",
        "MUSIC_FAKE_PROB": "0.2123456789",
        "VOICE_PRESENT_PROB": "0.9123456789",
        "MUSIC_PRESENT_PROB": "0.1123456789",
    }
    with submission.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(original))
        writer.writeheader()
        writer.writerow(original)

    spectra_fixed = tmp_path / "spectra-fixed.npz"
    np.savez(
        spectra_fixed,
        ids=np.asarray(["sample"]),
        fake_margin=np.asarray([1.25], dtype=np.float32),
        valid=np.asarray([True], dtype=np.bool_),
    )
    xlsr_sliding = tmp_path / "xlsr-sliding.npz"
    np.savez(
        xlsr_sliding,
        ids=np.asarray(["sample"]), offsets=np.asarray([0, 2]),
        starts=np.asarray([0, 64_000]),
        scores=np.asarray([0.3, 0.8], dtype=np.float32),
        durations=np.asarray([8.0], dtype=np.float32),
        window=np.asarray(64_000, dtype=np.int64),
    )
    spectra_sliding = tmp_path / "spectra-sliding.npz"
    np.savez(
        spectra_sliding,
        ids=np.asarray(["sample"]), offsets=np.asarray([0, 2]),
        starts=np.asarray([0, 64_600]),
        fake_margins=np.asarray([-0.5, 1.5], dtype=np.float32),
        durations=np.asarray([8.0], dtype=np.float32),
        valid=np.asarray([True], dtype=np.bool_),
        window=np.asarray(64_600, dtype=np.int64),
    )
    head = tmp_path / "head.npz"
    np.savez(
        head,
        feature_mean=np.zeros(4), feature_scale=np.ones(4),
        local_weight=np.asarray([1.0, 0.5, 0.0, 0.0]),
        local_bias=np.asarray(0.0), pool_temperature=np.asarray(2.0),
        bag_weight=np.asarray(1.0), bag_bias=np.asarray(0.0),
    )

    apply_spectra_voice_fusion(
        submission, spectra_fixed,
        voice_weight=0.10, file_weight=0.0,
    )
    apply_sparse_call_consensus(
        submission, xlsr_sliding, spectra_sliding, head,
        voice_weight=0.60, file_voice_only_weight=0.0,
    )
    updated = _read_single_row(submission)
    assert updated["VOICE_FAKE_PROB"] != original["VOICE_FAKE_PROB"]
    for column in UNTOUCHED_COLUMNS:
        assert updated[column] == original[column]


def _write_minimal_package_zip(path: Path, extra=()) -> None:
    with zipfile.ZipFile(path, "w") as handle:
        handle.writestr("model/src/pipeline.py", "pass\n")
        handle.writestr("script.py", "pass\n")
        handle.writestr("requirements.txt", "")
        for name, value in extra:
            handle.writestr(name, value)


@pytest.mark.parametrize(
    "unsafe", ["../escape", "/absolute", "bad\\path", "./noncanonical"]
)
def test_archive_inspection_rejects_unsafe_members(tmp_path: Path, unsafe: str):
    archive = tmp_path / "unsafe.zip"
    _write_minimal_package_zip(archive, [(unsafe, "bad")])
    with pytest.raises(ValueError, match="unsafe ZIP member"):
        inspect_archive(archive)


def test_archive_inspection_rejects_duplicate_members(tmp_path: Path):
    archive = tmp_path / "duplicate.zip"
    with pytest.warns(UserWarning, match="Duplicate name"):
        _write_minimal_package_zip(
            archive, [("model/src/pipeline.py", "replaced")]
        )
    with pytest.raises(ValueError, match="duplicate ZIP member"):
        inspect_archive(archive)


def test_archive_is_wrapper_free_and_entrypoint_cleans_all_private_stats(
    tmp_path: Path,
):
    package = tmp_path / "package"
    (package / "model").mkdir(parents=True)
    (package / "model/weight.bin").write_bytes(b"weight")
    (package / "script.py").write_text("pass\n", encoding="utf-8")
    (package / "requirements.txt").write_text("", encoding="utf-8")
    archive = tmp_path / "submission.zip"
    make_zip(package, archive)
    inspect_archive(archive)
    with zipfile.ZipFile(archive) as handle:
        assert set(name.split("/", 1)[0] for name in handle.namelist()) == {
            "model", "script.py", "requirements.txt",
        }

    entrypoint = patch_entrypoint(
        (ROOT / "component_query_or_v47/script.py").read_text("utf-8")
    )
    cleanup = entrypoint[entrypoint.index("    for path in (\n"):]
    for name in ("spectra_stats", "xlsr_consensus", "spectra_consensus"):
        assert name in cleanup
    assert "path.unlink(missing_ok=True)" in cleanup
