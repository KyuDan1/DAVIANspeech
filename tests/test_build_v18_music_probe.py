import csv
import hashlib
from pathlib import Path
import zipfile

import pytest

from scripts.build_v18_music_probe import (
    BASE_ARCHIVE_SHA256,
    BASE_EXPANDED_BYTES,
    BASE_MEMBER_COUNT,
    BASE_SCRIPT_SHA256,
    _apply_music_constant_probe,
    inspect_inventory,
    patch_entrypoint,
    verify_prediction_pair,
)


ROOT = Path(__file__).resolve().parents[1]
BASE_ZIP = ROOT / "channel_invariant_moe_v18.zip"


def test_frozen_official_v18_identity_and_inventory():
    assert BASE_ZIP.stat().st_size == 7_907_816_613
    with BASE_ZIP.open("rb") as handle:
        digest = hashlib.file_digest(handle, "sha256").hexdigest()
    assert digest == BASE_ARCHIVE_SHA256
    with zipfile.ZipFile(BASE_ZIP) as handle:
        inventory = inspect_inventory(handle)
        assert inventory["member_count"] == BASE_MEMBER_COUNT
        assert inventory["expanded_bytes"] == BASE_EXPANDED_BYTES
        assert inventory["top_level"] == ["model", "requirements.txt", "script.py"]
        assert inventory["duplicates"] == []
        assert inventory["unsafe_paths"] == []
        assert inventory["symlinks"] == []
        script = handle.read("script.py")
    assert hashlib.sha256(script).hexdigest() == BASE_SCRIPT_SHA256


def test_entrypoint_override_runs_once_after_all_fusions():
    with zipfile.ZipFile(BASE_ZIP) as handle:
        patched = patch_entrypoint(handle.read("script.py"))
    text = patched.decode("utf-8")
    compile(text, "script.py", "exec")
    assert text.count("_apply_music_constant_probe(args.output)") == 1
    assert text.index("_apply_music_constant_probe(args.output)") > text.index(
        "apply_dual_domain_fusion("
    )
    assert text.index("_apply_music_constant_probe(args.output)") < text.index(
        "for path in (eat_stats, spear_stats)"
    )


def test_probe_changes_only_music_field_strings(tmp_path):
    output = tmp_path / "submission.csv"
    before = [
        [
            "ID",
            "FILE_FAKE_PROB",
            "VOICE_FAKE_PROB",
            "MUSIC_FAKE_PROB",
            "VOICE_PRESENT_PROB",
            "MUSIC_PRESENT_PROB",
        ],
        ["a.wav", "0.123456789", "9e-07", "0.991", "0.88", "0.77"],
        ["b,quoted.wav", "0.4", "0.3", "0.2", "0.1", "0.05"],
    ]
    with output.open("w", encoding="utf-8", newline="") as handle:
        csv.writer(handle, lineterminator="\n").writerows(before)

    _apply_music_constant_probe(output)

    with output.open("r", encoding="utf-8", newline="") as handle:
        after = list(csv.reader(handle))
    music_index = before[0].index("MUSIC_FAKE_PROB")
    assert after[0] == before[0]
    for old, new in zip(before[1:], after[1:]):
        assert new[music_index] == "0.5"
        assert new[:music_index] + new[music_index + 1 :] == (
            old[:music_index] + old[music_index + 1 :]
        )


def test_probe_rejects_malformed_csv_without_replacing_source(tmp_path):
    output = tmp_path / "submission.csv"
    original = "ID,FILE_FAKE_PROB\na,0.4\n"
    output.write_text(original, encoding="utf-8")
    with pytest.raises(ValueError, match="MUSIC_FAKE_PROB"):
        _apply_music_constant_probe(output)
    assert output.read_text(encoding="utf-8") == original
    assert not list(tmp_path.glob("*.tmp"))


def test_prediction_pair_requires_other_fields_bit_exact(tmp_path):
    header = (
        "ID,FILE_FAKE_PROB,VOICE_FAKE_PROB,MUSIC_FAKE_PROB,"
        "VOICE_PRESENT_PROB,MUSIC_PRESENT_PROB\n"
    )
    anchor = tmp_path / "anchor.csv"
    probe = tmp_path / "probe.csv"
    anchor.write_text(header + "a,0.1,0.2,0.3,0.4,0.5\n", encoding="utf-8")
    probe.write_text(header + "a,0.1,0.2,0.5,0.4,0.5\n", encoding="utf-8")
    report = verify_prediction_pair(anchor, probe)
    assert report["rows"] == 1
    probe.write_text(header + "a,0.1,0.9,0.5,0.4,0.5\n", encoding="utf-8")
    with pytest.raises(ValueError, match="VOICE_FAKE_PROB"):
        verify_prediction_pair(anchor, probe)


def test_inventory_flags_unsafe_and_duplicate_members(tmp_path):
    archive = tmp_path / "unsafe.zip"
    with zipfile.ZipFile(archive, "w") as handle:
        handle.writestr("script.py", "pass\n")
        handle.writestr("../escape", "x")
        with pytest.warns(UserWarning, match="Duplicate name"):
            handle.writestr("script.py", "pass\n")
    with zipfile.ZipFile(archive) as handle:
        inventory = inspect_inventory(handle)
    assert inventory["duplicates"] == ["script.py"]
    assert inventory["unsafe_paths"] == ["../escape"]
