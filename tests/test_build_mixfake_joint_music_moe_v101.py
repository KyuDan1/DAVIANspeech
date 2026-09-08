import sys
from pathlib import Path
import zipfile


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from build_mixfake_joint_music_moe_v101 import (  # noqa: E402
    BASE, NEW_ENTRIES, patched_script,
)


def test_v101_patch_is_single_and_runs_after_music_anchor():
    with zipfile.ZipFile(BASE) as handle:
        original = handle.read("script.py").decode("utf-8")
    changed = patched_script(original)
    assert changed.count("from mixfake_multistream_file_inference_v99 import") == 1
    assert changed.count("apply_mixfake_joint_music_moe(") == 1
    assert changed.index("apply_music_only(") < changed.index(
        "apply_mixfake_joint_music_moe("
    )
    assert "file_weight=0.40, voice_weight=0.12" in changed
    assert "music_weight=0.28" in changed
    assert "file_or_weight=0.15" in changed
    assert "joint_views=3, music_views=1" in changed


def test_v101_new_members_are_minimal_and_exist():
    assert set(NEW_ENTRIES) == {
        "model/src/multistream_prompt_spectra.py",
        "model/src/mixfake_multistream_file_inference_v99.py",
        "model/mixfake-joint-v101/multistream_prompt.pt",
        "model/mixfake-music-v101/multistream_prompt.pt",
    }
    assert all(path.is_file() for path in NEW_ENTRIES.values())
