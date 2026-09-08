import sys
from pathlib import Path
import zipfile


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from build_mixfake_file_moe_v99 import BASE, NEW_ENTRIES, patched_script  # noqa: E402


def test_v99_patch_is_single_and_after_music_replacement():
    with zipfile.ZipFile(BASE) as handle:
        original = handle.read("script.py").decode("utf-8")
    changed = patched_script(original)
    assert changed.count("from mixfake_multistream_file_inference_v99 import") == 1
    assert changed.count("apply_mixfake_file_moe(") == 1
    assert changed.index("apply_music_only(") < changed.index("apply_mixfake_file_moe(")
    assert "weight=0.40, batch_size=12, views=3" in changed


def test_v99_new_members_are_within_model_tree():
    assert set(NEW_ENTRIES) == {
        "model/src/multistream_prompt_spectra.py",
        "model/src/mixfake_multistream_file_inference_v99.py",
        "model/mixfake-file-v99/multistream_prompt.pt",
    }
    assert all(path.is_file() for path in NEW_ENTRIES.values())
