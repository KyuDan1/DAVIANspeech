import csv
from pathlib import Path

import numpy as np

from scripts.build_forensics_voice_file_v54_submission import (
    EXTERNAL_ASSETS,
    patch_entrypoint,
    validate_execution_order,
    validate_hashes,
)
from src.component_consistent_file_fusion import (
    apply_component_consistent_file_fusion,
)
from src.forensics_xlsr_wild_fusion import (
    apply_voice_logit_residual,
    fixed_windows,
    mix_voice_logit,
)


ROOT = Path(__file__).resolve().parents[1]
COLUMNS = [
    "ID", "FILE_FAKE_PROB", "VOICE_FAKE_PROB", "MUSIC_FAKE_PROB",
    "VOICE_PRESENT_PROB", "MUSIC_PRESENT_PROB",
]


def test_fixed_windows_are_normalized_evenly_spaced_and_deterministic():
    audio = np.arange(12, dtype=np.float32)
    first = fixed_windows(audio, length=4, count=3)
    second = fixed_windows(audio, length=4, count=3)
    np.testing.assert_array_equal(first, second)
    np.testing.assert_allclose(first[:, 0], np.asarray([0, 4, 8]) / 11)
    assert first.shape == (3, 4)
    assert float(np.abs(first).max()) < 1.0 + 1e-6


def test_short_audio_is_tiled_to_three_equal_views():
    windows = fixed_windows(np.asarray([1.0, -1.0]), length=5, count=3)
    assert windows.shape == (3, 5)
    np.testing.assert_array_equal(windows[0], windows[1])
    np.testing.assert_array_equal(windows[1], windows[2])


def test_voice_then_file_preserves_music_and_cps_bit_exact(tmp_path):
    submission = tmp_path / "submission.csv"
    original = {
        "ID": "mixed",
        "FILE_FAKE_PROB": "0.2000000001",
        "VOICE_FAKE_PROB": "0.8000000002",
        "MUSIC_FAKE_PROB": "0.3000000003",
        "VOICE_PRESENT_PROB": "0.9000000004",
        "MUSIC_PRESENT_PROB": "0.7000000005",
    }
    with submission.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS)
        writer.writeheader(); writer.writerow(original)
    apply_voice_logit_residual(
        submission, np.asarray(["mixed"]), np.asarray([2.0]),
        voice_weight=0.075,
    )
    apply_component_consistent_file_fusion(
        submission, file_weight=0.20,
        voice_presence_threshold=0.10, music_presence_threshold=0.20,
    )
    with submission.open(encoding="utf-8", newline="") as handle:
        final = next(csv.DictReader(handle))
    assert final["VOICE_FAKE_PROB"] != original["VOICE_FAKE_PROB"]
    assert final["FILE_FAKE_PROB"] != original["FILE_FAKE_PROB"]
    for column in ("MUSIC_FAKE_PROB", "VOICE_PRESENT_PROB", "MUSIC_PRESENT_PROB"):
        assert final[column] == original[column]


def test_voice_residual_uses_raw_fake_logit_at_fixed_weight():
    actual = mix_voice_logit(np.asarray([0.5]), np.asarray([2.0]), voice_weight=.075)
    expected = 1.0 / (1.0 + np.exp(-.15))
    np.testing.assert_allclose(actual, [expected], rtol=0, atol=1e-12)


def test_v54_builder_pins_external_assets_and_execution_order():
    validate_hashes(
        ROOT / "models/external/forensics_xlsr_wild",
        EXTERNAL_ASSETS,
        "Forensics",
    )
    source = (ROOT / "sparse_voice_wpt_file_v50_frozen/script.py").read_text("utf-8")
    patched = patch_entrypoint(source)
    compile(patched, "script.py", "exec")
    validate_execution_order(patched)
    assert "voice_weight=0.075" in patched
    assert "file_weight=0.20" in patched
