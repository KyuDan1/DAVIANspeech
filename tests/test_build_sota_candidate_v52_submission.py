import csv
from pathlib import Path

from scripts.build_component_consistent_file_v51_submission import patch_entrypoint
from scripts.build_sota_candidate_v52_submission import (
    validate_execution_order,
    validate_music_base,
)
from src.component_consistent_file_fusion import (
    apply_component_consistent_file_fusion,
)


ROOT = Path(__file__).resolve().parents[1]
COLUMNS = [
    "ID", "FILE_FAKE_PROB", "VOICE_FAKE_PROB", "MUSIC_FAKE_PROB",
    "VOICE_PRESENT_PROB", "MUSIC_PRESENT_PROB",
]


def test_frozen_music_base_assets_match_configured_hashes():
    validate_music_base(ROOT / "music_only_v51_frozen")


def test_v52_execution_order_is_music_then_voice_then_final_file():
    base = (ROOT / "music_only_v51_frozen/script.py").read_text("utf-8")
    patched = patch_entrypoint(base)
    compile(patched, "script.py", "exec")
    validate_execution_order(patched)


def test_v52_file_stage_preserves_the_other_four_outputs_exactly(tmp_path):
    path = tmp_path / "submission.csv"
    original = {
        "ID": "mixed",
        "FILE_FAKE_PROB": "0.2000000001",
        "VOICE_FAKE_PROB": "0.8500000002",
        "MUSIC_FAKE_PROB": "0.3000000003",
        "VOICE_PRESENT_PROB": "0.8000000004",
        "MUSIC_PRESENT_PROB": "0.9000000005",
    }
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerow(original)
    apply_component_consistent_file_fusion(
        path,
        file_weight=0.20,
        voice_presence_threshold=0.10,
        music_presence_threshold=0.20,
    )
    with path.open(newline="", encoding="utf-8") as handle:
        final = next(csv.DictReader(handle))
    assert final["FILE_FAKE_PROB"] != original["FILE_FAKE_PROB"]
    for column in COLUMNS[2:]:
        assert final[column] == original[column]
