from __future__ import annotations

from pathlib import Path
import json
import sys

import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import build_all_type_blind_v9 as v9  # noqa: E402


RESERVATION = ROOT / "reports/all_type_blind_v9_design/reservation_initial/reservation.csv"


def test_initial_reservation_has_exact_all_type_contract():
    frame = pd.read_csv(RESERVATION, dtype=str).fillna("")
    v9.validate_reservation(frame)
    assert frame.AUDIO_TYPE.value_counts().to_dict() == {
        "mixed": 72, "voice": 24, "music": 24,
    }
    assert len(frame) * len(v9.CHANNELS) == 600


def test_joint_suno_udio_is_ff_concurrent_and_never_rf():
    frame = pd.read_csv(RESERVATION, dtype=str).fillna("")
    joint = frame.MUSIC_SOURCE_KIND.eq(v9.JOINT_KIND)
    assert joint.sum() == 6
    assert frame.loc[joint, "COMPONENT_CASE"].eq("FF").all()
    assert frame.loc[joint, "MIX_MODE"].eq("concurrent").all()
    assert not frame.loc[frame.COMPONENT_CASE.eq("RF"), "MUSIC_PLATFORM"].isin({"suno", "udio"}).any()


def test_suno_udio_in_rf_is_rejected():
    frame = pd.read_csv(RESERVATION, dtype=str).fillna("")
    index = frame.index[frame.COMPONENT_CASE.eq("RF")][0]
    frame.loc[index, "MUSIC_PLATFORM"] = "suno"
    with pytest.raises(ValueError, match="leaked outside"):
        v9.validate_reservation(frame)


def test_semantic_loader_rejects_authenticity_columns(tmp_path: Path):
    path = tmp_path / "semantic.csv"
    pd.DataFrame({"MUSIC_SOURCE_ID": ["m"], "SEMANTIC_SCREEN_PASS": [True],
                  "MUSIC_FAKE_PROB": [0.9]}).to_csv(path, index=False)
    with pytest.raises(ValueError, match="authenticity/scoring columns forbidden"):
        v9.load_semantic([path])


def test_protection_snapshot_includes_prior_rendered_audio_hashes():
    _, _, snapshot, provenance = v9.load_reservation(RESERVATION.parent)
    protected = set(snapshot.PATH)
    assert "data/eval/codec_mixed_blind_v8/audio_hashes.csv" in protected
    assert provenance["authenticity_detector_inference"] is False
    assert provenance["score_open_count"] == 0


def test_builder_does_not_import_authenticity_detector_or_score():
    source = (ROOT / "scripts/build_all_type_blind_v9.py").read_text("utf-8")
    assert "artifactnet_detector" not in source
    assert "AntiDeepfakeDetector" not in source
    assert "score_frame(" not in source
    assert "roc_curve(" not in source


def test_identity_check_excludes_generator_and_platform_family_labels():
    assert "VOICE_GENERATOR" not in v9.IDENTITY_COLUMNS
    assert "MUSIC_GENERATOR" not in v9.IDENTITY_COLUMNS
    assert "MUSIC_PLATFORM" not in v9.IDENTITY_COLUMNS
    assert {"VOICE_SOURCE_ID", "VOICE_SPEAKER", "MUSIC_SOURCE_ID", "MUSIC_GROUP_ID"} <= set(v9.IDENTITY_COLUMNS)


def test_final_source_and_bank_validations_are_locked_unscored():
    source = json.loads((ROOT / "reports/all_type_blind_v9_design/source_validation/validation.json").read_text("utf-8"))
    bank = json.loads((ROOT / "reports/all_type_blind_v9_design/bank_validation/validation.json").read_text("utf-8"))
    registration = json.loads((ROOT / "reports/all_type_blind_v9_design/registration.json").read_text("utf-8"))
    assert source["status"] == "PASS"
    assert bank["status"] == "PASS" and bank["checks_passed"] == bank["checks_total"] == 14
    assert registration["status"] == "locked_unscored"
    assert registration["authenticity_detector_inference"] is False
    assert registration["truth_open_count"] == registration["score_open_count"] == 0
    truth = ROOT / registration["truth_path"]
    assert v9.sha256_file(truth) == registration["truth_sha256"]
    assert registration["truth_path"] in (ROOT / "configs/data_partitions.yaml").read_text("utf-8")
