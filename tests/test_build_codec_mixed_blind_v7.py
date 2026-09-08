from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import build_codec_mixed_blind_v7 as v7  # noqa: E402


def voice_rows(generators: int = 11, per_generator: int = 10) -> list[dict]:
    rows = []
    for generator in range(generators):
        for index in range(per_generator):
            token = f"{generator:02d}_{index:02d}"
            rows.append({
                "generator": f"g{generator:02d}", "utt_id": f"u_{token}",
                "speaker": f"s_{token}", "reference_speaker": f"r_{token}",
                "content_id": f"c_{token}", "reference_id": f"x_{token}",
            })
    return rows


def test_balanced_voice_take_is_deterministic_diverse_and_identity_unique():
    rows = voice_rows()
    first = v7.balanced_voice_take(rows, 36, 123)
    second = v7.balanced_voice_take(rows, 36, 123)
    assert [row["utt_id"] for row in first] == [row["utt_id"] for row in second]
    counts = pd.Series([row["generator"] for row in first]).value_counts()
    assert len(counts) == 11
    assert counts.max() - counts.min() <= 1
    speakers = [value for row in first for value in (row["speaker"], row["reference_speaker"])]
    contents = [value for row in first for value in (row["content_id"], row["reference_id"])]
    assert len(speakers) == len(set(speakers))
    assert len(contents) == len(set(contents))


def test_balanced_music_take_keeps_all_generators_and_unique_songs():
    rows = []
    for generator in v7.FAKE_MUSIC_GENERATORS:
        for index in range(20):
            rows.append({
                "generator": generator, "group": f"song:{generator}:{index}",
                "source_id": f"src:{generator}:{index}",
                "semantic_status": v7.PENDING_SEMANTIC,
            })
    selected = v7.balanced_music_take(rows, 36, 456)
    counts = pd.Series([row["generator"] for row in selected]).value_counts()
    assert set(counts.index) == set(v7.FAKE_MUSIC_GENERATORS)
    assert counts.max() - counts.min() <= 1
    assert len({row["group"] for row in selected}) == 36


@pytest.mark.parametrize("dataset,generator", [("SONICS", "suno"), ("SONICS", "udio")])
def test_suno_and_udio_are_never_allowed_generators(dataset, generator):
    assert dataset == "SONICS"
    assert generator not in v7.FAKE_MUSIC_GENERATORS


def test_semantic_input_rejects_authenticity_predictions(tmp_path: Path):
    path = tmp_path / "semantic.csv"
    pd.DataFrame({
        "MUSIC_SOURCE_ID": ["m1"], "ACOUSTIC_SCREEN_PASS": [True],
        "PANNS_THRESHOLD": [0.20], "DEMUCS_THRESHOLD_DB": [-1.5],
        "MUSIC_FAKE_PROB": [0.8],
    }).to_csv(path, index=False)
    with pytest.raises(ValueError, match="authenticity/scoring columns forbidden"):
        v7.merge_semantic_scores([path])


def test_protection_includes_historical_source_and_speaker_columns(tmp_path: Path):
    truth = tmp_path / "truth.csv"
    hashes = tmp_path / "source_hashes.csv"
    pd.DataFrame({
        "SOURCE_ID": ["common_voice_en_1.mp3"],
        "SPEAKER_ID": ["speaker-secret"],
        "AUDIO_TYPE": ["voice"],
    }).to_csv(truth, index=False)
    pd.DataFrame({"SHA256": ["ab" * 32]}).to_csv(hashes, index=False)
    # build_protection records paths relative to ROOT in production.  Use repo
    # scratch paths here so the same escape check remains exercised.
    local = ROOT / "scratch" / "v7_test_protection"
    local.mkdir(parents=True, exist_ok=True)
    local_truth = local / "truth.csv"
    local_hashes = local / "source_hashes.csv"
    pd.read_csv(truth).to_csv(local_truth, index=False)
    pd.read_csv(hashes).to_csv(local_hashes, index=False)
    try:
        result = v7.build_protection([local_truth], [local_hashes])
        assert "common_voice_en_1.mp3" in result.exact
        assert "common_voice_en_1" in result.exact
        assert "speaker-secret" in result.exact
        assert "ab" * 32 in result.audio_hashes
    finally:
        local_truth.unlink(missing_ok=True)
        local_hashes.unlink(missing_ok=True)
        local.rmdir()


def valid_reservation() -> pd.DataFrame:
    fake_voice_generators = [f"tts{i:02d}" for i in range(11)]
    fake_voice_assignments = [
        generator
        for index, generator in enumerate(fake_voice_generators)
        for _ in range(4 if index < 3 else 3)
    ]
    fake_music_assignments = [
        generator
        for index, generator in enumerate(v7.FAKE_MUSIC_GENERATORS)
        for _ in range(8 if index == 0 else 7)
    ]
    voice_offset = {0: 0, 1: 0}
    music_offset = {0: 0, 1: 0}
    records = []
    for mode in v7.MODES:
        for voice_fake, music_fake in v7.CELLS:
            for _ in range(6):
                vindex = voice_offset[voice_fake]
                mindex = music_offset[music_fake]
                voice_offset[voice_fake] += 1
                music_offset[music_fake] += 1
                records.append({
                    "BASE_ID": f"b{len(records):03d}", "MIX_MODE": mode,
                    "VOICE_FAKE": voice_fake, "MUSIC_FAKE": music_fake,
                    "FILE_FAKE": int(voice_fake or music_fake),
                    "VOICE_SOURCE_ID": f"v{voice_fake}_{vindex}",
                    "MUSIC_SOURCE_ID": f"m{music_fake}_{mindex}",
                    "MUSIC_GROUP_ID": f"song:{music_fake}:{mindex}",
                    "VOICE_SPEAKER": f"speaker:{voice_fake}:{vindex}",
                    "VOICE_REFERENCE_SPEAKER": (
                        f"reference:{vindex}" if voice_fake else ""
                    ),
                    "VOICE_GENERATOR": (
                        fake_voice_assignments[vindex] if voice_fake else "bonafide"
                    ),
                    "MUSIC_GENERATOR": (
                        fake_music_assignments[mindex] if music_fake else "FMA"
                    ),
                    "MUSIC_VOCAL_SCREEN": v7.SEMANTIC_PASS,
                })
    return pd.DataFrame(records)


def test_validate_reservation_accepts_exact_72_base_contract():
    v7.validate_reservation(valid_reservation(), per_cell=6, require_semantic_pass=True)


def test_validate_reservation_rejects_suno_vocal_as_music():
    frame = valid_reservation()
    frame.loc[frame.MUSIC_FAKE.eq(1).idxmax(), "MUSIC_GENERATOR"] = "suno"
    with pytest.raises(ValueError, match="generator imbalance|Suno"):
        v7.validate_reservation(frame, per_cell=6, require_semantic_pass=True)
