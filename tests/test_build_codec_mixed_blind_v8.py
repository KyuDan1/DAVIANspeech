from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import build_codec_mixed_blind_v8 as v8  # noqa: E402


def valid_reservation() -> pd.DataFrame:
    replay_generators = [f"replay:tts{i:02d}" for i in range(11)]
    replay = [name for i, name in enumerate(replay_generators) for _ in range(2 if i < 7 else 1)]
    sonics_generators = [
        *("sonics:chirp-v3",) * 4,
        *("sonics:chirp-v3.5",) * 4,
        *("sonics:udio-120s",) * 4,
        *("sonics:chirp-v2-xxl-alpha",) * 3,
        *("sonics:udio-30s",) * 3,
    ]
    sonics_platforms = ["suno"] * 11 + ["udio"] * 7
    fake_voice_index = 0
    ff_index = 0
    rows = []
    for mode in v8.MODES:
        for voice_fake, music_fake in v8.CELLS:
            cell = f"{'F' if voice_fake else 'R'}{'F' if music_fake else 'R'}"
            for _ in range(v8.PER_CELL):
                index = len(rows)
                joint = cell == "FF"
                if joint:
                    voice_generator = sonics_generators[ff_index] + ":vocal"
                    music_generator = sonics_generators[ff_index]
                    platform = sonics_platforms[ff_index]
                    ff_index += 1
                else:
                    voice_generator = replay[fake_voice_index] if voice_fake else "bonafide"
                    platform = "FakeMusicCaps" if music_fake else "FMA"
                    music_generator = "musicldm" if music_fake else "FMA"
                    fake_voice_index += int(voice_fake)
                rows.append({
                    "BASE_ID": f"base_{index:03d}", "MIX_MODE": mode,
                    "COMPONENT_CASE": cell, "VOICE_FAKE": voice_fake,
                    "MUSIC_FAKE": music_fake, "FILE_FAKE": int(voice_fake or music_fake),
                    "VOICE_SOURCE_ID": f"voice_{index:03d}",
                    "VOICE_SPEAKER": f"speaker_{index:03d}",
                    "VOICE_REFERENCE_SPEAKER": f"ref_{index:03d}" if voice_fake and not joint else "",
                    "VOICE_GENERATOR": voice_generator,
                    "VOICE_SOURCE_KIND": "sonics_joint" if joint else ("echofake_replay" if voice_fake else "echofake_bonafide"),
                    "MUSIC_SOURCE_ID": f"music_{index:03d}",
                    "MUSIC_GROUP_ID": f"song_{index:03d}",
                    "MUSIC_GENERATOR": music_generator,
                    "MUSIC_PLATFORM": platform,
                    "MUSIC_SOURCE_KIND": "sonics_joint" if joint else ("fakemusiccaps_instrumental" if music_fake else "fma_instrumental"),
                    "MUSIC_VOCAL_SCREEN": v8.SEMANTIC_PASS,
                })
    return pd.DataFrame(rows)


def test_strongest_vocal_crop_is_deterministic_and_stem_aligned():
    length = 30 * v8.SR
    vocal = np.zeros(length, dtype=np.float32)
    music = np.arange(length, dtype=np.float32)
    vocal[10 * v8.SR:22 * v8.SR] = 1.0
    first_vocal, first_music, first_start = v8.strongest_vocal_crop(vocal, music)
    second_vocal, second_music, second_start = v8.strongest_vocal_crop(vocal, music)
    assert first_start == second_start == 10 * v8.SR
    np.testing.assert_array_equal(first_vocal, second_vocal)
    np.testing.assert_array_equal(first_music, music[first_start:first_start + 12 * v8.SR])


def test_historical_semantic_failures_are_never_reused():
    provenance = {
        "replacement_history": [
            {"old": "failed_initial", "new": "failed_revision1"},
            {"old": "failed_revision1", "new": "current"},
        ]
    }
    assert v8.excluded_semantic_source_ids(provenance, {"current", "passed"}) == {
        "failed_initial", "failed_revision1", "current", "passed",
    }


def test_exact_balance_and_suno_udio_only_ff_contract():
    frame = valid_reservation()
    v8.validate_reservation(frame, require_semantic=True)
    assert frame.groupby(["MIX_MODE", "COMPONENT_CASE"]).size().eq(v8.PER_CELL).all()
    assert frame.loc[frame.MUSIC_PLATFORM.isin({"suno", "udio"}), "COMPONENT_CASE"].eq("FF").all()
    assert frame.loc[frame.MUSIC_PLATFORM.isin({"suno", "udio"}), "VOICE_FAKE"].eq(1).all()


def test_suno_or_udio_in_rf_is_rejected():
    frame = valid_reservation()
    index = frame.index[frame.COMPONENT_CASE.eq("RF")][0]
    frame.loc[index, "MUSIC_PLATFORM"] = "suno"
    with pytest.raises(ValueError, match="contaminated RF"):
        v8.validate_reservation(frame, require_semantic=True)


def test_builder_has_no_authenticity_detector_import_or_scoring_call():
    source = (ROOT / "scripts/build_codec_mixed_blind_v8.py").read_text("utf-8")
    assert "artifactnet_detector" not in source
    assert "AntiDeepfakeDetector" not in source
    assert "compute_metrics(" not in source
