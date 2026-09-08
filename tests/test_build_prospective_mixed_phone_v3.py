from pathlib import Path

import pandas as pd
import pytest

from scripts.build_prospective_mixed_phone_v3 import (
    FAKEMUSICCAPS_GENERATORS,
    MUSIC_VOCAL_SCREEN_PASS,
    canonical_music_group,
    locate_source,
    make_reservation,
    musiccaps_vocal_screen,
    parse_asvspoof_protocol,
    validate_reservation,
)


def write_fixture(root: Path) -> dict[str, Path]:
    rows = []
    fore_lines = []
    back_lines = []
    asv_lines = []
    musiccaps_rows = []
    generators = FAKEMUSICCAPS_GENERATORS
    for index in range(150):
        for voice_fake in (0, 1):
            voice_id = f"LA_E_{voice_fake}{index:06d}"
            fore_auth = "spoof" if voice_fake else "bonafide"
            voice_archive = f"{'FS' if voice_fake else 'RS'}_ASVspoof_{voice_id}"
            fore_lines.append(
                f"ASVspoof {voice_archive} - - {fore_auth} "
                f"/yourownpath/unmixed_dataset/eval/{voice_archive}.wav - - eval\n"
            )
            attack = f"A{7 + index % 13:02d}" if voice_fake else "-"
            asv_lines.append(
                f"speaker_{index % 31:02d} {voice_id} - {attack} {fore_auth}\n"
            )
        for music_fake in (0, 1):
            generator = generators[index % len(generators)]
            if music_fake:
                ytid = f"yt{index:09d}"
                music_archive = f"FM_{generator}_{ytid}"
                sub_dataset = "FakeMusicCaps"
                back_auth = "spoof"
                musiccaps_rows.append({
                    "ytid": ytid,
                    "caption": "This is an instrumental recording with no vocals.",
                    "aspect_list": "['instrumental', 'no voice']",
                })
            else:
                music_archive = f"RM_{index // 1000:03d}_{index:06d}"
                back_auth = "bonafide"
                sub_dataset = "FMA"
            rows.append({
                "file_path": f"/yourownpath/unmixed_dataset/eval/{music_archive}.wav",
                "major_type": "Background",
                "sub_type": "Music",
                "authenticity": back_auth,
                "sub_dataset": sub_dataset,
                "split": "eval",
            })
            back_lines.append(
                f"Music {music_archive} - - {back_auth} "
                f"/yourownpath/unmixed_dataset/eval/{music_archive}.wav - - eval\n"
            )
    sonics_archive = "FM_fake_songs_wav_fake_99999_suno_0"
    rows.append({
        "file_path": f"/yourownpath/unmixed_dataset/eval/{sonics_archive}.wav",
        "major_type": "Background",
        "sub_type": "Music",
        "authenticity": "spoof",
        "sub_dataset": "SONICS",
        "split": "eval",
    })
    back_lines.append(
        f"Music {sonics_archive} - - spoof "
        f"/yourownpath/unmixed_dataset/eval/{sonics_archive}.wav - - eval\n"
    )
    paths = {
        "details": root / "details.csv",
        "fore": root / "fore.txt",
        "back": root / "back.txt",
        "asv": root / "asv.txt",
        "musiccaps": root / "musiccaps.csv",
        "config": root / "partitions.yaml",
    }
    pd.DataFrame(rows).drop_duplicates().to_csv(paths["details"], index=False)
    paths["fore"].write_text("".join(dict.fromkeys(fore_lines)), "utf-8")
    paths["back"].write_text("".join(dict.fromkeys(back_lines)), "utf-8")
    paths["asv"].write_text("".join(dict.fromkeys(asv_lines)), "utf-8")
    pd.DataFrame(musiccaps_rows).drop_duplicates("ytid").to_csv(
        paths["musiccaps"], index=False
    )
    paths["config"].write_text("train: []\nlocked_eval: []\n", "utf-8")
    return paths


def test_canonical_music_groups_collapse_aliases():
    assert canonical_music_group("/FMA/044/044959.wav") == "fma:44959"
    assert canonical_music_group("fma_044959") == "fma:44959"
    assert canonical_music_group("fake_53851_suno_0.wav") == "sonics:53851"
    assert canonical_music_group("sonics_53851") == "sonics:53851"
    assert canonical_music_group("/FakeMusicCaps/musicgen/zUaZGKZIKic.wav") == "fmc:zUaZGKZIKic"


def test_strict_mode_requires_asvspoof_provenance(tmp_path: Path):
    paths = write_fixture(tmp_path)
    with pytest.raises(ValueError, match="STRICT PROVENANCE FAILURE"):
        make_reservation(
            paths["details"], paths["fore"], paths["back"], paths["config"],
            None, seed=17,
        )


def test_strict_mode_requires_musiccaps_metadata(tmp_path: Path):
    paths = write_fixture(tmp_path)
    with pytest.raises(ValueError, match="STRICT SEMANTIC FAILURE"):
        make_reservation(
            paths["details"], paths["fore"], paths["back"], paths["config"],
            paths["asv"], seed=17,
        )


def test_asvspoof_parser_rejects_spoof_without_attack(tmp_path: Path):
    path = tmp_path / "bad.txt"
    path.write_text("speaker LA_E_1 - - spoof\n", "utf-8")
    with pytest.raises(ValueError, match="Missing attack ID"):
        parse_asvspoof_protocol(path)


def test_musiccaps_screen_requires_explicit_negative_and_no_positive_signal():
    assert musiccaps_vocal_screen({
        "caption": "An instrumental track with no vocals.",
        "aspect_list": "['instrumental', 'no voice']",
    }) == (True, MUSIC_VOCAL_SCREEN_PASS)
    assert musiccaps_vocal_screen({
        "caption": "Instrumental backing with a female singer.",
        "aspect_list": "['instrumental', 'female vocal']",
    }) == (False, "excluded_positive_voice_signal")
    assert musiccaps_vocal_screen({
        "caption": "A piano plays a slow melody.",
        "aspect_list": "['piano', 'slow']",
    }) == (False, "excluded_no_explicit_instrumental_or_no_vocal")
    assert musiccaps_vocal_screen(None) == (
        False, "excluded_missing_musiccaps_metadata"
    )


def test_multi_root_source_resolution_requires_exactly_one_match(tmp_path: Path):
    first = tmp_path / "first"
    second = tmp_path / "second"
    member = Path("unmixed_dataset/eval/source.wav")
    (second / member).parent.mkdir(parents=True)
    (second / member).write_bytes(b"source")
    assert locate_source([first, second], str(member)) == second / member
    (first / member).parent.mkdir(parents=True)
    (first / member).write_bytes(b"duplicate")
    with pytest.raises(FileNotFoundError, match="Expected exactly one source"):
        locate_source([first, second], str(member))
    with pytest.raises(FileNotFoundError, match="Expected exactly one source"):
        locate_source([first, second], "unmixed_dataset/eval/missing.wav")


def test_strict_reservation_is_balanced_unique_and_deterministic(tmp_path: Path):
    paths = write_fixture(tmp_path)
    first, provenance = make_reservation(
        paths["details"], paths["fore"], paths["back"], paths["config"],
        paths["asv"], seed=20260904,
        musiccaps_metadata_path=paths["musiccaps"],
    )
    second, _ = make_reservation(
        paths["details"], paths["fore"], paths["back"], paths["config"],
        paths["asv"], seed=20260904,
        musiccaps_metadata_path=paths["musiccaps"],
    )
    pd.testing.assert_frame_equal(first, second)
    validate_reservation(first)
    assert len(first) == 240
    assert first.VOICE_SOURCE_ID.nunique() == 240
    assert first.MUSIC_GROUP_ID.nunique() == 240
    assert provenance["provenance_mode"] == "strict"
    assert provenance["rendered_rows_planned"] == 1200
    fake_music = first.loc[first.MUSIC_FAKE.eq(1), "MUSIC_GENERATOR"].value_counts()
    assert fake_music.max() - fake_music.min() <= 1
    assert set(fake_music.index) == set(FAKEMUSICCAPS_GENERATORS)
    assert set(first.loc[first.MUSIC_FAKE.eq(1), "MUSIC_VOCAL_SCREEN"]) == {
        MUSIC_VOCAL_SCREEN_PASS
    }
    assert set(first.loc[first.MUSIC_FAKE.eq(0), "MUSIC_VOCAL_SCREEN"]) == {
        "unscreened_real_noncausal"
    }
    assert provenance["inputs"]["musiccaps_metadata"]["sha256"]
    assert provenance["music_vocal_screen"]["counts"][
        "fake_musiccaps_passed_text_screen"
    ] == 150
    assert provenance["music_vocal_screen"]["counts"][
        "excluded_sonics_suno_udio"
    ] == 1
    fake_voice = first.loc[first.VOICE_FAKE.eq(1), "VOICE_GENERATOR"].value_counts()
    assert fake_voice.max() - fake_voice.min() <= 1


def test_predeclared_generator_subset_is_balanced_and_recorded(tmp_path: Path):
    paths = write_fixture(tmp_path)
    generators = tuple(
        generator
        for generator in FAKEMUSICCAPS_GENERATORS
        if generator != "musicldm"
    )
    frame, provenance = make_reservation(
        paths["details"], paths["fore"], paths["back"], paths["config"],
        paths["asv"], seed=20260906, per_cell=8,
        musiccaps_metadata_path=paths["musiccaps"],
        fake_music_generators=generators,
        id_prefix="blindv5",
    )
    validate_reservation(frame, per_cell=8, fake_music_generators=generators)
    fake_music = frame.loc[frame.MUSIC_FAKE.eq(1), "MUSIC_GENERATOR"]
    assert set(fake_music) == set(generators)
    assert set(fake_music.value_counts()) == {12}
    assert provenance["selection"]["requested_fake_music_generators"] == list(
        generators
    )
    assert provenance["rendered_rows_planned"] == 480
    assert provenance["id_prefix"] == "blindv5"
    assert frame.BASE_ID.str.startswith("blindv5_").all()


def test_registered_voice_speaker_is_excluded_from_reservation(tmp_path: Path):
    paths = write_fixture(tmp_path)
    pd.DataFrame({
        "ID": ["protected"],
        "VOICE_SPEAKER": ["speaker_00"],
    }).to_csv(tmp_path / "protected.csv", index=False)
    paths["config"].write_text(
        "train: []\ndevelopment:\n"
        f"  - {tmp_path.name}/protected.csv\n"
        "locked_eval: []\n",
        "utf-8",
    )
    frame, _ = make_reservation(
        paths["details"], paths["fore"], paths["back"], paths["config"],
        paths["asv"], seed=20260904,
        musiccaps_metadata_path=paths["musiccaps"],
    )
    assert "speaker_00" not in set(frame.VOICE_SPEAKER)


def test_semantic_exclusions_replace_only_three_failed_fake_music_rows(
    tmp_path: Path,
):
    paths = write_fixture(tmp_path)
    baseline, _ = make_reservation(
        paths["details"], paths["fore"], paths["back"], paths["config"],
        paths["asv"], seed=20260904,
        musiccaps_metadata_path=paths["musiccaps"],
    )
    fake = baseline.loc[baseline.MUSIC_FAKE.eq(1)].copy()
    failed = (
        fake.drop_duplicates("MUSIC_GENERATOR")
        .head(3)["MUSIC_SOURCE_ID"].tolist()
    )
    audit = pd.DataFrame({
        "MUSIC_SOURCE_ID": fake.MUSIC_SOURCE_ID,
        "ACOUSTIC_SCREEN_PASS": [
            source_id not in failed for source_id in fake.MUSIC_SOURCE_ID
        ],
    })
    exclusions = tmp_path / "source_scores.csv"
    audit.to_csv(exclusions, index=False)
    revised, provenance = make_reservation(
        paths["details"], paths["fore"], paths["back"], paths["config"],
        paths["asv"], seed=20260904,
        musiccaps_metadata_path=paths["musiccaps"],
        semantic_exclusions_path=exclusions,
    )
    old = baseline.set_index("BASE_ID").sort_index()
    new = revised.set_index("BASE_ID").sort_index()
    assert old.VOICE_SOURCE_ID.equals(new.VOICE_SOURCE_ID)
    assert old.VOICE_ARCHIVE_MEMBER.equals(new.VOICE_ARCHIVE_MEMBER)
    changed = old.MUSIC_SOURCE_ID.ne(new.MUSIC_SOURCE_ID)
    assert changed.sum() == 3
    assert set(old.loc[changed, "MUSIC_SOURCE_ID"]) == set(failed)
    assert old.loc[~changed, "MUSIC_SOURCE_ID"].equals(
        new.loc[~changed, "MUSIC_SOURCE_ID"]
    )
    assert old.loc[changed, "MUSIC_GENERATOR"].equals(
        new.loc[changed, "MUSIC_GENERATOR"]
    )
    assert not set(new.MUSIC_SOURCE_ID) & set(failed)
    assert old.loc[old.MUSIC_FAKE.eq(0), "MUSIC_SOURCE_ID"].equals(
        new.loc[new.MUSIC_FAKE.eq(0), "MUSIC_SOURCE_ID"]
    )
    validate_reservation(revised)
    semantic = provenance["semantic_exclusions"]
    assert semantic["selection_rule"] == "ACOUSTIC_SCREEN_PASS=false"
    assert semantic["input_rows"] == 120
    assert semantic["exclusion_rows"] == 3
    assert semantic["exclusion_unique_ids"] == 3
    assert semantic["eligible_pool_matches"] == 3
    assert semantic["selected_replacement_count"] == 3
    assert len(semantic["replacements"]) == 3
    assert provenance["inputs"]["semantic_exclusions"]["sha256"]


def test_fallback_is_explicit_and_stamped(tmp_path: Path):
    paths = write_fixture(tmp_path)
    frame, provenance = make_reservation(
        paths["details"], paths["fore"], paths["back"], paths["config"],
        None, seed=21, allow_fallback=True,
        musiccaps_metadata_path=paths["musiccaps"],
    )
    validate_reservation(frame)
    assert provenance["provenance_mode"] == "fallback_generator_seen"
    assert set(frame.VOICE_SPEAKER) == {"unknown"}
