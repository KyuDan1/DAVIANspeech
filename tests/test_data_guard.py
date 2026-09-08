from pathlib import Path

import pandas as pd
import pytest

from src.data_guard import (
    assert_development_eval_separation,
    assert_no_locked_eval_leakage,
)


def test_rejects_reused_component_source(tmp_path: Path):
    (tmp_path / "locked.csv").write_text(
        "ID,VOICE_SOURCE_ID\neval_1,voice_source_7\n", encoding="utf-8"
    )
    (tmp_path / "train.csv").write_text(
        "ID,VOICE_SOURCE_ID\ntrain_1,voice_source_7\n", encoding="utf-8"
    )
    (tmp_path / "partitions.yaml").write_text(
        "locked_eval:\n  - locked.csv\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="TRAIN/EVAL LEAKAGE"):
        assert_no_locked_eval_leakage(
            tmp_path / "train.csv", tmp_path / "partitions.yaml"
        )


def test_allows_disjoint_sources(tmp_path: Path):
    pd.DataFrame({"ID": ["eval_1"], "GROUP_ID": ["song_a"]}).to_csv(
        tmp_path / "locked.csv", index=False
    )
    pd.DataFrame({"ID": ["train_1"], "GROUP_ID": ["song_b"]}).to_csv(
        tmp_path / "train.csv", index=False
    )
    (tmp_path / "partitions.yaml").write_text(
        "locked_eval:\n  - locked.csv\n", encoding="utf-8"
    )
    assert_no_locked_eval_leakage(
        tmp_path / "train.csv", tmp_path / "partitions.yaml"
    )


def test_rejects_reused_second_call_speaker(tmp_path: Path):
    pd.DataFrame({"ID": ["eval_1"], "GROUP_ID": ["speaker_7"]}).to_csv(
        tmp_path / "locked.csv", index=False
    )
    pd.DataFrame({
        "ID": ["call_1"], "FIRST_GROUP": ["speaker_1"],
        "SECOND_GROUP": ["speaker_7"],
    }).to_csv(tmp_path / "train.csv", index=False)
    (tmp_path / "partitions.yaml").write_text(
        "locked_eval:\n  - locked.csv\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="TRAIN/EVAL LEAKAGE"):
        assert_no_locked_eval_leakage(
            tmp_path / "train.csv", tmp_path / "partitions.yaml"
        )


@pytest.mark.parametrize(
    "identity_column",
    ("MUSIC_GROUP_ID", "VOICE_SPEAKER", "PAIR_GROUP", "PARENT_ID"),
)
def test_rejects_new_identity_columns(tmp_path: Path, identity_column: str):
    pd.DataFrame({"ID": ["eval_1"], identity_column: ["shared_7"]}).to_csv(
        tmp_path / "locked.csv", index=False
    )
    pd.DataFrame({"ID": ["train_1"], identity_column: ["shared_7"]}).to_csv(
        tmp_path / "train.csv", index=False
    )
    (tmp_path / "partitions.yaml").write_text(
        "locked_eval:\n  - locked.csv\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="TRAIN/EVAL LEAKAGE"):
        assert_no_locked_eval_leakage(
            tmp_path / "train.csv", tmp_path / "partitions.yaml"
        )


@pytest.mark.parametrize("eval_role", ("locked_eval", "ood_holdout"))
def test_rejects_development_holdout_overlap(
    tmp_path: Path, eval_role: str,
):
    pd.DataFrame({"ID": ["dev_1"], "MUSIC_GROUP_ID": ["song_7"]}).to_csv(
        tmp_path / "development.csv", index=False
    )
    pd.DataFrame({"ID": ["eval_1"], "MUSIC_GROUP_ID": ["song_7"]}).to_csv(
        tmp_path / "evaluation.csv", index=False
    )
    (tmp_path / "partitions.yaml").write_text(
        "development:\n  - development.csv\n"
        f"{eval_role}:\n  - evaluation.csv\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="DEVELOPMENT/EVAL IDENTITY OVERLAP"):
        assert_development_eval_separation(tmp_path / "partitions.yaml")


def test_training_guard_also_validates_partition_role_overlap(tmp_path: Path):
    pd.DataFrame({"ID": ["dev_1"], "PAIR_GROUP": ["pair_7"]}).to_csv(
        tmp_path / "development.csv", index=False
    )
    pd.DataFrame({"ID": ["locked_1"], "PAIR_GROUP": ["pair_7"]}).to_csv(
        tmp_path / "locked.csv", index=False
    )
    pd.DataFrame({"ID": ["train_1"]}).to_csv(
        tmp_path / "train.csv", index=False
    )
    (tmp_path / "partitions.yaml").write_text(
        "development:\n  - development.csv\n"
        "locked_eval:\n  - locked.csv\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="DEVELOPMENT/EVAL IDENTITY OVERLAP"):
        assert_no_locked_eval_leakage(
            tmp_path / "train.csv", tmp_path / "partitions.yaml"
        )


def test_allows_intentional_stress_parent_overlap_but_protects_from_train(
    tmp_path: Path,
):
    pd.DataFrame({"ID": ["dev_1"], "PARENT_ID": ["parent_7"]}).to_csv(
        tmp_path / "development.csv", index=False
    )
    pd.DataFrame({"ID": ["stress_1"], "PARENT_ID": ["parent_7"]}).to_csv(
        tmp_path / "stress.csv", index=False
    )
    pd.DataFrame({"ID": ["train_1"], "PARENT_ID": ["parent_7"]}).to_csv(
        tmp_path / "train.csv", index=False
    )
    (tmp_path / "partitions.yaml").write_text(
        "development:\n  - development.csv\n"
        "stress_eval:\n  - stress.csv\n",
        encoding="utf-8",
    )
    assert_development_eval_separation(tmp_path / "partitions.yaml")
    with pytest.raises(ValueError, match="TRAIN/EVAL LEAKAGE"):
        assert_no_locked_eval_leakage(
            tmp_path / "train.csv", tmp_path / "partitions.yaml"
        )


def test_retrospective_diagnostic_is_protected_from_train(tmp_path: Path):
    pd.DataFrame({"ID": ["audit_1"], "PARENT_ID": ["parent_9"]}).to_csv(
        tmp_path / "retrospective.csv", index=False
    )
    pd.DataFrame({"ID": ["train_1"], "PARENT_ID": ["parent_9"]}).to_csv(
        tmp_path / "train.csv", index=False
    )
    (tmp_path / "partitions.yaml").write_text(
        "retrospective_diagnostic:\n  - retrospective.csv\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="TRAIN/EVAL LEAKAGE"):
        assert_no_locked_eval_leakage(
            tmp_path / "train.csv", tmp_path / "partitions.yaml"
        )


def test_invalid_eval_is_protected_from_train(tmp_path: Path):
    pd.DataFrame({"ID": ["invalid_1"], "PARENT_ID": ["parent_11"]}).to_csv(
        tmp_path / "invalid.csv", index=False
    )
    pd.DataFrame({"ID": ["train_1"], "PARENT_ID": ["parent_11"]}).to_csv(
        tmp_path / "train.csv", index=False
    )
    (tmp_path / "partitions.yaml").write_text(
        "invalid_eval:\n  - invalid.csv\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="TRAIN/EVAL LEAKAGE"):
        assert_no_locked_eval_leakage(
            tmp_path / "train.csv", tmp_path / "partitions.yaml"
        )


def test_training_validation_allows_source_overlap_but_not_row_overlap(
    tmp_path: Path,
):
    pd.DataFrame({
        "ID": ["validation_1"], "MUSIC_GROUP_ID": ["shared_source"],
    }).to_csv(tmp_path / "validation.csv", index=False)
    pd.DataFrame({
        "ID": ["train_1"], "MUSIC_GROUP_ID": ["shared_source"],
    }).to_csv(tmp_path / "train.csv", index=False)
    (tmp_path / "partitions.yaml").write_text(
        "training_validation:\n  - validation.csv\n"
        "locked_eval: []\n",
        encoding="utf-8",
    )
    assert_no_locked_eval_leakage(
        tmp_path / "train.csv", tmp_path / "partitions.yaml"
    )

    pd.DataFrame({
        "ID": ["validation_1"], "MUSIC_GROUP_ID": ["another_source"],
    }).to_csv(tmp_path / "train.csv", index=False)
    with pytest.raises(ValueError, match="TRAIN/VALIDATION ROW LEAKAGE"):
        assert_no_locked_eval_leakage(
            tmp_path / "train.csv", tmp_path / "partitions.yaml"
        )


def test_rejects_training_validation_source_overlap_with_ood(tmp_path: Path):
    pd.DataFrame({
        "ID": ["validation_1"], "MUSIC_GROUP_ID": ["song_8"],
    }).to_csv(tmp_path / "validation.csv", index=False)
    pd.DataFrame({
        "ID": ["ood_1"], "MUSIC_GROUP_ID": ["song_8"],
    }).to_csv(tmp_path / "ood.csv", index=False)
    (tmp_path / "partitions.yaml").write_text(
        "training_validation:\n  - validation.csv\n"
        "ood_holdout:\n  - ood.csv\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="DEVELOPMENT/EVAL IDENTITY OVERLAP"):
        assert_development_eval_separation(tmp_path / "partitions.yaml")


@pytest.mark.parametrize("bad_id", ["", None])
def test_rejects_missing_or_blank_training_validation_ids(
    tmp_path: Path, bad_id,
):
    pd.DataFrame({"ID": [bad_id]}).to_csv(
        tmp_path / "validation.csv", index=False
    )
    pd.DataFrame({"ID": ["train_1"]}).to_csv(
        tmp_path / "train.csv", index=False
    )
    (tmp_path / "partitions.yaml").write_text(
        "training_validation:\n  - validation.csv\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="missing IDs|blank IDs"):
        assert_no_locked_eval_leakage(
            tmp_path / "train.csv", tmp_path / "partitions.yaml"
        )
