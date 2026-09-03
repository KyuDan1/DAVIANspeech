from pathlib import Path

import pandas as pd
import pytest

from src.data_guard import assert_no_locked_eval_leakage


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
