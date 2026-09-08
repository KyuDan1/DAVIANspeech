import csv

import numpy as np
import pytest

from src.music_only_residual import (
    apply_music_probability_fusion,
    fuse_music_logits,
    logit_ensemble,
)


def test_logit_ensemble_and_fusion_have_expected_odds_geometry():
    members = logit_ensemble([np.array([0.2]), np.array([0.8])])
    assert members == pytest.approx([0.5])
    fused = fuse_music_logits(np.array([0.2]), np.array([0.8]), 0.5)
    assert fused == pytest.approx([0.5])


def test_music_fusion_preserves_four_other_columns_and_aligns_ids(tmp_path):
    path = tmp_path / "submission.csv"
    columns = [
        "ID", "FILE_FAKE_PROB", "VOICE_FAKE_PROB", "MUSIC_FAKE_PROB",
        "VOICE_PRESENT_PROB", "MUSIC_PRESENT_PROB",
    ]
    rows = [
        dict(zip(columns, ["b", ".1", ".2", ".3", ".4", ".5"])),
        dict(zip(columns, ["a", ".6", ".7", ".8", ".9", ".95"])),
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    apply_music_probability_fusion(
        path, np.array(["a", "b"]), np.array([0.25, 0.75]), music_weight=0.1
    )
    with path.open(encoding="utf-8") as handle:
        changed = list(csv.DictReader(handle))
    for before, after in zip(rows, changed):
        for column in columns:
            if column != "MUSIC_FAKE_PROB":
                assert after[column] == before[column]
    assert float(changed[0]["MUSIC_FAKE_PROB"]) > 0.3
    assert float(changed[1]["MUSIC_FAKE_PROB"]) < 0.8


def test_music_fusion_rejects_bad_contracts(tmp_path):
    with pytest.raises(ValueError, match=r"\[0,1\]"):
        fuse_music_logits(np.array([.5]), np.array([.5]), 1.1)
    with pytest.raises(ValueError, match="shapes differ"):
        logit_ensemble([np.zeros(2), np.zeros(3)])
