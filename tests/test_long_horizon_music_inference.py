from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from long_horizon_music_inference import (
    _fuse,
    apply_long_horizon_music_fusion,
    predict_long_horizon_music,
)


def write_checkpoint(path: Path, feature_count: int) -> None:
    rng = np.random.default_rng(7)
    np.savez(
        path,
        eat_matrix=rng.normal(size=(768, 2)).astype(np.float32),
        spear_matrix=rng.normal(size=(1280, 2)).astype(np.float32),
        member_count=np.asarray(2),
        member_logit_weights=np.asarray([0.4, 0.6]),
        phone_member_logit_weights=np.asarray([1.0, 0.0]),
        music_weight=np.asarray(0.4),
        file_weight=np.asarray(0.2),
        file_music_presence_threshold=np.asarray(0.7),
        member_0_mean=np.zeros(feature_count, np.float32),
        member_0_std=np.ones(feature_count, np.float32),
        member_0_weight=np.linspace(-0.01, 0.01, feature_count, dtype=np.float32),
        member_0_bias=np.asarray(0.1, np.float32),
        member_1_mean=np.zeros(feature_count, np.float32),
        member_1_std=np.ones(feature_count, np.float32),
        member_1_weight=np.linspace(0.01, -0.01, feature_count, dtype=np.float32),
        member_1_bias=np.asarray(-0.2, np.float32),
    )


def test_phone_route_uses_average_risk_member(tmp_path: Path) -> None:
    rng = np.random.default_rng(11)
    eat = rng.normal(size=(2, 3, 4, 768)).astype(np.float16)
    spear = rng.normal(size=(2, 3, 13, 4, 1280)).astype(np.float16)
    mask = np.ones((2, 3), dtype=bool)
    checkpoint = tmp_path / "head.npz"
    write_checkpoint(checkpoint, 2 * (4 + 13 * 4) * 2 + 1)
    normal = predict_long_horizon_music(
        eat, spear, mask, mask, checkpoint, device="cpu",
        phone_mask=np.asarray([False, False]),
    )
    routed = predict_long_horizon_music(
        eat, spear, mask, mask, checkpoint, device="cpu",
        phone_mask=np.asarray([False, True]),
    )
    assert routed[0] == normal[0]
    assert routed[1] != normal[1]
    assert np.all((routed >= 0) & (routed <= 1))


def test_fusion_changes_only_music_and_routed_file(tmp_path: Path) -> None:
    rng = np.random.default_rng(13)
    ids = np.asarray(["a", "b"])
    eat = rng.normal(size=(2, 3, 4, 768)).astype(np.float16)
    spear = rng.normal(size=(2, 3, 13, 4, 1280)).astype(np.float16)
    mask = np.ones((2, 3), dtype=bool)
    eat_path, spear_path = tmp_path / "eat.npz", tmp_path / "spear.npz"
    np.savez(eat_path, ids=ids, statistics=eat, view_mask=mask)
    np.savez(spear_path, ids=ids[::-1], statistics=spear[::-1], view_mask=mask[::-1])
    checkpoint = tmp_path / "head.npz"
    write_checkpoint(checkpoint, 2 * (4 + 13 * 4) * 2 + 1)
    submission = tmp_path / "submission.csv"
    rows = [
        {"ID": "a", "FILE_FAKE_PROB": "0.2", "VOICE_FAKE_PROB": "0.3",
         "MUSIC_FAKE_PROB": "0.4", "VOICE_PRESENT_PROB": "0.5",
         "MUSIC_PRESENT_PROB": "0.8"},
        {"ID": "b", "FILE_FAKE_PROB": "0.2", "VOICE_FAKE_PROB": "0.3",
         "MUSIC_FAKE_PROB": "0.4", "VOICE_PRESENT_PROB": "0.5",
         "MUSIC_PRESENT_PROB": "0.6"},
    ]
    with submission.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    apply_long_horizon_music_fusion(
        submission, eat_path, spear_path, checkpoint, device="cpu",
    )
    with submission.open(newline="", encoding="utf-8") as handle:
        output = list(csv.DictReader(handle))
    assert float(output[0]["FILE_FAKE_PROB"]) != 0.2
    assert float(output[1]["FILE_FAKE_PROB"]) == 0.2
    assert all(float(row["MUSIC_FAKE_PROB"]) != 0.4 for row in output)
    assert all(float(row["VOICE_FAKE_PROB"]) == 0.3 for row in output)
    assert all(float(row["VOICE_PRESENT_PROB"]) == 0.5 for row in output)
    assert all(float(row["MUSIC_PRESENT_PROB"]) in {0.6, 0.8} for row in output)


def test_logit_fusion_endpoints() -> None:
    anchor = np.asarray([0.1, 0.9])
    expert = np.asarray([0.8, 0.2])
    np.testing.assert_allclose(_fuse(anchor, expert, 0), anchor)
    np.testing.assert_allclose(_fuse(anchor, expert, 1), expert)
