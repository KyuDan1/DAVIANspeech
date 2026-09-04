import csv
from pathlib import Path

import numpy as np
import torch

from src.unified_dual_ssl_head import UnifiedDualSSLHead
from src.unified_dual_ssl_inference import (
    apply_unified_music_fusion, predict_unified_music, predict_unified_tasks,
)


def test_unified_music_fusion_changes_only_music(tmp_path: Path):
    rng = np.random.default_rng(4)
    ids = np.asarray(["a", "b"])
    eat_projection = rng.normal(size=(7, 5)).astype(np.float32)
    spear_projection = rng.normal(size=(9, 6)).astype(np.float32)
    config = {
        "eat_layers": 2, "eat_stats": 3, "eat_dimension": 5,
        "spear_layers": 2, "spear_stats": 4, "spear_dimension": 6,
        "width": 8, "heads": 2, "context_layers": 1,
        "maximum_views": 2, "maximum_bins": 3, "dropout": 0,
        "file_component_weight": .1,
    }
    model = UnifiedDualSSLHead(**config)
    checkpoint = tmp_path / "head.pt"
    torch.save({
        "model": model.state_dict(), "config": config,
        "eat_projection": eat_projection,
        "spear_projection": spear_projection,
        "spear_layers": np.asarray([0, 1]), "spear_bins": 3,
    }, checkpoint)
    eat_path, spear_path = tmp_path / "eat.npz", tmp_path / "spear.npz"
    np.savez(
        eat_path, ids=ids,
        statistics=rng.normal(size=(2, 2, 2, 3, 5)).astype(np.float16),
        view_mask=np.ones((2, 2), bool), projection=eat_projection,
    )
    np.savez(
        spear_path, ids=ids[::-1],
        features=rng.normal(size=(2, 2, 3, 2, 4, 6)).astype(np.float16),
        mask=np.ones((2, 2, 3), bool), projection=spear_projection,
        layers=np.asarray([0, 1]), bins=np.asarray(3),
    )
    submission = tmp_path / "submission.csv"
    columns = [
        "ID", "FILE_FAKE_PROB", "VOICE_FAKE_PROB", "MUSIC_FAKE_PROB",
        "VOICE_PRESENT_PROB", "MUSIC_PRESENT_PROB",
    ]
    original = [
        {"ID": item, "FILE_FAKE_PROB": .2, "VOICE_FAKE_PROB": .3,
         "MUSIC_FAKE_PROB": .4, "VOICE_PRESENT_PROB": .8,
         "MUSIC_PRESENT_PROB": .9}
        for item in ids
    ]
    with submission.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(original)
    expert_path = tmp_path / "expert.npz"
    apply_unified_music_fusion(
        submission, eat_path, spear_path, [checkpoint], device="cpu",
        music_weight=.2, expert_output_path=expert_path,
    )
    with np.load(expert_path, allow_pickle=False) as archive:
        assert np.array_equal(archive["ids"], ids)
        assert archive["probabilities"].shape == (2, 3)
    with submission.open(newline="") as handle:
        updated = list(csv.DictReader(handle))
    for before, after in zip(original, updated):
        for column in columns[1:]:
            if column == "MUSIC_FAKE_PROB":
                assert float(after[column]) != before[column]
            else:
                assert float(after[column]) == before[column]

    with np.load(eat_path, allow_pickle=False) as eat, np.load(
        spear_path, allow_pickle=False
    ) as spear:
        order = np.asarray([1, 0])
        tasks, task_metadata = predict_unified_tasks(
            eat["statistics"], spear["features"][order], eat["view_mask"],
            spear["mask"][order], [checkpoint], device="cpu",
        )
        music, music_metadata = predict_unified_music(
            eat["statistics"], spear["features"][order], eat["view_mask"],
            spear["mask"][order], [checkpoint], device="cpu",
        )
    np.testing.assert_array_equal(tasks[:, 1], music)
    for left, right in zip(task_metadata[:3], music_metadata[:3]):
        np.testing.assert_array_equal(left, right)
    assert task_metadata[3] == music_metadata[3]


def test_unified_file_music_fusion_keeps_voice_and_presence(tmp_path: Path):
    rng = np.random.default_rng(11)
    ids = np.asarray(["sample"])
    eat_projection = rng.normal(size=(7, 5)).astype(np.float32)
    spear_projection = rng.normal(size=(9, 6)).astype(np.float32)
    config = {
        "eat_layers": 2, "eat_stats": 3, "eat_dimension": 5,
        "spear_layers": 2, "spear_stats": 4, "spear_dimension": 6,
        "width": 8, "heads": 2, "context_layers": 1,
        "maximum_views": 2, "maximum_bins": 3, "dropout": 0,
        "file_component_weight": .1,
    }
    model = UnifiedDualSSLHead(**config)
    checkpoint = tmp_path / "head.pt"
    torch.save({
        "model": model.state_dict(), "config": config,
        "eat_projection": eat_projection,
        "spear_projection": spear_projection,
        "spear_layers": np.asarray([0, 1]), "spear_bins": 3,
    }, checkpoint)
    eat_path, spear_path = tmp_path / "eat.npz", tmp_path / "spear.npz"
    np.savez(
        eat_path, ids=ids,
        statistics=rng.normal(size=(1, 2, 2, 3, 5)).astype(np.float16),
        view_mask=np.ones((1, 2), bool), projection=eat_projection,
    )
    np.savez(
        spear_path, ids=ids,
        features=rng.normal(size=(1, 2, 3, 2, 4, 6)).astype(np.float16),
        mask=np.ones((1, 2, 3), bool), projection=spear_projection,
        layers=np.asarray([0, 1]), bins=np.asarray(3),
    )
    submission = tmp_path / "submission.csv"
    original = {
        "ID": "sample", "FILE_FAKE_PROB": .2, "VOICE_FAKE_PROB": .3,
        "MUSIC_FAKE_PROB": .4, "VOICE_PRESENT_PROB": .8,
        "MUSIC_PRESENT_PROB": .9,
    }
    with submission.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(original))
        writer.writeheader()
        writer.writerow(original)
    apply_unified_music_fusion(
        submission, eat_path, spear_path, [checkpoint], device="cpu",
        music_weight=.05, file_weight=.05,
    )
    with submission.open(newline="") as handle:
        updated = next(csv.DictReader(handle))
    assert float(updated["FILE_FAKE_PROB"]) != original["FILE_FAKE_PROB"]
    assert float(updated["MUSIC_FAKE_PROB"]) != original["MUSIC_FAKE_PROB"]
    for column in (
        "VOICE_FAKE_PROB", "VOICE_PRESENT_PROB", "MUSIC_PRESENT_PROB",
    ):
        assert float(updated[column]) == original[column]
