import csv
from pathlib import Path

import numpy as np
import torch

from src.eat_patch_graph import EatPatchGraphHead
from src.eat_patch_graph_inference import apply_eat_patch_graph_fusion


def test_patch_graph_fusion_changes_only_file_and_music(tmp_path: Path):
    rng = np.random.default_rng(21)
    config = {
        "layers": 1, "dimension": 4, "width": 8, "heads": 2, "depth": 1,
        "maximum_views": 2, "maximum_time_nodes": 3,
        "maximum_frequency_nodes": 2, "dropout": 0,
        "temperature": 5., "file_component_weight": .1,
    }
    model = EatPatchGraphHead(**config)
    projection = rng.normal(size=(8, 4)).astype(np.float32)
    checkpoint = tmp_path / "head.pt"
    torch.save({
        "model_type": "eat_patch_graph_multitask",
        "model": model.state_dict(), "config": config,
        "projection": projection, "eat_layers": np.asarray([1]),
    }, checkpoint)
    statistics = tmp_path / "stats.npz"
    np.savez(
        statistics, ids=np.asarray(["sample"]),
        temporal=rng.normal(size=(1, 2, 1, 2, 3, 4)).astype(np.float16),
        spectral=rng.normal(size=(1, 2, 1, 2, 2, 4)).astype(np.float16),
        view_mask=np.ones((1, 2), bool), projection=projection,
        layers=np.asarray([1]),
    )
    original = {
        "ID": "sample", "FILE_FAKE_PROB": .2, "VOICE_FAKE_PROB": .3,
        "MUSIC_FAKE_PROB": .4, "VOICE_PRESENT_PROB": .8,
        "MUSIC_PRESENT_PROB": .9,
    }
    submission = tmp_path / "submission.csv"
    with submission.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(original))
        writer.writeheader()
        writer.writerow(original)
    apply_eat_patch_graph_fusion(
        submission, statistics, [checkpoint], device="cpu",
        file_weight=.05, music_weight=.05,
    )
    with submission.open(newline="") as handle:
        updated = next(csv.DictReader(handle))
    assert float(updated["FILE_FAKE_PROB"]) != original["FILE_FAKE_PROB"]
    assert float(updated["MUSIC_FAKE_PROB"]) != original["MUSIC_FAKE_PROB"]
    for column in (
        "VOICE_FAKE_PROB", "VOICE_PRESENT_PROB", "MUSIC_PRESENT_PROB",
    ):
        assert float(updated[column]) == original[column]
