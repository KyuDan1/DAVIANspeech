from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path
import shutil
import sys

import numpy as np
import pandas as pd
import pytest

from scripts.run_v47_anchor_cache import (
    XLSR_EMBEDDING_DIM,
    _load_validated_npz,
    _write_deterministic_npz,
    build_entrypoint_overlay,
    build_post_entrypoint_overlay,
    build_pipeline_overlay,
    merge_run,
    prepare_run,
)


ROOT = Path(__file__).resolve().parents[1]
TRAINING_SPEC = importlib.util.spec_from_file_location(
    "v47_cache_training_integration",
    ROOT / "scripts/train_three_stream_anchor_residual.py",
)
TRAINING = importlib.util.module_from_spec(TRAINING_SPEC)
assert TRAINING_SPEC.loader is not None
sys.modules[TRAINING_SPEC.name] = TRAINING
TRAINING_SPEC.loader.exec_module(TRAINING)


def _truth(path: Path, ids: list[str], sources: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({
        "ID": ids,
        "VOICE_SOURCE_ID": sources,
        "FILE_FAKE": [1] * len(ids),  # Must never enter a cache-runner artifact.
    }).to_csv(path, index=False)


def _package(path: Path) -> Path:
    (path / "model" / "src").mkdir(parents=True)
    (path / "model" / "weights").mkdir()
    shutil.copy2(ROOT / "component_query_or_v47" / "script.py", path / "script.py")
    shutil.copy2(
        ROOT / "component_query_or_v47" / "model" / "src" / "pipeline.py",
        path / "model" / "src" / "pipeline.py",
    )
    (path / "model" / "src" / "presence.py").write_text("VALUE = 1\n")
    (path / "model" / "weights" / "head.bin").write_bytes(b"weights")
    return path


def _fixture_tree(tmp_path: Path):
    train = tmp_path / "data" / "train" / "truth.csv"
    dev = tmp_path / "data" / "dev" / "truth.csv"
    locked = tmp_path / "data" / "locked" / "truth.csv"
    _truth(train, ["train-a", "train-b"], ["train-source-a", "train-source-b"])
    _truth(dev, ["dev-a"], ["dev-source-a"])
    _truth(locked, ["locked-a"], ["locked-source-a"])
    for directory, files in (
        (train.parent / "audio", {"train-a.wav": b"train-a", "train-b.flac": b"train-b"}),
        (dev.parent / "audio", {"dev-a.wav": b"dev-a"}),
        (locked.parent / "audio", {"locked-a.wav": b"locked"}),
    ):
        directory.mkdir()
        for name, payload in files.items():
            (directory / name).write_bytes(payload)
    config = tmp_path / "configs" / "partitions.yaml"
    config.parent.mkdir()
    config.write_text(
        "train:\n  - data/train/truth.csv\n"
        "development:\n  - data/dev/truth.csv\n"
        "locked_eval:\n  - data/locked/truth.csv\n",
        encoding="utf-8",
    )
    return config, train, dev, locked, _package(tmp_path / "package")


def test_overlays_are_narrow_opt_in_extensions_of_exact_package():
    pipeline = build_pipeline_overlay(
        ROOT / "component_query_or_v47/model/src/pipeline.py",
        ROOT / "src/pipeline.py",
    )
    entrypoint = build_entrypoint_overlay(ROOT / "component_query_or_v47/script.py")
    compile(pipeline, "pipeline.py", "exec")
    compile(entrypoint, "script.py", "exec")
    assert "xlsr-cache-extra-pass" in pipeline
    assert "CACHE_EXPORT_TIMING" in pipeline
    assert "return_windows=True" in pipeline
    assert "CACHE_RETAINED" in entrypoint
    assert "EXACT_V47_PREDICTION" in entrypoint
    assert "apply_component_query_mhfa_fusion(" in entrypoint


def test_post_overlay_skips_only_base_pipeline_and_keeps_all_fusions():
    entrypoint = build_post_entrypoint_overlay(
        ROOT / "component_query_or_v47/script.py"
    )
    compile(entrypoint, "post_script.py", "exec")
    assert "RESUME_VERIFIED_BASE_POST_FUSIONS" in entrypoint
    assert "    run(args)\n" not in entrypoint
    assert "apply_eat_presence_fusion(" in entrypoint
    assert "apply_component_query_mhfa_fusion(" in entrypoint
    assert "CACHE_RETAINED" in entrypoint
    assert "EXACT_V47_PREDICTION" in entrypoint


def test_prepare_is_label_blind_unique_and_uses_isolated_symlink_workdirs(tmp_path):
    config, train, dev, _, package = _fixture_tree(tmp_path)
    work = tmp_path / "run"
    prepare_run(
        config, package, ROOT / "src/pipeline.py", work,
        [str(train), str(dev)], num_shards=2,
    )

    mapping = pd.read_csv(work / "corpus/dataset_map.csv", dtype=str)
    sample = pd.read_csv(work / "corpus/sample_submission.csv", dtype=str)
    assert len(mapping) == 3
    assert mapping.COMBINED_ID.is_unique
    assert mapping.COMBINED_ID.tolist() == sample.ID.tolist()
    assert set(mapping.ORIGINAL_ID) == {"train-a", "train-b", "dev-a"}
    assert "FILE_FAKE" not in mapping.columns
    assert "FILE_FAKE" not in sample.columns
    for row in mapping.itertuples():
        link = work / "corpus/audio" / row.AUDIO_FILENAME
        assert link.is_symlink()
        assert link.resolve() == Path(row.SOURCE_AUDIO)

    for index in range(2):
        shard = work / "shards" / f"shard_{index:03d}"
        assert not (shard / "script.py").is_symlink()
        assert not (shard / "model/src/pipeline.py").is_symlink()
        assert (shard / "model/src/presence.py").is_symlink()
        assert (shard / "model/weights").is_symlink()
        assert (shard / "data/test").is_dir()
        assert not (shard / "data/test").is_symlink()
        expected = mapping.iloc[index::2]
        assert sorted(path.name for path in (shard / "data/test").iterdir()) == sorted(
            expected.AUDIO_FILENAME.tolist()
        )
        for row in expected.itertuples():
            link = shard / "data/test" / row.AUDIO_FILENAME
            assert link.is_symlink()
            assert link.resolve() == (work / "corpus/audio" / row.AUDIO_FILENAME).resolve()
        shard_sample = pd.read_csv(shard / "data/sample_submission.csv", dtype=str)
        assert shard_sample.ID.tolist() == expected.COMBINED_ID.tolist()
        assert not (shard / "data/sample_submission.csv").is_symlink()
        assert list((shard / "output").iterdir()) == []


def test_data_guard_fails_before_prepare_writes(tmp_path):
    config, train, _, locked, package = _fixture_tree(tmp_path)
    locked_frame = pd.read_csv(locked)
    locked_frame.loc[0, "VOICE_SOURCE_ID"] = "train-source-a"
    locked_frame.to_csv(locked, index=False)
    work = tmp_path / "must-not-exist"
    with pytest.raises(ValueError, match="TRAIN/EVAL LEAKAGE"):
        prepare_run(
            config, package, ROOT / "src/pipeline.py", work,
            [str(train)], num_shards=1,
        )
    assert not work.exists()


@pytest.mark.parametrize(
    "forbidden_role",
    ("locked_eval", "ood_holdout", "stress_eval", "retrospective_diagnostic", "invalid_eval"),
)
def test_forbidden_role_is_never_materialized(tmp_path, forbidden_role):
    config, _, _, locked, package = _fixture_tree(tmp_path)
    config.write_text(
        "train:\n  - data/train/truth.csv\n"
        "development:\n  - data/dev/truth.csv\n"
        f"{forbidden_role}:\n  - data/locked/truth.csv\n",
        encoding="utf-8",
    )
    work = tmp_path / "must-not-exist"
    with pytest.raises(ValueError, match="forbidden for caching"):
        prepare_run(
            config, package, ROOT / "src/pipeline.py", work,
            [str(locked)], num_shards=1,
        )
    assert not work.exists()


def test_locked_policy_accepts_only_separately_registered_locked_eval(tmp_path):
    config, train, _, locked, package = _fixture_tree(tmp_path)
    work = tmp_path / "locked-run"
    prepare_run(
        config, package, ROOT / "src/pipeline.py", work,
        [str(locked)], num_shards=1, cache_policy="one_shot_locked",
    )
    manifest = json.loads(
        (work / "run_manifest.json").read_text("utf-8")
    )
    assert manifest["cache_policy"] == "one_shot_locked"
    assert manifest["allowed_roles"] == ["locked_eval"]
    with pytest.raises(ValueError, match="forbidden for caching"):
        prepare_run(
            config, package, ROOT / "src/pipeline.py", tmp_path / "bad-run",
            [str(train)], num_shards=1, cache_policy="one_shot_locked",
        )


@pytest.mark.parametrize("role", ("router_train", "training_validation"))
def test_non_train_development_roles_are_not_cache_inputs(tmp_path, role):
    config, _, _, locked, package = _fixture_tree(tmp_path)
    config.write_text(
        "development:\n  - data/dev/truth.csv\n"
        f"{role}:\n  - data/locked/truth.csv\n",
        encoding="utf-8",
    )
    work = tmp_path / "must-not-exist"
    with pytest.raises(ValueError, match="authorized train/development role"):
        prepare_run(
            config, package, ROOT / "src/pipeline.py", work,
            [str(locked)], num_shards=1,
        )
    assert not work.exists()


def _write_shard_outputs(work: Path) -> None:
    mapping = pd.read_csv(work / "corpus/dataset_map.csv", dtype=str)
    ids = mapping.COMBINED_ID.tolist()
    for shard_index in range(2):
        selected = ids[shard_index::2]
        output = work / "shards" / f"shard_{shard_index:03d}" / "output"
        with (output / "submission.csv").open("w", newline="", encoding="utf-8") as handle:
            columns = [
                "ID", "FILE_FAKE_PROB", "VOICE_FAKE_PROB", "MUSIC_FAKE_PROB",
                "VOICE_PRESENT_PROB", "MUSIC_PRESENT_PROB",
            ]
            writer = csv.DictWriter(handle, fieldnames=columns)
            writer.writeheader()
            for item_id in selected:
                value = (ids.index(item_id) + 1) / 10
                writer.writerow({"ID": item_id, **{column: value for column in columns[1:]}})
        rows = len(selected)
        global_values = np.asarray([ids.index(item_id) + 1 for item_id in selected], dtype=np.float32)
        xlsr = np.zeros((rows, 1, XLSR_EMBEDDING_DIM), dtype=np.float32)
        xlsr[:, 0, 0] = global_values
        _write_deterministic_npz(output / "xlsr_windows.npz", {
            "ids": np.asarray(selected),
            "embeddings": xlsr,
            "mask": np.ones((rows, 1), dtype=np.bool_),
            "starts": np.zeros((rows, 1), dtype=np.int64),
            "window": np.asarray(64_000, dtype=np.int64),
            "sample_rate": np.asarray(16_000, dtype=np.int64),
        })
        temporal = np.zeros((rows, 1, 1, 2, 2, 3), dtype=np.float32)
        temporal[:, 0, 0, 0, 0, 0] = global_values
        _write_deterministic_npz(output / "eat_patch_graph.npz", {
            "ids": np.asarray(selected),
            "temporal": temporal,
            "spectral": np.zeros((rows, 1, 1, 2, 1, 3), dtype=np.float32),
            "view_mask": np.ones((rows, 1), dtype=np.bool_),
            "projection": np.ones((3, 2), dtype=np.float32),
            "layers": np.asarray([1], dtype=np.int64),
        })
        features = np.zeros((rows, 1, 2, 1, 4, 3), dtype=np.float32)
        features[:, 0, 0, 0, 0, 0] = global_values
        _write_deterministic_npz(output / "spear_component_bins.npz", {
            "ids": np.asarray(selected),
            "features": features,
            "mask": np.ones((rows, 1, 2), dtype=np.bool_),
            "projection": np.ones((3, 2), dtype=np.float32),
            "layers": np.asarray([0], dtype=np.int16),
            "bins": np.asarray(2, dtype=np.int64),
        })


def test_merge_restores_dataset_rows_and_all_three_cache_schemas(tmp_path):
    config, train, dev, _, package = _fixture_tree(tmp_path)
    work = tmp_path / "run"
    prepare_run(
        config, package, ROOT / "src/pipeline.py", work,
        [str(train), str(dev)], num_shards=2,
    )
    _write_shard_outputs(work)
    result = merge_run(work)

    datasets = sorted((result / "datasets").iterdir())
    assert len(datasets) == 2
    train_dataset = result / "datasets/train"
    first_predictions = pd.read_csv(train_dataset / "exact_v47_predictions.csv")
    assert first_predictions.ID.tolist() == ["train-a", "train-b"]
    assert first_predictions.FILE_FAKE_PROB.tolist() == pytest.approx([0.1, 0.2])
    xlsr = _load_validated_npz(train_dataset / "xlsr_windows.npz", ["train-a", "train-b"])
    assert xlsr["embeddings"][:, 0, 0].tolist() == [1.0, 2.0]
    with np.load(train_dataset / "eat_patch_graph.npz", allow_pickle=False) as eat:
        assert set(eat.files) == {"ids", "temporal", "spectral", "view_mask", "projection", "layers"}
        assert eat["ids"].tolist() == ["train-a", "train-b"]
        assert eat["temporal"][:, 0, 0, 0, 0, 0].tolist() == [1.0, 2.0]
    with np.load(train_dataset / "spear_component_bins.npz", allow_pickle=False) as spear:
        assert set(spear.files) == {"ids", "features", "mask", "projection", "layers", "bins"}
        assert spear["features"][:, 0, 0, 0, 0, 0].tolist() == [1.0, 2.0]
    compatibility = {
        "anchor/train/predictions.csv": train_dataset / "exact_v47_predictions.csv",
        "xlsr/train/features.npz": train_dataset / "xlsr_windows.npz",
        "eat/train/features.npz": train_dataset / "eat_patch_graph.npz",
        "spear/train/features.npz": train_dataset / "spear_component_bins.npz",
    }
    for relative, target in compatibility.items():
        link = result / relative
        assert link.is_symlink()
        assert link.resolve() == target.resolve()
    frame = pd.DataFrame({"ID": ["train-a", "train-b"]})
    eat_cache = TRAINING.load_eat_cache(result / "eat", "train", frame)
    assert eat_cache["temporal"][:, 0, 0, 0, 0, 0].tolist() == [1.0, 2.0]
    spear_cache = TRAINING.load_spear_caches(result / "spear", ["train", "dev"])
    assert spear_cache["train"]["features"][:, 0, 0, 0, 0, 0].tolist() == [1.0, 2.0]
    xlsr_path = TRAINING.resolve_cache_file(result / "xlsr", "train", "xlsr")
    xlsr_cache = TRAINING.load_xlsr_window_cache(xlsr_path, frame.ID)
    assert xlsr_cache["embeddings"][:, 0, 0].tolist() == [1.0, 2.0]
    anchor_path = TRAINING.resolve_cache_file(result / "anchor", "train", "anchor")
    anchor = TRAINING.load_anchor_cache(anchor_path, frame.ID)
    assert anchor["authenticity"].shape == (2, 3)
    provenance = (result / "merge_provenance.json").read_text("utf-8")
    assert "eat_patch_graph_sha256" in provenance
    assert "spear_component_bins_sha256" in provenance


def test_merge_rejects_wrong_round_robin_ids_without_output(tmp_path):
    config, train, dev, _, package = _fixture_tree(tmp_path)
    work = tmp_path / "run"
    prepare_run(
        config, package, ROOT / "src/pipeline.py", work,
        [str(train), str(dev)], num_shards=2,
    )
    _write_shard_outputs(work)
    path = work / "shards/shard_000/output/submission.csv"
    frame = pd.read_csv(path)
    frame.loc[0, "ID"] = "wrong"
    frame.to_csv(path, index=False)
    with pytest.raises(ValueError, match="Prediction ID/order mismatch"):
        merge_run(work)
    assert not (work / "cache").exists()


def test_merge_rechecks_audio_hashes(tmp_path):
    config, train, dev, _, package = _fixture_tree(tmp_path)
    work = tmp_path / "run"
    prepare_run(
        config, package, ROOT / "src/pipeline.py", work,
        [str(train), str(dev)], num_shards=2,
    )
    _write_shard_outputs(work)
    (train.parent / "audio" / "train-a.wav").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="Audio hash changed"):
        merge_run(work)
    assert not (work / "cache").exists()
