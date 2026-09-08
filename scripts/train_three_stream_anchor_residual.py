#!/usr/bin/env python3
"""Train/validate the EAT+SPEAR+(optional XLS-R) anchor-residual head.

This is an offline experiment scaffold only.  It reads frozen feature and
anchor caches, and it will only admit truth manifests declared under ``train``
and ``development`` in ``configs/data_partitions.yaml``.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import random
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from data_guard import (  # noqa: E402
    assert_development_eval_separation,
    assert_no_locked_eval_leakage,
)
from evaluate_diagnostic import score_frame  # noqa: E402
from train_eat_patch_graph import load_cache as load_eat_cache, sample_weights  # noqa: E402
from train_spear_temporal_bin_mil import load_archive as load_spear_archive  # noqa: E402
from train_unified_dual_ssl_head import targets  # noqa: E402
from three_stream_anchor_residual import (  # noqa: E402
    XLSR_WINDOW_DIMENSION,
    ThreeStreamAnchorResidualHead,
    initial_model_consistency_loss,
    robust_anchor_residual_loss,
)


ALLOWED_ROLES = ("train", "development")
BASE_KEYS = (
    "temporal", "spectral", "eat_mask", "spear", "spear_mask",
    "anchor_authenticity", "anchor_presence",
)
XLSR_KEYS = ("xlsr_embeddings", "xlsr_mask")
AUTHENTICITY_COLUMNS = (
    "VOICE_FAKE_PROB", "MUSIC_FAKE_PROB", "FILE_FAKE_PROB",
)
PRESENCE_COLUMNS = ("VOICE_PRESENT_PROB", "MUSIC_PRESENT_PROB")
LAYOUT_COLUMNS = ("MIX_MODE", "LAYOUT", "CONDITION")
CHANNEL_COLUMNS = ("CHANNEL", "CODEC")
VOICE_GENERATOR_COLUMNS = (
    "VOICE_GENERATOR", "FIRST_GENERATOR", "SECOND_GENERATOR", "ATTACK",
)
MUSIC_GENERATOR_COLUMNS = ("MUSIC_GENERATOR",)
VOICE_IDENTITY_COLUMNS = (
    "VOICE_SOURCE_ID", "FIRST_SOURCE_ID", "SECOND_SOURCE_ID",
    "VOICE_SPEAKER", "VOICE_GROUP", "FIRST_GROUP", "SECOND_GROUP",
    "VOICE_ARCHIVE_ID", "VOICE_ARCHIVE_MEMBER",
)
MUSIC_IDENTITY_COLUMNS = (
    "MUSIC_SOURCE_ID", "MUSIC_GROUP", "MUSIC_GROUP_ID",
    "MUSIC_ARCHIVE_ID", "MUSIC_ARCHIVE_MEMBER",
)
PARENT_IDENTITY_COLUMNS = (
    "PARENT_ID", "MIXTURE_ID", "PAIR_GROUP", "GROUP_ID", "SOURCE_FILE",
)
STRICT_IDENTITY_COLUMNS = tuple(dict.fromkeys(
    ("ID",)
    + VOICE_IDENTITY_COLUMNS
    + MUSIC_IDENTITY_COLUMNS
    + PARENT_IDENTITY_COLUMNS
    + ("SPEAKER", "FMA_TRACK_ID", "ORIGINAL_AUDIO", "BASE_ID")
))


@dataclass(frozen=True)
class Partition:
    name: str
    truth_path: Path
    role: str


def authorized_partitions(
    partition_config: Path,
    role: str,
    requested: list[str] | tuple[str, ...] | None = None,
) -> list[Partition]:
    """Resolve dataset names exclusively from an authorized config role."""
    if role not in ALLOWED_ROLES:
        raise ValueError(
            f"role {role!r} is not allowed; choose one of {ALLOWED_ROLES}"
        )
    config = yaml.safe_load(partition_config.read_text("utf-8")) or {}
    if not isinstance(config, dict):
        raise ValueError("partition config must be a role-to-path mapping")
    declared = config.get(role, []) or []
    if not isinstance(declared, list) or not all(isinstance(item, str) for item in declared):
        raise ValueError(f"partition role {role!r} must be a list of paths")
    config_dir = partition_config.resolve().parent
    root = config_dir.parent if config_dir.name == "configs" else config_dir
    by_name: dict[str, Partition] = {}
    for relative in declared:
        path = (root / relative).resolve()
        name = path.parent.name
        if name in by_name:
            raise ValueError(f"role {role!r} declares dataset {name!r} more than once")
        by_name[name] = Partition(name, path, role)
    if requested is None:
        return list(by_name.values())
    if not requested:
        raise ValueError(f"at least one {role} dataset is required")
    if len(set(requested)) != len(requested):
        raise ValueError(f"duplicate {role} dataset request")
    unknown = sorted(set(requested).difference(by_name))
    if unknown:
        raise ValueError(
            f"datasets are not declared under {role!r}: {unknown}"
        )
    return [by_name[name] for name in requested]


def guard_partitions(
    train: list[Partition], development: list[Partition], partition_config: Path,
) -> None:
    """Run the repository guard after role resolution and before cache reads."""
    if not train or not development:
        raise ValueError("both train and development partitions are required")
    if any(item.role != "train" for item in train):
        raise ValueError("training inputs must come from the train role")
    if any(item.role != "development" for item in development):
        raise ValueError("selection inputs must come from the development role")
    assert_development_eval_separation(partition_config)
    for item in train:
        assert_no_locked_eval_leakage(item.truth_path, partition_config)


def load_truth(partition: Partition) -> pd.DataFrame:
    if not partition.truth_path.is_file():
        raise FileNotFoundError(partition.truth_path)
    frame = pd.read_csv(partition.truth_path, dtype={"ID": str})
    required = {
        "ID", "FILE_FAKE", "VOICE_PRESENT", "MUSIC_PRESENT",
        "VOICE_FAKE", "MUSIC_FAKE",
    }
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(
            f"{partition.truth_path} misses truth columns: {sorted(missing)}"
        )
    ids = frame["ID"]
    if ids.isna().any() or ids.str.strip().eq("").any() or ids.duplicated().any():
        raise ValueError(f"{partition.truth_path} contains invalid or duplicate IDs")
    frame = frame.reset_index(drop=True)
    frame["DATASET"] = partition.name
    return frame


def align_ids(
    cached_ids: np.ndarray,
    expected_ids: pd.Series | np.ndarray,
    source: str,
    *,
    exact: bool = False,
) -> np.ndarray:
    ids = np.asarray(cached_ids).astype(str)
    if ids.ndim != 1:
        raise ValueError(f"{source} IDs must be one-dimensional")
    if len(set(ids.tolist())) != len(ids):
        raise ValueError(f"{source} contains duplicate IDs")
    expected = np.asarray(expected_ids).astype(str)
    if expected.ndim != 1 or len(set(expected.tolist())) != len(expected):
        raise ValueError("truth IDs must be one-dimensional and unique")
    index = {sample_id: offset for offset, sample_id in enumerate(ids)}
    missing = sorted(set(expected).difference(index))
    extra = sorted(set(ids).difference(expected)) if exact else []
    if missing or extra:
        raise ValueError(
            f"{source}/truth ID mismatch: missing={missing[:5]}, extra={extra[:5]}"
        )
    return np.asarray([index[sample_id] for sample_id in expected], dtype=np.int64)


def _positive_scalar(archive: np.lib.npyio.NpzFile, key: str, path: Path) -> int:
    value = archive[key]
    if value.shape != () or value.dtype.kind not in "iu":
        raise ValueError(f"{path}: {key} must be a scalar integer")
    result = int(value)
    if result <= 0:
        raise ValueError(f"{path}: {key} must be positive")
    return result


def load_xlsr_window_cache(
    path: Path, expected_ids: pd.Series | np.ndarray,
) -> dict[str, np.ndarray | int]:
    """Load the frozen ordered-window schema and reject silent misalignment."""
    required = {
        "ids", "embeddings", "mask", "starts", "window", "sample_rate",
    }
    with np.load(path, allow_pickle=False) as archive:
        missing = required.difference(archive.files)
        if missing:
            raise ValueError(f"{path} misses XLS-R fields: {sorted(missing)}")
        ids = archive["ids"].astype(str)
        embeddings = archive["embeddings"]
        mask = archive["mask"]
        starts = archive["starts"]
        window = _positive_scalar(archive, "window", path)
        sample_rate = _positive_scalar(archive, "sample_rate", path)

    if embeddings.ndim != 3 or embeddings.shape[-1] != XLSR_WINDOW_DIMENSION:
        raise ValueError(
            f"{path}: embeddings must have shape [N,V,{XLSR_WINDOW_DIMENSION}]"
        )
    if embeddings.shape[0] != len(ids):
        raise ValueError(f"{path}: XLS-R ID and embedding counts differ")
    if mask.dtype != np.bool_:
        raise TypeError(f"{path}: mask must be boolean")
    if mask.shape != embeddings.shape[:2]:
        raise ValueError(f"{path}: mask and embeddings axes differ")
    if starts.dtype.kind not in "iu" or starts.shape != mask.shape:
        raise ValueError(f"{path}: starts must be integer [N,V]")
    if embeddings.dtype.kind != "f" or not np.isfinite(embeddings).all():
        raise ValueError(f"{path}: embeddings must be finite floating point")
    if mask.shape[1] == 0 or not mask.any(axis=1).all():
        raise ValueError(f"{path}: every item needs a valid XLS-R window")
    if np.any(mask[:, 1:] & ~mask[:, :-1]):
        raise ValueError(f"{path}: XLS-R valid windows must form a prefix")
    if np.any(starts[mask] < 0) or np.any(starts[~mask] != -1):
        raise ValueError(f"{path}: mask must equal starts != -1")
    adjacent_valid = mask[:, 1:] & mask[:, :-1]
    if np.any((np.diff(starts, axis=1) <= 0) & adjacent_valid):
        raise ValueError(f"{path}: valid XLS-R starts must be strictly increasing")
    if np.any(embeddings[~mask] != 0):
        raise ValueError(f"{path}: padded XLS-R embeddings must be zero")

    order = align_ids(ids, expected_ids, str(path), exact=True)
    return {
        "embeddings": embeddings[order],
        "mask": mask[order],
        "starts": starts[order],
        "window": window,
        "sample_rate": sample_rate,
    }


def _probability_logits(values: np.ndarray, source: str) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if not np.isfinite(values).all() or np.any((values < 0) | (values > 1)):
        raise ValueError(f"{source} probabilities must be finite and in [0,1]")
    clipped = np.clip(values, 1e-6, 1 - 1e-6)
    return np.log(clipped / (1 - clipped)).astype(np.float32)


def load_anchor_cache(
    path: Path, expected_ids: pd.Series | np.ndarray,
) -> dict[str, np.ndarray]:
    """Load anchor logits directly from NPZ or convert a standard prediction CSV."""
    if path.suffix == ".npz":
        required = {"ids", "authenticity_logits", "presence_logits"}
        with np.load(path, allow_pickle=False) as archive:
            missing = required.difference(archive.files)
            if missing:
                raise ValueError(f"{path} misses anchor fields: {sorted(missing)}")
            ids = archive["ids"].astype(str)
            authenticity = archive["authenticity_logits"]
            presence = archive["presence_logits"]
    elif path.suffix == ".csv":
        frame = pd.read_csv(path, dtype={"ID": str})
        required = {"ID", *AUTHENTICITY_COLUMNS, *PRESENCE_COLUMNS}
        missing = required.difference(frame.columns)
        if missing:
            raise ValueError(f"{path} misses anchor columns: {sorted(missing)}")
        ids = frame["ID"].astype(str).to_numpy()
        authenticity = _probability_logits(
            frame[list(AUTHENTICITY_COLUMNS)].to_numpy(), str(path)
        )
        presence = _probability_logits(
            frame[list(PRESENCE_COLUMNS)].to_numpy(), str(path)
        )
    else:
        raise ValueError(f"anchor cache must be .npz or .csv: {path}")
    if authenticity.shape != (len(ids), 3) or presence.shape != (len(ids), 2):
        raise ValueError(f"{path}: anchor logits must be [N,3] and [N,2]")
    if authenticity.dtype.kind != "f" or presence.dtype.kind != "f":
        raise TypeError(f"{path}: anchor logits must be floating point")
    if not np.isfinite(authenticity).all() or not np.isfinite(presence).all():
        raise ValueError(f"{path}: anchor logits contain non-finite values")
    order = align_ids(ids, expected_ids, str(path), exact=True)
    return {
        "authenticity": authenticity[order].astype(np.float32, copy=False),
        "presence": presence[order].astype(np.float32, copy=False),
    }


def resolve_cache_file(root: Path, name: str, kind: str) -> Path:
    suffixes = (
        (root / f"{name}.npz", root / name / "features.npz")
        if kind == "xlsr"
        else (
            root / f"{name}.npz", root / f"{name}.csv",
            root / name / "anchor.npz", root / name / "predictions.csv",
        )
    )
    existing = [path for path in suffixes if path.is_file()]
    if len(existing) != 1:
        raise FileNotFoundError(
            f"expected exactly one {kind} cache for {name}; found {existing}"
        )
    return existing[0]


def load_spear_caches(
    root: Path, dataset_names: list[str],
) -> dict[str, dict[str, np.ndarray]]:
    """Load per-dataset SPEAR caches, with the established union fallback."""
    if not dataset_names or len(set(dataset_names)) != len(dataset_names):
        raise ValueError("SPEAR dataset names must be nonempty and unique")
    paths = {name: root / name / "features.npz" for name in dataset_names}
    existing = {name for name, path in paths.items() if path.is_file()}
    if not existing:
        return load_spear_archive(root)
    missing_paths = sorted(set(dataset_names).difference(existing))
    if missing_paths:
        raise FileNotFoundError(
            "per-dataset SPEAR cache root is incomplete for: "
            f"{missing_paths}"
        )

    result: dict[str, dict[str, np.ndarray]] = {}
    reference: tuple[np.ndarray, np.ndarray, int] | None = None
    required = {"ids", "features", "mask", "projection", "layers", "bins"}
    for name in dataset_names:
        path = paths[name]
        with np.load(path, allow_pickle=False) as archive:
            missing = required.difference(archive.files)
            if missing:
                raise ValueError(f"{path} misses SPEAR fields: {sorted(missing)}")
            ids = archive["ids"].astype(str)
            features = archive["features"]
            mask = archive["mask"]
            projection = archive["projection"].astype(np.float32)
            layers = archive["layers"].astype(np.int64)
            bins_value = archive["bins"]
        if bins_value.shape != () or bins_value.dtype.kind not in "iu":
            raise ValueError(f"{path}: bins must be a scalar integer")
        bins = int(bins_value)
        if bins <= 0:
            raise ValueError(f"{path}: bins must be positive")
        if projection.ndim != 2 or min(projection.shape) <= 0:
            raise ValueError(f"{path}: projection must be a nonempty matrix")
        if (
            layers.ndim != 1 or not len(layers)
            or np.any(layers[1:] <= layers[:-1])
        ):
            raise ValueError(f"{path}: layers must be nonempty and increasing")
        if ids.ndim != 1 or len(set(ids.tolist())) != len(ids):
            raise ValueError(f"{path}: SPEAR IDs must be one-dimensional and unique")
        if (
            features.ndim != 6 or features.shape[0] != len(ids)
            or features.shape[2] != bins or features.shape[3] != len(layers)
        ):
            raise ValueError(f"{path}: SPEAR feature axes differ from metadata")
        if mask.dtype != np.bool_ or mask.shape != features.shape[:3]:
            raise ValueError(f"{path}: SPEAR mask must be boolean [N,V,bins]")
        if features.dtype.kind != "f" or not np.isfinite(features).all():
            raise ValueError(f"{path}: SPEAR features must be finite floating point")
        current = (projection, layers, bins)
        if reference is None:
            reference = current
        elif not (
            np.array_equal(reference[0], projection)
            and np.array_equal(reference[1], layers)
            and reference[2] == bins
        ):
            raise ValueError("per-dataset SPEAR cache metadata differs")
        result[name] = {"ids": ids, "features": features, "mask": mask}
    assert reference is not None
    result["__metadata__"] = {
        "projection": reference[0],
        "layers": reference[1],
        "bins": np.asarray(reference[2]),
    }
    return result


def load_block(
    partition: Partition,
    eat_root: Path,
    spear_cache: dict[str, dict[str, np.ndarray]],
    anchor_root: Path,
    xlsr_root: Path | None,
) -> tuple[pd.DataFrame, dict[str, np.ndarray], dict[str, object]]:
    frame = load_truth(partition)
    eat = load_eat_cache(eat_root, partition.name, frame)
    if partition.name not in spear_cache:
        raise FileNotFoundError(f"SPEAR cache has no dataset {partition.name}")
    spear_source = spear_cache[partition.name]
    spear_order = align_ids(
        spear_source["ids"], frame["ID"], f"SPEAR {partition.name}"
    )
    anchor = load_anchor_cache(
        resolve_cache_file(anchor_root, partition.name, "anchor"), frame["ID"]
    )
    block = {
        "temporal": eat["temporal"],
        "spectral": eat["spectral"],
        "eat_mask": eat["view_mask"],
        "spear": spear_source["features"][spear_order],
        "spear_mask": spear_source["mask"][spear_order],
        "anchor_authenticity": anchor["authenticity"],
        "anchor_presence": anchor["presence"],
    }
    metadata: dict[str, object] = {
        "eat_projection": eat["projection"],
        "eat_layers": eat["layers"],
    }
    if xlsr_root is not None:
        xlsr = load_xlsr_window_cache(
            resolve_cache_file(xlsr_root, partition.name, "xlsr"), frame["ID"]
        )
        block["xlsr_embeddings"] = xlsr["embeddings"]
        block["xlsr_mask"] = xlsr["mask"]
        metadata.update({
            "xlsr_window": xlsr["window"],
            "xlsr_sample_rate": xlsr["sample_rate"],
        })
    batch = len(frame)
    if any(len(block[key]) != batch for key in block):
        raise ValueError(f"aligned cache count differs for {partition.name}")
    return frame, block, metadata


def concatenate(blocks: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    if not blocks:
        raise ValueError("no cache blocks were supplied")
    keys = tuple(blocks[0])
    if any(tuple(block) != keys for block in blocks[1:]):
        raise ValueError("cache blocks do not expose identical streams")
    for key in keys:
        tail = blocks[0][key].shape[1:]
        if any(block[key].shape[1:] != tail for block in blocks[1:]):
            raise ValueError(f"cache blocks have incompatible {key} shapes")
    return {key: np.concatenate([block[key] for block in blocks]) for key in keys}


def pad_xlsr_window_axis(blocks: list[dict[str, np.ndarray]]) -> None:
    """Pad per-corpus XLS-R windows to one global axis before concatenation.

    Runner artifacts intentionally pad only to the longest file in each
    corpus.  Training and development must nevertheless share one model
    maximum, so this operation is applied jointly and only adds masked zeros.
    """
    if not blocks:
        raise ValueError("no cache blocks were supplied")
    enabled = ["xlsr_embeddings" in block for block in blocks]
    if not any(enabled):
        return
    if not all(enabled) or any("xlsr_mask" not in block for block in blocks):
        raise ValueError("XLS-R streams must be present in every cache block")
    maximum = max(block["xlsr_embeddings"].shape[1] for block in blocks)
    for block in blocks:
        embeddings = block["xlsr_embeddings"]
        mask = block["xlsr_mask"]
        if embeddings.ndim != 3 or mask.shape != embeddings.shape[:2]:
            raise ValueError("XLS-R cache axes are malformed before padding")
        missing = maximum - embeddings.shape[1]
        if missing:
            block["xlsr_embeddings"] = np.pad(
                embeddings, ((0, 0), (0, missing), (0, 0)),
                mode="constant", constant_values=0,
            )
            block["xlsr_mask"] = np.pad(
                mask, ((0, 0), (0, missing)),
                mode="constant", constant_values=False,
            )


def validate_numpy_block(block: dict[str, np.ndarray]) -> None:
    """Validate every cached row without running the neural head."""
    required = set(BASE_KEYS)
    missing = required.difference(block)
    if missing:
        raise ValueError(f"feature block misses fields: {sorted(missing)}")
    temporal, spectral = block["temporal"], block["spectral"]
    if temporal.ndim != 6 or spectral.ndim != 6:
        raise ValueError("EAT caches must have [N,V,L,2,nodes,D] axes")
    batch, views, layers, statistics = temporal.shape[:4]
    if (
        batch <= 0 or views <= 0 or layers <= 0 or statistics != 2
        or spectral.shape[:4] != (batch, views, layers, statistics)
        or spectral.shape[-1] != temporal.shape[-1]
    ):
        raise ValueError("EAT temporal/spectral cache axes are incompatible")
    eat_mask = block["eat_mask"]
    if eat_mask.dtype != np.bool_ or eat_mask.shape != (batch, views):
        raise ValueError("EAT mask must be boolean [N,V]")
    if not eat_mask.any(axis=1).all() or np.any(eat_mask[:, 1:] & ~eat_mask[:, :-1]):
        raise ValueError("EAT valid views must form a nonempty prefix")

    spear, spear_mask = block["spear"], block["spear_mask"]
    if spear.ndim != 6 or spear.shape[:2] != (batch, views):
        raise ValueError("SPEAR cache must be [N,V,bins,L,stats,D]")
    if spear_mask.dtype != np.bool_ or spear_mask.shape != spear.shape[:3]:
        raise ValueError("SPEAR mask must be boolean [N,V,bins]")
    if np.any(spear_mask[:, :, 1:] & ~spear_mask[:, :, :-1]):
        raise ValueError("SPEAR valid bins must form a prefix")
    if not spear_mask.any(axis=(1, 2)).all():
        raise ValueError("every row must contain at least one valid SPEAR bin")

    authenticity = block["anchor_authenticity"]
    presence = block["anchor_presence"]
    if authenticity.shape != (batch, 3) or presence.shape != (batch, 2):
        raise ValueError("anchor logit caches must be [N,3] and [N,2]")
    floating = (temporal, spectral, spear, authenticity, presence)
    if any(values.dtype.kind != "f" for values in floating):
        raise TypeError("feature and anchor caches must be floating point")
    if any(not np.isfinite(values).all() for values in floating):
        raise ValueError("feature or anchor cache contains non-finite values")

    has_embeddings = "xlsr_embeddings" in block
    has_mask = "xlsr_mask" in block
    if has_embeddings != has_mask:
        raise ValueError("XLS-R embeddings and mask must be supplied together")
    if has_embeddings:
        embeddings, mask = block["xlsr_embeddings"], block["xlsr_mask"]
        if (
            embeddings.ndim != 3
            or embeddings.shape[0] != batch
            or embeddings.shape[-1] != XLSR_WINDOW_DIMENSION
        ):
            raise ValueError("XLS-R cache must be [N,V,1920]")
        if mask.dtype != np.bool_ or mask.shape != embeddings.shape[:2]:
            raise ValueError("XLS-R mask must be boolean [N,V]")
        if not mask.any(axis=1).all() or np.any(mask[:, 1:] & ~mask[:, :-1]):
            raise ValueError("XLS-R valid windows must form a nonempty prefix")
        if embeddings.dtype.kind != "f" or not np.isfinite(embeddings).all():
            raise ValueError("XLS-R embeddings must be finite floating point")


def validate_metadata(metadata: list[dict[str, object]]) -> None:
    if not metadata:
        raise ValueError("no cache metadata was supplied")
    reference = metadata[0]
    for current in metadata[1:]:
        if set(current) != set(reference):
            raise ValueError("optional stream metadata differs across datasets")
        for key in reference:
            left, right = reference[key], current[key]
            if isinstance(left, np.ndarray):
                equal = isinstance(right, np.ndarray) and np.array_equal(left, right)
            else:
                equal = left == right
            if not equal:
                raise ValueError(f"cache metadata differs for {key}")


def complete_pair_group_quartets(frame: pd.DataFrame) -> np.ndarray:
    """Return canonical RR/RF/FR/FF index quartets when metadata is complete."""
    if "PAIR_GROUP" not in frame:
        return np.empty((0, 4), dtype=np.int64)
    quartets: list[tuple[int, int, int, int]] = []
    available = (
        frame["PAIR_GROUP"].notna()
        & frame["PAIR_GROUP"].astype(str).str.strip().ne("")
    )
    group_columns = ["PAIR_GROUP"]
    if "DATASET" in frame:
        group_columns.insert(0, "DATASET")
    for _, group in frame.loc[available].groupby(group_columns, sort=True):
        cells: dict[tuple[int, int], list[int]] = {}
        for index, row in group.iterrows():
            if (
                int(row.get("VOICE_PRESENT", 0)) != 1
                or int(row.get("MUSIC_PRESENT", 0)) != 1
                or pd.isna(row.get("VOICE_FAKE"))
                or pd.isna(row.get("MUSIC_FAKE"))
            ):
                continue
            cell = (int(row["VOICE_FAKE"]), int(row["MUSIC_FAKE"]))
            if cell not in {(0, 0), (0, 1), (1, 0), (1, 1)}:
                raise ValueError("PAIR_GROUP authenticity labels must be binary")
            cells.setdefault(cell, []).append(int(index))
        order = ((0, 0), (0, 1), (1, 0), (1, 1))
        if not all(cell in cells for cell in order):
            continue
        for offset in range(min(len(cells[cell]) for cell in order)):
            quartets.append(tuple(cells[cell][offset] for cell in order))
    return np.asarray(quartets, dtype=np.int64).reshape(-1, 4)


def _same_channel_targets(frame: pd.DataFrame, first: int, second: int) -> bool:
    left, right = frame.loc[first], frame.loc[second]
    for column in ("FILE_FAKE", "VOICE_PRESENT", "MUSIC_PRESENT"):
        if int(left[column]) != int(right[column]):
            return False
    for component in ("VOICE", "MUSIC"):
        if int(left[f"{component}_PRESENT"]) == 0:
            continue
        if (
            pd.isna(left[f"{component}_FAKE"])
            or pd.isna(right[f"{component}_FAKE"])
            or int(left[f"{component}_FAKE"]) != int(right[f"{component}_FAKE"])
        ):
            return False
    return True


def channel_consistency_groups(
    frame: pd.DataFrame,
) -> tuple[list[np.ndarray], np.ndarray]:
    """Find clean/codec groups and pairs from MIXTURE_ID or PARENT_ID.

    A corpus lacking these fields simply returns no auxiliary groups.  A
    declared relation whose labels change is rejected because consistency on
    such a pair would silently train against the truth.
    """
    groups: list[np.ndarray] = []
    pairs: set[tuple[int, int]] = set()
    id_to_indices: dict[str, list[int]] = {}
    if "ID" in frame:
        for index, sample_id in frame["ID"].items():
            text = str(sample_id)
            id_to_indices.setdefault(text, []).append(int(index))

    if "PARENT_ID" in frame:
        children: dict[int, list[int]] = {}
        for child, parent in frame["PARENT_ID"].items():
            if pd.isna(parent) or not str(parent).strip():
                continue
            candidates = id_to_indices.get(str(parent), [])
            if "DATASET" in frame and candidates:
                dataset = frame.loc[child, "DATASET"]
                same_dataset = [
                    index for index in candidates
                    if frame.loc[index, "DATASET"] == dataset
                ]
                if same_dataset:
                    candidates = same_dataset
            # Ambiguous cross-corpus IDs are not enough evidence to impose a
            # consistency relation; safely skip only this auxiliary pair.
            if len(candidates) != 1:
                continue
            clean = candidates[0]
            child = int(child)
            if clean == child:
                continue
            if not _same_channel_targets(frame, clean, child):
                raise ValueError("PARENT_ID channel pair has inconsistent targets")
            pairs.add((clean, child))
            children.setdefault(clean, []).append(child)
        groups.extend(
            np.asarray([clean, *sorted(items)], dtype=np.int64)
            for clean, items in children.items()
        )

    if "MIXTURE_ID" in frame:
        selected = (
            frame["MIXTURE_ID"].notna()
            & frame["MIXTURE_ID"].astype(str).str.strip().ne("")
        )
        group_columns = ["MIXTURE_ID"]
        if "DATASET" in frame:
            group_columns.insert(0, "DATASET")
        for _, block in frame.loc[selected].groupby(group_columns, sort=True):
            if len(block) < 2:
                continue
            clean_candidates: list[int] = []
            if "CHANNEL" in block:
                normalized = block["CHANNEL"].fillna("").astype(str).str.lower()
                clean_candidates.extend(block.index[normalized.isin({
                    "clean", "clean_flac", "original", "none",
                })].astype(int).tolist())
            if not clean_candidates and "PARENT_ID" in block:
                no_parent = block["PARENT_ID"].isna() | block["PARENT_ID"].astype(str).str.strip().eq("")
                clean_candidates.extend(block.index[no_parent].astype(int).tolist())
            clean_candidates = sorted(set(clean_candidates))
            if len(clean_candidates) != 1:
                continue
            clean = clean_candidates[0]
            codecs = sorted(int(index) for index in block.index if int(index) != clean)
            if not codecs:
                continue
            for codec in codecs:
                if not _same_channel_targets(frame, clean, codec):
                    raise ValueError("MIXTURE_ID channel group has inconsistent targets")
                pairs.add((clean, codec))
            groups.append(np.asarray([clean, *codecs], dtype=np.int64))

    # PARENT_ID and MIXTURE_ID often describe the same clean/codec family.
    # Keep one atomic group and one copy of each consistency pair.
    unique_groups: dict[tuple[int, ...], np.ndarray] = {}
    for group in groups:
        key = tuple(sorted(map(int, group)))
        unique_groups.setdefault(key, group)
    pair_array = np.asarray(sorted(pairs), dtype=np.int64).reshape(-1, 2)
    return list(unique_groups.values()), pair_array


def _coalesced_metadata(frame: pd.DataFrame, columns: tuple[str, ...]) -> pd.Series:
    result = pd.Series("", index=frame.index, dtype=object)
    for column in columns:
        if column not in frame:
            continue
        values = frame[column].fillna("").astype(str).str.strip()
        selected = result.eq("") & values.ne("")
        result.loc[selected] = values.loc[selected]
    return result


def _generator_signature(frame: pd.DataFrame) -> pd.Series:
    """Return a stable voice/music generator signature without using labels.

    Some sequential-call corpora expose two speech generators rather than a
    single VOICE_GENERATOR. Preserve both names, because collapsing them to
    the first generator would merge genuinely different synthesis domains.
    """
    voice = pd.Series("", index=frame.index, dtype=object)
    for column in VOICE_GENERATOR_COLUMNS:
        if column not in frame:
            continue
        values = frame[column].fillna("").astype(str).str.strip()
        for index in frame.index[values.ne("")]:
            current = [value for value in str(voice.loc[index]).split("+") if value]
            value = values.loc[index]
            if value not in current:
                current.append(value)
            voice.loc[index] = "+".join(sorted(current))
    music = _coalesced_metadata(frame, MUSIC_GENERATOR_COLUMNS)
    return "voice=" + voice.replace("", "unknown") + "|music=" + music.replace(
        "", "unknown"
    )


def _identity_values(
    frame: pd.DataFrame, columns: tuple[str, ...], *, selected: pd.Series,
) -> set[str]:
    """Collect nonempty identity values from every declared component column."""
    values: set[str] = set()
    for column in columns:
        if column not in frame:
            continue
        current = frame.loc[selected, column].dropna().astype(str).str.strip()
        values.update(current.loc[current.ne("")])
    return values


def assert_component_split_contract(
    train: pd.DataFrame, development: pd.DataFrame,
) -> dict[str, object]:
    """Reject train/dev source or parent reuse for either mixed component.

    Generator *families* are deliberately reported rather than rejected: the
    configured development role contains both seen-generator robustness data
    and unseen-generator data.  A family name such as ``suno`` is a domain,
    not an audio identity.  Source/group/speaker/archive/parent values are the
    atomic identities and must be disjoint.
    """
    reports: dict[str, object] = {}
    for label, frame in (("train", train), ("development", development)):
        mixed = (
            pd.to_numeric(frame["VOICE_PRESENT"], errors="coerce").eq(1)
            & pd.to_numeric(frame["MUSIC_PRESENT"], errors="coerce").eq(1)
        )
        voice_present = pd.to_numeric(
            frame["VOICE_PRESENT"], errors="coerce"
        ).eq(1)
        music_present = pd.to_numeric(
            frame["MUSIC_PRESENT"], errors="coerce"
        ).eq(1)
        missing_voice = mixed & pd.Series(
            [
                not any(
                    column in frame
                    and pd.notna(frame.at[index, column])
                    and str(frame.at[index, column]).strip()
                    for column in VOICE_IDENTITY_COLUMNS
                )
                for index in frame.index
            ],
            index=frame.index,
        )
        missing_music = mixed & pd.Series(
            [
                not any(
                    column in frame
                    and pd.notna(frame.at[index, column])
                    and str(frame.at[index, column]).strip()
                    for column in MUSIC_IDENTITY_COLUMNS
                )
                for index in frame.index
            ],
            index=frame.index,
        )
        if missing_voice.any() or missing_music.any():
            raise ValueError(
                f"{label} mixed rows lack component identity: "
                f"voice={frame.loc[missing_voice, 'ID'].head(5).tolist()}, "
                f"music={frame.loc[missing_music, 'ID'].head(5).tolist()}"
            )
        reports[label] = {
            "rows": int(len(frame)),
            "mixed_rows": int(mixed.sum()),
            "voice_identities": len(_identity_values(
                frame, VOICE_IDENTITY_COLUMNS, selected=voice_present,
            )),
            "music_identities": len(_identity_values(
                frame, MUSIC_IDENTITY_COLUMNS, selected=music_present,
            )),
            "parent_identities": len(_identity_values(
                frame, PARENT_IDENTITY_COLUMNS,
                selected=pd.Series(True, index=frame.index),
            )),
        }

    train_voice = _identity_values(
        train, VOICE_IDENTITY_COLUMNS,
        selected=pd.to_numeric(train["VOICE_PRESENT"], errors="coerce").eq(1),
    )
    development_voice = _identity_values(
        development, VOICE_IDENTITY_COLUMNS,
        selected=pd.to_numeric(
            development["VOICE_PRESENT"], errors="coerce"
        ).eq(1),
    )
    train_music = _identity_values(
        train, MUSIC_IDENTITY_COLUMNS,
        selected=pd.to_numeric(train["MUSIC_PRESENT"], errors="coerce").eq(1),
    )
    development_music = _identity_values(
        development, MUSIC_IDENTITY_COLUMNS,
        selected=pd.to_numeric(
            development["MUSIC_PRESENT"], errors="coerce"
        ).eq(1),
    )
    all_train = pd.Series(True, index=train.index)
    all_development = pd.Series(True, index=development.index)
    train_parent = _identity_values(
        train, PARENT_IDENTITY_COLUMNS, selected=all_train,
    )
    development_parent = _identity_values(
        development, PARENT_IDENTITY_COLUMNS, selected=all_development,
    )
    overlaps = {
        "voice": sorted(train_voice & development_voice),
        "music": sorted(train_music & development_music),
        "parent": sorted(train_parent & development_parent),
    }
    if any(overlaps.values()):
        summary = {
            key: {"count": len(values), "examples": values[:5]}
            for key, values in overlaps.items()
        }
        raise ValueError(f"TRAIN/DEVELOPMENT COMPONENT IDENTITY LEAKAGE: {summary}")

    train_generator = _generator_signature(train)
    development_generator = _generator_signature(development)
    reports["overlap"] = {"voice": 0, "music": 0, "parent": 0}
    reports["generator_signatures"] = {
        "train": int(train_generator.nunique()),
        "development": int(development_generator.nunique()),
        "shared": int(
            len(set(train_generator.astype(str)) & set(
                development_generator.astype(str)
            ))
        ),
        "policy": "domain_overlap_reported_not_identity_leakage",
    }
    return reports


def load_train_identity_exclusions(path: Path | None) -> tuple[str, ...]:
    if path is None:
        return ()
    source = path.resolve(strict=True).read_text("utf-8").splitlines()
    values = tuple(
        line.strip() for line in source
        if line.strip() and not line.lstrip().startswith("#")
    )
    if not values or len(values) != len(set(values)):
        raise ValueError("train identity exclusions must be nonempty and unique")
    return values


def filter_train_pair(
    pair: tuple[pd.DataFrame, dict[str, np.ndarray], dict[str, object]],
    exclusions: tuple[str, ...],
) -> tuple[
    tuple[pd.DataFrame, dict[str, np.ndarray], dict[str, object]], list[str]
]:
    """Drop only rows carrying a predeclared contaminated source identity."""
    frame, block, metadata = pair
    if not exclusions:
        return pair, []
    rejected = pd.Series(False, index=frame.index)
    excluded = set(exclusions)
    for column in STRICT_IDENTITY_COLUMNS:
        if column not in frame:
            continue
        values = frame[column].fillna("").astype(str).str.strip()
        rejected |= values.isin(excluded)
    ids = frame.loc[rejected, "ID"].astype(str).tolist()
    keep = (~rejected).to_numpy()
    if not keep.any():
        raise ValueError(f"identity exclusions removed all rows from {frame.DATASET.iloc[0]}")
    selected_block = {key: value[keep] for key, value in block.items()}
    return (
        frame.loc[~rejected].reset_index(drop=True), selected_block, metadata,
    ), ids


def assert_strict_identity_split(
    train: pd.DataFrame, development: pd.DataFrame,
) -> dict[str, object]:
    def tokens(frame: pd.DataFrame) -> set[str]:
        result: set[str] = set()
        for column in STRICT_IDENTITY_COLUMNS:
            if column not in frame:
                continue
            values = frame[column].dropna().astype(str).str.strip()
            result.update(values.loc[values.ne("")])
        return result

    train_tokens, development_tokens = tokens(train), tokens(development)
    overlap = sorted(train_tokens & development_tokens)
    if overlap:
        raise ValueError(
            f"TRAIN/DEVELOPMENT STRICT IDENTITY LEAKAGE: {overlap[:10]}"
        )
    return {
        "train_identity_tokens": len(train_tokens),
        "development_identity_tokens": len(development_tokens),
        "overlap": 0,
        "columns": list(STRICT_IDENTITY_COLUMNS),
    }


def environment_codes(
    frame: pd.DataFrame, generator_aware: bool = False,
) -> np.ndarray:
    """Encode robustness environments; unavailable base metadata is -1."""
    result = np.full(len(frame), -1, dtype=np.int64)
    if "DATASET" not in frame:
        return result
    normalized = pd.DataFrame({
        "DATASET": frame["DATASET"].fillna("").astype(str).str.strip(),
        "LAYOUT": _coalesced_metadata(frame, LAYOUT_COLUMNS),
        "CHANNEL": _coalesced_metadata(frame, CHANNEL_COLUMNS),
    })
    if generator_aware:
        normalized["GENERATOR"] = _generator_signature(frame)
    valid = normalized.ne("").all(axis=1)
    keys = [tuple(row) for row in normalized.loc[valid].to_numpy()]
    mapping = {key: index for index, key in enumerate(sorted(set(keys)))}
    result[np.flatnonzero(valid.to_numpy())] = [mapping[key] for key in keys]
    return result


def _atomic_units(size: int, groups: list[np.ndarray]) -> list[np.ndarray]:
    parent = np.arange(size, dtype=np.int64)

    def find(item: int) -> int:
        while parent[item] != item:
            parent[item] = parent[parent[item]]
            item = int(parent[item])
        return item

    def union(first: int, second: int) -> None:
        left, right = find(first), find(second)
        if left != right:
            parent[right] = left

    for group in groups:
        values = np.asarray(group, dtype=np.int64)
        if values.ndim != 1 or not len(values):
            raise ValueError("sampling groups must be nonempty vectors")
        if values.min() < 0 or values.max() >= size or len(set(values)) != len(values):
            raise ValueError("sampling group indices are invalid")
        for item in values[1:]:
            union(int(values[0]), int(item))
    members: dict[int, list[int]] = {}
    for index in range(size):
        members.setdefault(find(index), []).append(index)
    return [np.asarray(items, dtype=np.int64) for items in members.values()]


def _sampling_metadata(
    frame: pd.DataFrame, generator_aware: bool = False,
) -> tuple[pd.Series, ...]:
    columns = (
        frame["DATASET"].fillna("").astype(str).str.strip()
        if "DATASET" in frame else pd.Series("", index=frame.index),
        _coalesced_metadata(frame, LAYOUT_COLUMNS),
        _coalesced_metadata(frame, CHANNEL_COLUMNS),
    )
    return columns + ((_generator_signature(frame),) if generator_aware else ())


def _unit_stratum(
    columns: tuple[pd.Series, ...], unit: np.ndarray,
    generator_aware: bool = False,
) -> tuple[str, ...]:
    values: list[str] = []
    for column_index, column in enumerate(columns):
        current = sorted(set(column.loc[unit]).difference({""}))
        if len(current) == 1:
            values.append(current[0])
        elif current and generator_aware and column_index == len(columns) - 1:
            values.append("paired:" + "+".join(current))
        else:
            values.append("paired" if current else "missing")
    return tuple(values)


def group_balanced_batches(
    frame: pd.DataFrame,
    batch_size: int,
    generator: np.random.Generator,
    quartets: np.ndarray,
    channel_groups: list[np.ndarray],
    epoch_size: int | None = None,
    generator_aware: bool = False,
) -> list[np.ndarray]:
    """Sample robustness strata while never splitting atomic groups."""
    if batch_size <= 0:
        raise ValueError("batch size must be positive")
    if epoch_size is None:
        epoch_size = len(frame)
    if epoch_size <= 0 or not len(frame):
        raise ValueError("epoch size and training frame must be nonempty")
    groups = [row for row in quartets] + list(channel_groups)
    units = _atomic_units(len(frame), groups)
    largest = max(map(len, units))
    if largest > batch_size:
        raise ValueError(
            f"batch size {batch_size} cannot hold atomic group of {largest}"
        )
    strata: dict[tuple[str, ...], list[np.ndarray]] = {}
    metadata = _sampling_metadata(frame, generator_aware=generator_aware)
    for unit in units:
        strata.setdefault(
            _unit_stratum(metadata, unit, generator_aware=generator_aware), []
        ).append(unit)
    keys = sorted(strata)
    batches: list[np.ndarray] = []
    current: list[int] = []
    emitted = 0
    while emitted < epoch_size:
        for key_index in generator.permutation(len(keys)):
            choices = strata[keys[int(key_index)]]
            unit = choices[int(generator.integers(len(choices)))]
            if current and len(current) + len(unit) > batch_size:
                batches.append(np.asarray(current, dtype=np.int64))
                current = []
            current.extend(unit.tolist())
            emitted += len(unit)
            if emitted >= epoch_size:
                break
    if current:
        batches.append(np.asarray(current, dtype=np.int64))
    return batches


def remap_complete_groups(batch: np.ndarray, groups: np.ndarray) -> np.ndarray:
    """Map repeated global atomic groups to their corresponding batch rows."""
    width = groups.shape[1] if groups.ndim == 2 else 0
    if groups.ndim != 2:
        raise ValueError("groups must be a matrix")
    positions: dict[int, list[int]] = {}
    for local, global_index in enumerate(np.asarray(batch, dtype=np.int64)):
        positions.setdefault(int(global_index), []).append(local)
    result: list[list[int]] = []
    for group in groups:
        counts = [len(positions.get(int(index), [])) for index in group]
        if any(counts) and len(set(counts)) != 1:
            raise RuntimeError("an atomic robustness group was split across batches")
        for occurrence in range(counts[0] if counts else 0):
            result.append([
                positions[int(index)][occurrence] for index in group
            ])
    return np.asarray(result, dtype=np.int64).reshape(-1, width)


def move_to_device(
    block: dict[str, np.ndarray], device: torch.device,
) -> dict[str, torch.Tensor]:
    return {key: torch.from_numpy(value).to(device) for key, value in block.items()}


def tensor_batch(
    block: dict[str, torch.Tensor], indices: np.ndarray, device: torch.device,
) -> tuple[torch.Tensor, ...]:
    selected = torch.as_tensor(indices, dtype=torch.long, device=device)
    keys = BASE_KEYS + (XLSR_KEYS if "xlsr_embeddings" in block else ())
    return tuple(block[key].index_select(0, selected) for key in keys)


@torch.inference_mode()
def predict(
    model: ThreeStreamAnchorResidualHead,
    block: dict[str, torch.Tensor],
    device: torch.device,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    authenticity, presence, residuals = [], [], []
    size = len(block["temporal"])
    for offset in range(0, size, batch_size):
        indices = np.arange(offset, min(offset + batch_size, size))
        fake_logits, presence_logits, residual = model(
            *tensor_batch(block, indices, device)
        )
        authenticity.append(fake_logits.sigmoid().cpu().numpy())
        presence.append(presence_logits.sigmoid().cpu().numpy())
        residuals.append(residual.cpu().numpy())
    return (
        np.concatenate(authenticity),
        np.concatenate(presence),
        np.concatenate(residuals),
    )


def normalized_available_ads(metric: dict[str, float | int]) -> float:
    """Return ADS with official weights renormalized over defined EER tasks.

    Speech-only and music-only development corpora do not have a competition
    ADS in isolation.  Ignoring them would remove useful generator-OOD
    evidence from checkpoint selection, so domain robustness uses this
    diagnostic score while the pooled development score remains the exact
    competition formula.
    """
    weighted_scores: list[float] = []
    active_weights: list[float] = []
    for key, weight in (
        ("FILE_EER", 0.5), ("VOICE_EER", 0.2), ("MUSIC_EER", 0.3),
    ):
        value = float(metric[key])
        if np.isfinite(value):
            weighted_scores.append(weight * (1.0 - value))
            active_weights.append(weight)
    if not active_weights:
        return float("nan")
    return float(sum(weighted_scores) / sum(active_weights))


def evaluate(
    model: ThreeStreamAnchorResidualHead,
    frames: tuple[pd.DataFrame, ...],
    blocks: tuple[dict[str, torch.Tensor], ...],
    device: torch.device,
    batch_size: int,
    selection_task: str = "all",
) -> tuple[pd.DataFrame, pd.DataFrame, float]:
    metrics, predictions, scored_frames, domain_ads = [], [], [], []
    for frame, block in zip(frames, blocks):
        fake, presence, residual = predict(model, block, device, batch_size)
        prediction = pd.DataFrame({
            "FILE_FAKE_PROB": fake[:, 2],
            "VOICE_FAKE_PROB": fake[:, 0],
            "MUSIC_FAKE_PROB": fake[:, 1],
            "VOICE_PRESENT_PROB": presence[:, 0],
            "MUSIC_PRESENT_PROB": presence[:, 1],
            "VOICE_LOGIT_RESIDUAL": residual[:, 0],
            "MUSIC_LOGIT_RESIDUAL": residual[:, 1],
            "DIRECT_FILE_LOGIT_RESIDUAL": residual[:, 2],
        }, index=frame["ID"])
        score_columns = prediction[list(
            ("FILE_FAKE_PROB", "VOICE_FAKE_PROB", "MUSIC_FAKE_PROB")
            + PRESENCE_COLUMNS
        )]
        scored = frame.set_index("ID").join(score_columns)
        metric = score_frame(scored)
        normalized_ads = normalized_available_ads(metric)
        metrics.append({
            "DATASET": frame["DATASET"].iloc[0],
            **metric,
            "NORMALIZED_AVAILABLE_ADS": normalized_ads,
            "MEAN_ABS_VOICE_RESIDUAL": float(np.abs(residual[:, 0]).mean()),
            "MEAN_ABS_MUSIC_RESIDUAL": float(np.abs(residual[:, 1]).mean()),
            "MEAN_ABS_DIRECT_FILE_RESIDUAL": float(np.abs(residual[:, 2]).mean()),
            "MAX_ABS_RESIDUAL": float(np.abs(residual).max()),
        })
        current = prediction.reset_index().rename(columns={"index": "ID"})
        current.insert(0, "DATASET", frame["DATASET"].to_numpy())
        predictions.append(current)
        scored_frames.append(scored.reset_index(drop=True))
        domain_ads.append(normalized_ads)
    finite_domains = np.asarray(domain_ads, dtype=np.float64)
    finite_domains = finite_domains[np.isfinite(finite_domains)]
    if not len(finite_domains):
        raise ValueError("no development partition has a defined authenticity task")
    pooled_ads = float(score_frame(pd.concat(scored_frames, ignore_index=True))["ADS"])
    if not np.isfinite(pooled_ads):
        raise ValueError("pooled development data does not define official ADS")
    # Half of selection remains sample-pooled and competition-exact.  The
    # other half prevents a large/easy corpus from hiding a weak OOD domain.
    selection = (
        0.50 * pooled_ads
        + 0.25 * finite_domains.mean()
        + 0.25 * finite_domains.min()
    )
    metric_frame = pd.DataFrame(metrics)
    if selection_task != "all":
        pooled = score_frame(pd.concat(scored_frames, ignore_index=True))
        selection = specialist_selection(metric_frame, pooled, selection_task)
    metric_frame["POOLED_DEVELOPMENT_ADS"] = pooled_ads
    metric_frame["SELECTION_SCORE"] = float(selection)
    return metric_frame, pd.concat(predictions), float(selection)


def specialist_selection(
    domain_metrics: pd.DataFrame, pooled_metrics: dict, task: str,
) -> float:
    """Select the deployed component, excluding undefined single-class EERs.

    The same 50/25/25 policy as joint training is used, but unrelated File or
    Voice improvements cannot select a worse Music checkpoint.
    """
    if task not in ("voice", "music", "file"):
        raise ValueError("specialist task must be voice, music, or file")
    column = f"{task.upper()}_EER"
    eers = domain_metrics[column].to_numpy(dtype=np.float64)
    eers = eers[np.isfinite(eers)]
    pooled = float(pooled_metrics[column])
    if not len(eers) or not np.isfinite(pooled):
        raise ValueError(f"no defined {task} EER for specialist selection")
    return float(1 - (0.50 * pooled + 0.25 * eers.mean() + 0.25 * eers.max()))


def model_config(
    block: dict[str, np.ndarray], args: argparse.Namespace,
) -> dict[str, object]:
    temporal, spectral, spear = (
        block["temporal"], block["spectral"], block["spear"]
    )
    return {
        "eat_layers": temporal.shape[2],
        "eat_dimension": temporal.shape[-1],
        "spear_layers": spear.shape[3],
        "spear_stats": spear.shape[4],
        "spear_dimension": spear.shape[-1],
        "xlsr_dimension": (
            XLSR_WINDOW_DIMENSION if "xlsr_embeddings" in block else None
        ),
        "width": args.width,
        "heads": args.heads,
        "depth": args.depth,
        "maximum_views": temporal.shape[1],
        "maximum_time_nodes": temporal.shape[-2],
        "maximum_frequency_nodes": spectral.shape[-2],
        "maximum_bins": spear.shape[2],
        "maximum_xlsr_windows": (
            max(args.maximum_xlsr_windows, block["xlsr_embeddings"].shape[1])
            if "xlsr_embeddings" in block else 1
        ),
        "dropout": args.dropout,
        "residual_limits": tuple(args.residual_limits),
        "task_interaction_depth": args.task_interaction_depth,
    }


def load_initial_checkpoint(
    model: ThreeStreamAnchorResidualHead,
    config: dict[str, object],
    path: Path,
    train_partitions: list[Partition],
    dev_partitions: list[Partition],
) -> dict:
    """Warm-start a zero-adapter architecture from an authorized old head."""
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if checkpoint.get("model_type") != "three_stream_anchor_residual_component_query_v1":
        raise ValueError("initial checkpoint is not a three-stream residual head")
    initial_config = checkpoint.get("config")
    if not isinstance(initial_config, dict):
        raise ValueError("initial checkpoint has no model config")
    for key, value in initial_config.items():
        if key == "task_interaction_depth":
            continue
        if key not in config or config[key] != value:
            raise ValueError(f"initial checkpoint config differs for {key}")
    authorized_train = {item.truth_path.resolve() for item in train_partitions}
    authorized_dev = {item.truth_path.resolve() for item in dev_partitions}
    initial_train = {Path(item).resolve() for item in checkpoint.get("train_truths", ())}
    initial_dev = {
        Path(item).resolve() for item in checkpoint.get("development_truths", ())
    }
    if not initial_train or not initial_train.issubset(authorized_train):
        raise ValueError("initial checkpoint training truths are not authorized")
    if not initial_dev or not initial_dev.issubset(authorized_dev):
        raise ValueError("initial checkpoint development truths are not authorized")
    incompatible = model.load_state_dict(checkpoint["model"], strict=False)
    allowed_missing = {
        name for name in model.state_dict() if name.startswith("task_interaction_")
    }
    if set(incompatible.missing_keys) != allowed_missing or incompatible.unexpected_keys:
        raise ValueError(
            "initial checkpoint parameters differ outside the task interaction adapter"
        )
    return checkpoint


def configure_trainable_parameters(
    model: ThreeStreamAnchorResidualHead, task_interaction_only: bool,
) -> list[torch.nn.Parameter]:
    if task_interaction_only:
        for name, parameter in model.named_parameters():
            parameter.requires_grad_(name.startswith("task_interaction_"))
    selected = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not selected:
        raise RuntimeError("training configuration has no trainable parameters")
    return selected


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--partition-config", type=Path,
        default=ROOT / "configs/data_partitions.yaml",
    )
    parser.add_argument(
        "--eat-cache-root", type=Path,
        default=ROOT / "output/eat_patch_graph_v1",
    )
    parser.add_argument(
        "--spear-cache-root", type=Path,
        default=ROOT / "reports/spear_temporal_attention_v1/cache",
    )
    parser.add_argument("--xlsr-cache-root", type=Path)
    parser.add_argument("--anchor-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--initial-checkpoint", type=Path)
    parser.add_argument("--initial-consistency-weight", type=float, default=0.0)
    parser.add_argument("--train-identity-exclusions", type=Path)
    parser.add_argument(
        "--selected-split-only", action="store_true",
        help=(
            "Do not read non-selected protected manifests; enforce strict "
            "train/development identities after loading the selected rows."
        ),
    )
    parser.add_argument("--train-datasets", nargs="+")
    parser.add_argument("--dev-datasets", nargs="+")
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--width", type=int, default=96)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--task-interaction-depth", type=int, default=0)
    parser.add_argument(
        "--maximum-xlsr-windows", type=int, default=16,
        help="Reserve positions for up to 60 s competition audio at inference.",
    )
    parser.add_argument("--dropout", type=float, default=0.15)
    parser.add_argument(
        "--residual-limits", type=float, nargs=3, default=(1.0, 1.0, 1.0),
    )
    parser.add_argument("--task-weights", type=float, nargs=3,
                        default=(0.2, 0.3, 0.5))
    parser.add_argument(
        "--selection-task", choices=("all", "voice", "music", "file"),
        default="all", help="Select checkpoints for the component actually deployed.",
    )
    parser.add_argument("--residual-weight", type=float, default=0.02)
    parser.add_argument("--ranking-weight", type=float, default=0.10)
    parser.add_argument("--ranking-margin", type=float, default=0.0)
    parser.add_argument("--quartet-invariance-weight", type=float, default=0.0)
    parser.add_argument(
        "--quartet-latent-invariance-weight", type=float, default=0.0,
    )
    parser.add_argument("--auc-ranking-weight", type=float, default=0.0)
    parser.add_argument("--auc-ranking-margin", type=float, default=0.0)
    parser.add_argument("--logit-consistency-weight", type=float, default=0.05)
    parser.add_argument("--latent-consistency-weight", type=float, default=0.01)
    parser.add_argument("--group-dro-weight", type=float, default=0.20)
    parser.add_argument("--group-dro-temperature", type=float, default=0.10)
    parser.add_argument(
        "--generator-aware", action="store_true",
        help=(
            "Include source-generator identity in balanced sampling and "
            "dataset/layout/channel Group-DRO environments."
        ),
    )
    parser.add_argument(
        "--train-task-interaction-only", action="store_true",
        help="Freeze the warm-started base and optimize only task-interaction parameters.",
    )
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=48)
    parser.add_argument("--eval-batch-size", type=int, default=96)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=2e-2)
    parser.add_argument("--eval-every", type=int, default=2)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260904)
    args = parser.parse_args()
    if not np.isfinite(args.initial_consistency_weight) or args.initial_consistency_weight < 0:
        parser.error("initial consistency weight must be finite and nonnegative")
    if args.initial_consistency_weight > 0 and args.initial_checkpoint is None:
        parser.error("initial consistency requires an authorized warm-start checkpoint")
    if not args.validate_only and args.output_dir is None:
        parser.error("--output-dir is required unless --validate-only is used")
    if min(args.width, args.heads, args.depth, args.maximum_xlsr_windows,
           args.epochs, args.batch_size,
           args.eval_batch_size, args.eval_every, args.patience) <= 0:
        parser.error("model and training counts must be positive")
    if args.task_interaction_depth < 0:
        parser.error("--task-interaction-depth cannot be negative")
    if args.train_task_interaction_only and (
        args.initial_checkpoint is None or args.task_interaction_depth <= 0
    ):
        parser.error(
            "--train-task-interaction-only requires a warm start and positive "
            "task interaction depth"
        )
    if args.width % args.heads:
        parser.error("--width must be divisible by --heads")
    if not 0 <= args.dropout < 1:
        parser.error("--dropout must lie in [0,1)")
    if len(args.task_weights) != 3 or any(weight < 0 for weight in args.task_weights):
        parser.error("--task-weights must be three nonnegative values")
    if sum(args.task_weights) <= 0:
        parser.error("--task-weights must have positive total mass")
    robustness_values = (
        args.residual_weight, args.ranking_weight, args.ranking_margin,
        args.quartet_invariance_weight,
        args.quartet_latent_invariance_weight,
        args.auc_ranking_weight, args.auc_ranking_margin,
        args.logit_consistency_weight, args.latent_consistency_weight,
        args.group_dro_weight, args.group_dro_temperature,
    )
    if any(value < 0 for value in robustness_values):
        parser.error("robustness weights, margin, and temperature cannot be negative")

    train_partitions = authorized_partitions(
        args.partition_config, "train", args.train_datasets
    )
    dev_partitions = authorized_partitions(
        args.partition_config, "development", args.dev_datasets
    )
    if not args.selected_split_only:
        guard_partitions(train_partitions, dev_partitions, args.partition_config)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    selected_names = list(dict.fromkeys(
        item.name for item in train_partitions + dev_partitions
    ))
    spear_cache = load_spear_caches(args.spear_cache_root, selected_names)
    pairs = [
        load_block(
            partition, args.eat_cache_root, spear_cache,
            args.anchor_root, args.xlsr_cache_root,
        )
        for partition in train_partitions + dev_partitions
    ]
    exclusions = load_train_identity_exclusions(args.train_identity_exclusions)
    excluded_train_ids: list[str] = []
    for index in range(len(train_partitions)):
        pairs[index], removed = filter_train_pair(pairs[index], exclusions)
        excluded_train_ids.extend(removed)
    pad_xlsr_window_axis([item[1] for item in pairs])
    train_count = len(train_partitions)
    train_pairs, dev_pairs = pairs[:train_count], pairs[train_count:]
    validate_metadata([item[2] for item in pairs])
    train_frames = [item[0] for item in train_pairs]
    dev_frames = tuple(item[0] for item in dev_pairs)
    train_numpy = concatenate([item[1] for item in train_pairs])
    dev_numpy = tuple(item[1] for item in dev_pairs)
    validate_numpy_block(train_numpy)
    for block in dev_numpy:
        validate_numpy_block(block)
    train_frame = pd.concat(train_frames, ignore_index=True)
    development_frame = pd.concat(dev_frames, ignore_index=True)
    component_split_contract = assert_component_split_contract(
        train_frame, development_frame,
    )
    strict_identity_contract = assert_strict_identity_split(
        train_frame, development_frame,
    )
    quartets = complete_pair_group_quartets(train_frame)
    channel_groups, channel_pairs = channel_consistency_groups(train_frame)
    environments = environment_codes(
        train_frame, generator_aware=args.generator_aware,
    )
    # Validate that configured batches can preserve every available atomic
    # group even in --validate-only mode.
    group_balanced_batches(
        train_frame, args.batch_size, np.random.default_rng(args.seed),
        quartets, channel_groups, epoch_size=min(len(train_frame), args.batch_size),
        generator_aware=args.generator_aware,
    )
    config = model_config(train_numpy, args)
    model = ThreeStreamAnchorResidualHead(**config)
    initial_checkpoint = None
    if args.initial_checkpoint is not None:
        initial_checkpoint = load_initial_checkpoint(
            model, config, args.initial_checkpoint.resolve(strict=True),
            train_partitions, dev_partitions,
        )
        required_exclusions = set(exclusions)
        initial_exclusions = set(initial_checkpoint.get("train_identity_exclusions", ()))
        if not required_exclusions.issubset(initial_exclusions):
            raise ValueError("warm-start checkpoint does not certify required train identity exclusions")
    trainable_parameters = configure_trainable_parameters(
        model, args.train_task_interaction_only,
    )

    # A small CPU pass validates all cross-stream tensor contracts without
    # launching training or loading another backbone.
    validation_size = min(2, len(train_numpy["temporal"]))
    validation_indices = np.arange(validation_size)
    validation_block = move_to_device(train_numpy, torch.device("cpu"))
    model.eval()
    with torch.inference_mode():
        output, presence, residual = model(
            *tensor_batch(validation_block, validation_indices, torch.device("cpu"))
        )
    anchor = validation_block["anchor_authenticity"][:validation_size]
    if initial_checkpoint is None and (
        not torch.equal(output, anchor) or not torch.count_nonzero(residual).eq(0)
    ):
        raise RuntimeError("zero-initialized residual head is not the anchor identity")
    if not torch.isfinite(output).all() or not torch.isfinite(residual).all():
        raise RuntimeError("scaffold validation produced non-finite authenticity")
    if not torch.equal(
        presence, validation_block["anchor_presence"][:validation_size]
    ):
        raise RuntimeError("presence changed during scaffold validation")
    print(json.dumps({
        "validated": True,
        "train_datasets": [item.name for item in train_partitions],
        "development_datasets": [item.name for item in dev_partitions],
        "train_examples": int(len(train_numpy["temporal"])),
        "xlsr_enabled": args.xlsr_cache_root is not None,
        "pair_group_quartets": int(len(quartets)),
        "channel_groups": int(len(channel_groups)),
        "channel_pairs": int(len(channel_pairs)),
        "group_dro_environments": int(len(set(environments[environments >= 0]))),
        "generator_aware": args.generator_aware,
        "component_split_contract": component_split_contract,
        "strict_identity_contract": strict_identity_contract,
        "excluded_train_ids": excluded_train_ids,
        "selected_split_only": args.selected_split_only,
        "warm_started": initial_checkpoint is not None,
    }), flush=True)
    if args.validate_only:
        return

    assert args.output_dir is not None
    device = torch.device(args.device)
    model = model.to(device)
    train = move_to_device(train_numpy, device)
    development = tuple(move_to_device(item, device) for item in dev_numpy)
    fake_numpy, presence_numpy = targets(train_frame)
    weight_numpy = sample_weights(train_frame)
    fake_targets = torch.from_numpy(fake_numpy).to(device)
    presence_targets = torch.from_numpy(presence_numpy).to(device)
    sample_weight = torch.from_numpy(weight_numpy).to(device)
    initial_logits = None
    if args.initial_consistency_weight > 0:
        # Cache targets once from approved train features ONLY. No teacher is
        # added to inference and no development labels/predictions are trained on.
        model.eval()
        chunks = []
        with torch.no_grad():
            for offset in range(0, len(train_frame), args.eval_batch_size):
                indices = np.arange(offset, min(offset + args.eval_batch_size, len(train_frame)))
                chunks.append(model(*tensor_batch(train, indices, device))[0].detach())
        initial_logits = torch.cat(chunks)
    optimizer = torch.optim.AdamW(
        trainable_parameters, lr=args.learning_rate, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.learning_rate * 0.05
    )
    generator = np.random.default_rng(args.seed)

    metrics, predictions, best_selection = evaluate(
        model, dev_frames, development, device, args.eval_batch_size,
        selection_task=args.selection_task,
    )
    best_epoch = 0
    best_state = copy.deepcopy(model.state_dict())
    history = [{
        "EPOCH": 0,
        "LOSS": np.nan,
        "SELECTION": best_selection,
        "POOLED_DEVELOPMENT_ADS": float(metrics["POOLED_DEVELOPMENT_ADS"].iloc[0]),
        "MEAN_NORMALIZED_DOMAIN_ADS": float(metrics["NORMALIZED_AVAILABLE_ADS"].mean()),
        "WORST_NORMALIZED_DOMAIN_ADS": float(metrics["NORMALIZED_AVAILABLE_ADS"].min()),
    }]
    stale = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        batches = group_balanced_batches(
            train_frame, args.batch_size, generator,
            quartets, channel_groups, epoch_size=len(train_frame),
            generator_aware=args.generator_aware,
        )
        losses = []
        epoch_terms: dict[str, list[float]] = {}
        for indices in batches:
            auth_logits, _, residuals, latent = model.forward_with_latent(
                *tensor_batch(train, indices, device)
            )
            selected = torch.as_tensor(indices, dtype=torch.long, device=device)
            local_quartets = torch.from_numpy(
                remap_complete_groups(indices, quartets)
            ).to(device)
            local_channel_pairs = torch.from_numpy(
                remap_complete_groups(indices, channel_pairs)
            ).to(device)
            local_environments = torch.from_numpy(environments[indices]).to(device)
            loss, terms = robust_anchor_residual_loss(
                auth_logits,
                fake_targets.index_select(0, selected),
                presence_targets.index_select(0, selected),
                sample_weight.index_select(0, selected),
                residuals,
                latent=latent,
                quartets=local_quartets,
                channel_pairs=local_channel_pairs,
                environment_ids=local_environments,
                task_weights=args.task_weights,
                residual_weight=args.residual_weight,
                ranking_weight=args.ranking_weight,
                ranking_margin=args.ranking_margin,
                quartet_invariance_weight=args.quartet_invariance_weight,
                quartet_latent_invariance_weight=(
                    args.quartet_latent_invariance_weight
                ),
                auc_ranking_weight=args.auc_ranking_weight,
                auc_ranking_margin=args.auc_ranking_margin,
                logit_consistency_weight=args.logit_consistency_weight,
                latent_consistency_weight=args.latent_consistency_weight,
                group_dro_weight=args.group_dro_weight,
                group_dro_temperature=args.group_dro_temperature,
            )
            if initial_logits is not None:
                retention = initial_model_consistency_loss(
                    auth_logits, initial_logits.index_select(0, selected),
                    presence_targets.index_select(0, selected), args.task_weights,
                )
                loss = loss + args.initial_consistency_weight * retention
                terms["initial_consistency"] = retention.detach()
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite training loss at epoch {epoch}")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0, error_if_nonfinite=True)
            optimizer.step()
            losses.append(float(loss.detach()))
            for name, value in terms.items():
                epoch_terms.setdefault(name, []).append(float(value))
        scheduler.step()
        if epoch % args.eval_every:
            continue
        metrics, predictions, selection = evaluate(
            model, dev_frames, development, device, args.eval_batch_size,
            selection_task=args.selection_task,
        )
        row = {
            "EPOCH": epoch,
            "LOSS": float(np.mean(losses)),
            "SELECTION": selection,
            "POOLED_DEVELOPMENT_ADS": float(metrics["POOLED_DEVELOPMENT_ADS"].iloc[0]),
            "MEAN_NORMALIZED_DOMAIN_ADS": float(metrics["NORMALIZED_AVAILABLE_ADS"].mean()),
            "WORST_NORMALIZED_DOMAIN_ADS": float(metrics["NORMALIZED_AVAILABLE_ADS"].min()),
            **{
                f"LOSS_{name.upper()}": float(np.mean(values))
                for name, values in epoch_terms.items()
            },
        }
        # Detect a saturated, almost constant correction: it can reduce BCE
        # while leaving every within-task rank (and thus EER) unchanged.
        for task, column, limit in zip(
            ("VOICE", "MUSIC", "FILE"),
            ("VOICE_LOGIT_RESIDUAL", "MUSIC_LOGIT_RESIDUAL", "DIRECT_FILE_LOGIT_RESIDUAL"),
            args.residual_limits,
        ):
            values = predictions[column].to_numpy(dtype=np.float64)
            row[f"{task}_RESIDUAL_STD"] = float(values.std())
            row[f"{task}_RESIDUAL_MEAN"] = float(values.mean())
            row[f"{task}_RESIDUAL_SATURATION"] = float(
                (np.abs(values) >= .99 * limit).mean()
            ) if limit > 0 else 0.0
        history.append(row)
        args.output_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(history).to_csv(args.output_dir / "history.partial.csv", index=False)
        print(json.dumps(row), flush=True)
        if selection > best_selection + 1e-5:
            best_selection, best_epoch = selection, epoch
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
            if stale >= args.patience:
                break

    model.load_state_dict(best_state)
    metrics, predictions, best_selection = evaluate(
        model, dev_frames, development, device, args.eval_batch_size,
        selection_task=args.selection_task,
    )
    checkpoint = {
        "model_type": "three_stream_anchor_residual_component_query_v1",
        "model": {key: value.cpu() for key, value in best_state.items()},
        "config": config,
        "train_truths": [str(item.truth_path) for item in train_partitions],
        "development_truths": [str(item.truth_path) for item in dev_partitions],
        "partition_config": str(args.partition_config.resolve()),
        "task_weights": list(args.task_weights),
        "residual_weight": args.residual_weight,
        "ranking_weight": args.ranking_weight,
        "ranking_margin": args.ranking_margin,
        "quartet_invariance_weight": args.quartet_invariance_weight,
        "quartet_latent_invariance_weight": (
            args.quartet_latent_invariance_weight
        ),
        "auc_ranking_weight": args.auc_ranking_weight,
        "auc_ranking_margin": args.auc_ranking_margin,
        "logit_consistency_weight": args.logit_consistency_weight,
        "latent_consistency_weight": args.latent_consistency_weight,
        "group_dro_weight": args.group_dro_weight,
        "group_dro_temperature": args.group_dro_temperature,
        "generator_aware": args.generator_aware,
        "component_split_contract": component_split_contract,
        "strict_identity_contract": strict_identity_contract,
        "train_identity_exclusions": list(exclusions),
        "train_identity_exclusions_sha256": (
            hashlib.sha256(
                args.train_identity_exclusions.resolve(strict=True).read_bytes()
            ).hexdigest()
            if args.train_identity_exclusions is not None else None
        ),
        "excluded_train_ids": excluded_train_ids,
        "selected_split_only": args.selected_split_only,
        "initial_checkpoint": (
            str(args.initial_checkpoint.resolve())
            if args.initial_checkpoint is not None else None
        ),
        "initial_checkpoint_sha256": (
            hashlib.sha256(args.initial_checkpoint.read_bytes()).hexdigest()
            if args.initial_checkpoint is not None else None
        ),
        "initial_consistency_weight": args.initial_consistency_weight,
        "train_task_interaction_only": args.train_task_interaction_only,
        "pair_group_quartets": len(quartets),
        "channel_pairs": len(channel_pairs),
        "group_dro_environments": len(set(environments[environments >= 0])),
        "best_epoch": best_epoch,
        "selection": best_selection,
        "selection_task": args.selection_task,
        "selection_formula": (
            "0.50*pooled_official_ADS + 0.25*mean_normalized_available_"
            "domain_ADS + 0.25*worst_normalized_available_domain_ADS"
            if args.selection_task == "all" else
            f"1-(0.50*pooled_{args.selection_task}_EER+0.25*mean_domain_"
            f"{args.selection_task}_EER+0.25*worst_domain_{args.selection_task}_EER)"
        ),
        "seed": args.seed,
        "feature_metadata": pairs[0][2],
        "spear_projection": spear_cache["__metadata__"]["projection"],
        "spear_layers": spear_cache["__metadata__"]["layers"],
        "spear_bins": spear_cache["__metadata__"]["bins"],
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, args.output_dir / "three_stream_anchor_residual.pt")
    pd.DataFrame(history).to_csv(args.output_dir / "history.csv", index=False)
    metrics.to_csv(args.output_dir / "dev_metrics.csv", index=False)
    predictions.to_csv(args.output_dir / "dev_predictions.csv", index=False)
    summary = {
        "best_epoch": best_epoch,
        "selection": best_selection,
        "selection_formula": checkpoint["selection_formula"],
        "parameters": sum(value.numel() for value in model.parameters()),
        "trainable_parameters": sum(
            value.numel() for value in model.parameters() if value.requires_grad
        ),
        "train_examples": len(train_frame),
        "xlsr_enabled": args.xlsr_cache_root is not None,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
