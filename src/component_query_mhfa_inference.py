"""Offline scoring and conservative fusion for component-query MHFA."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import torch

try:  # package imports in tests; flat imports in offline submission.
    from .component_query_mhfa import ComponentQueryMHFA
except ImportError:  # pragma: no cover
    from component_query_mhfa import ComponentQueryMHFA


def _logit(value: np.ndarray | float) -> np.ndarray:
    value = np.clip(np.asarray(value, dtype=np.float64), 1e-6, 1 - 1e-6)
    return np.log(value) - np.log1p(-value)


def _sigmoid(value: np.ndarray | float) -> np.ndarray:
    return 1 / (1 + np.exp(-np.clip(value, -60, 60)))


def _component_or_file(
    file_probability: float,
    voice_probability: float,
    music_probability: float,
    weight: float,
) -> float:
    component_or = 1 - (1 - voice_probability) * (1 - music_probability)
    return float(_sigmoid(
        (1 - weight) * _logit(file_probability)
        + weight * _logit(component_or)
    ))


def _require_metadata(
    archive: np.lib.npyio.NpzFile,
    checkpoint: dict,
    mapping: tuple[tuple[str, str], ...],
) -> None:
    for archive_key, checkpoint_key in mapping:
        if archive_key not in archive or not np.array_equal(
            archive[archive_key], checkpoint[checkpoint_key]
        ):
            raise ValueError(f"cache {archive_key} differs from checkpoint")


@torch.inference_mode()
def score_component_query_mhfa(
    eat_path: Path,
    spear_path: Path,
    checkpoint_path: Path,
    device: str = "cuda",
    batch_size: int = 64,
) -> tuple[np.ndarray, np.ndarray]:
    if batch_size <= 0:
        raise ValueError("batch size must be positive")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint.get("model_type") != "component_query_mhfa_v1":
        raise ValueError("not a component-query MHFA checkpoint")
    eat = np.load(eat_path, allow_pickle=False)
    spear = np.load(spear_path, allow_pickle=False)
    _require_metadata(eat, checkpoint, (
        ("projection", "eat_projection"), ("layers", "eat_layers"),
    ))
    _require_metadata(spear, checkpoint, (
        ("projection", "spear_projection"), ("layers", "spear_layers"),
        ("bins", "spear_bins"),
    ))
    eat_ids = eat["ids"].astype(str)
    spear_ids = spear["ids"].astype(str)
    if len(set(eat_ids)) != len(eat_ids) or len(set(spear_ids)) != len(spear_ids):
        raise ValueError("component-query cache contains duplicate IDs")
    if set(eat_ids) != set(spear_ids):
        raise ValueError("EAT and SPEAR component-query IDs differ")
    spear_lookup = {item: index for index, item in enumerate(spear_ids)}
    order = np.asarray([spear_lookup[item] for item in eat_ids], dtype=np.int64)
    target = torch.device(device)
    model = ComponentQueryMHFA(**checkpoint["config"]).to(target)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()
    predictions = []
    for offset in range(0, len(eat_ids), batch_size):
        end = offset + batch_size
        spear_order = order[offset:end]
        inputs = (
            torch.from_numpy(eat["temporal"][offset:end]).to(target),
            torch.from_numpy(eat["spectral"][offset:end]).to(target),
            torch.from_numpy(eat["view_mask"][offset:end]).to(target),
            torch.from_numpy(spear["features"][spear_order]).to(target),
            torch.from_numpy(spear["mask"][spear_order]).to(target),
        )
        authenticity, _, _ = model(*inputs)
        predictions.append(model.probabilities(authenticity).float().cpu().numpy())
    return eat_ids, np.concatenate(predictions)


def apply_component_query_mhfa_fusion(
    submission_path: Path,
    eat_path: Path,
    spear_path: Path,
    checkpoint_path: Path,
    device: str = "cuda",
    file_weight: float = .025,
    music_weight: float = .05,
    file_or_weight: float = 0.,
) -> None:
    """Fuse only File/Music; preserve Voice and both CPS outputs byte-exactly."""
    if (
        not 0 <= file_weight <= 1 or not 0 <= music_weight <= 1
        or not 0 <= file_or_weight <= 1
    ):
        raise ValueError("fusion weights must lie in [0,1]")
    ids, expert = score_component_query_mhfa(
        eat_path, spear_path, checkpoint_path, device=device
    )
    by_id = dict(zip(ids, expert))
    with submission_path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        columns = list(reader.fieldnames or [])
        rows = list(reader)
    if {row["ID"] for row in rows} != set(by_id):
        raise ValueError("submission and component-query IDs differ")
    for row in rows:
        probability = by_id[row["ID"]]
        for column, index, weight in (
            ("FILE_FAKE_PROB", 2, file_weight),
            ("MUSIC_FAKE_PROB", 1, music_weight),
        ):
            row[column] = round(float(_sigmoid(
                (1 - weight) * _logit(float(row[column]))
                + weight * _logit(probability[index])
            )), 10)
        if file_or_weight:
            row["FILE_FAKE_PROB"] = round(_component_or_file(
                float(row["FILE_FAKE_PROB"]),
                float(row["VOICE_FAKE_PROB"]),
                float(row["MUSIC_FAKE_PROB"]), file_or_weight,
            ), 10)
    temporary = submission_path.with_suffix(".tmp.csv")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(submission_path)
