"""Offline inference and domain-conditional fusion for hierarchical EAT heads."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import torch

try:  # package import in tests; flat import in the offline submission
    from .hierarchical_eat_music import HierarchicalEatMusicHead
except ImportError:  # pragma: no cover - exercised by script.py
    from hierarchical_eat_music import HierarchicalEatMusicHead


def _logit(values: np.ndarray) -> np.ndarray:
    values = np.clip(np.asarray(values, dtype=np.float64), 1e-5, 1 - 1e-5)
    return np.log(values) - np.log1p(-values)


def _sigmoid(values: np.ndarray) -> np.ndarray:
    return np.exp(-np.logaddexp(0.0, -np.asarray(values, dtype=np.float64)))


def _fuse(anchor: float, expert: float, weight: float) -> float:
    return float(_sigmoid((1 - weight) * _logit(anchor) + weight * _logit(expert)))


def _model(checkpoint, device: torch.device) -> HierarchicalEatMusicHead:
    state = checkpoint["model"]
    model = HierarchicalEatMusicHead(
        state["normalization_mean"], state["normalization_std"],
        **checkpoint["config"],
    )
    model.load_state_dict(state, strict=True)
    return model.to(device).eval()


@torch.inference_mode()
def predict_hierarchical_music(
    statistics: np.ndarray,
    view_mask: np.ndarray,
    checkpoint_paths: list[Path],
    device: str = "cuda",
    batch_size: int = 128,
) -> np.ndarray:
    if not checkpoint_paths:
        raise ValueError("at least one hierarchical EAT checkpoint is required")
    target = torch.device(device)
    member_logits = []
    expected_projection = None
    for path in checkpoint_paths:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        if checkpoint.get("model_type") != "hierarchical_eat_music":
            raise ValueError(f"invalid hierarchical EAT checkpoint: {path}")
        projection = np.asarray(checkpoint["projection"], dtype=np.float32)
        if expected_projection is None:
            expected_projection = projection
        elif not np.array_equal(expected_projection, projection):
            raise ValueError("hierarchical EAT checkpoints use different projections")
        model = _model(checkpoint, target)
        logits = []
        for offset in range(0, len(statistics), batch_size):
            values = torch.from_numpy(statistics[offset:offset + batch_size]).to(
                device=target, dtype=torch.float32
            )
            mask = torch.from_numpy(view_mask[offset:offset + batch_size]).to(target)
            with torch.autocast(
                device_type=target.type, dtype=torch.bfloat16,
                enabled=target.type == "cuda",
            ):
                logits.append(model(values, mask).float().cpu().numpy())
        member_logits.append(np.concatenate(logits))
        del model
    return _sigmoid(np.mean(member_logits, axis=0))


def apply_hierarchical_eat_music_fusion(
    submission_path: Path,
    statistics_path: Path,
    checkpoint_paths: list[Path],
    device: str = "cuda",
    telephone_ids_path: Path | None = None,
    music_weight: float = 0.15,
    file_weight: float = 0.15,
    phone_music_weight: float = 0.30,
    phone_file_weight: float = 0.30,
    file_music_presence_threshold: float = 0.50,
) -> None:
    for value in (
        music_weight, file_weight, phone_music_weight, phone_file_weight,
        file_music_presence_threshold,
    ):
        if not 0 <= value <= 1:
            raise ValueError("fusion weights and threshold must lie in [0, 1]")
    values = np.load(statistics_path, allow_pickle=False)
    ids = values["ids"].astype(str)
    statistics = values["statistics"]
    view_mask = values["view_mask"].astype(bool)
    if len(set(ids)) != len(ids):
        raise ValueError("hierarchical statistics contain duplicate IDs")
    expert = predict_hierarchical_music(
        statistics, view_mask, checkpoint_paths, device=device
    )
    expert_by_id = dict(zip(ids, expert))
    phone_ids: set[str] = set()
    if telephone_ids_path is not None:
        telephone = np.load(telephone_ids_path, allow_pickle=False)
        phone_ids = set(telephone["ids"].astype(str))

    with submission_path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        columns = list(reader.fieldnames or [])
        rows = list(reader)
    row_ids = [row["ID"] for row in rows]
    if set(row_ids) != set(expert_by_id):
        raise ValueError("submission and hierarchical statistics IDs differ")
    for row in rows:
        item_id = row["ID"]
        phone = item_id in phone_ids
        score = expert_by_id[item_id]
        row["MUSIC_FAKE_PROB"] = round(_fuse(
            float(row["MUSIC_FAKE_PROB"]), score,
            phone_music_weight if phone else music_weight,
        ), 10)
        if float(row["MUSIC_PRESENT_PROB"]) >= file_music_presence_threshold:
            row["FILE_FAKE_PROB"] = round(_fuse(
                float(row["FILE_FAKE_PROB"]), score,
                phone_file_weight if phone else file_weight,
            ), 10)
    temporary = submission_path.with_suffix(".tmp.csv")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(submission_path)
