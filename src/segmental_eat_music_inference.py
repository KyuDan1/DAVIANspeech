"""Offline soft-MoE inference for endpoint and long-range EAT music heads."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import torch

try:  # package import in tests; flat import in the offline submission
    from .hierarchical_eat_music_inference import (
        _fuse, _logit, _sigmoid, predict_hierarchical_music,
    )
    from .segmental_eat_music import SegmentalEatMusicHead
except ImportError:  # pragma: no cover - exercised by script.py
    from hierarchical_eat_music_inference import (
        _fuse, _logit, _sigmoid, predict_hierarchical_music,
    )
    from segmental_eat_music import SegmentalEatMusicHead


def _model(checkpoint, device: torch.device) -> SegmentalEatMusicHead:
    state = checkpoint["model"]
    model = SegmentalEatMusicHead(
        state["normalization_mean"], state["normalization_std"],
        **checkpoint["config"],
    )
    model.load_state_dict(state, strict=True)
    return model.to(device).eval()


def _shared_projection(paths: list[Path], expected_type: str) -> np.ndarray:
    projection = None
    for path in paths:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        if checkpoint.get("model_type") != expected_type:
            raise ValueError(f"invalid {expected_type} checkpoint: {path}")
        current = np.asarray(checkpoint["projection"], dtype=np.float32)
        if projection is None:
            projection = current
        elif not np.array_equal(projection, current):
            raise ValueError(f"{expected_type} checkpoints use different projections")
    if projection is None:
        raise ValueError(f"at least one {expected_type} checkpoint is required")
    return projection


@torch.inference_mode()
def predict_segmental_music(
    statistics: np.ndarray,
    view_mask: np.ndarray,
    checkpoint_paths: list[Path],
    device: str = "cuda",
    batch_size: int = 128,
) -> np.ndarray:
    """Return an equal-logit ensemble of frozen segment-content heads."""
    if not checkpoint_paths:
        raise ValueError("at least one segmental EAT checkpoint is required")
    target = torch.device(device)
    member_logits = []
    _shared_projection(checkpoint_paths, "segmental_eat_music")
    for path in checkpoint_paths:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
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


def apply_segmental_eat_music_fusion(
    submission_path: Path,
    hierarchical_statistics_path: Path,
    hierarchical_checkpoint_paths: list[Path],
    segmental_statistics_path: Path,
    segmental_checkpoint_paths: list[Path],
    device: str = "cuda",
    telephone_ids_path: Path | None = None,
    expert_weight: float = 0.25,
    music_weight: float = 0.20,
    file_weight: float = 0.10,
    phone_music_weight: float = 0.30,
    phone_file_weight: float = 0.30,
    file_music_presence_threshold: float = 0.50,
) -> None:
    """Fuse a fixed inner soft-MoE, then apply the established phone weights."""
    for value in (
        expert_weight, music_weight, file_weight, phone_music_weight,
        phone_file_weight, file_music_presence_threshold,
    ):
        if not 0 <= value <= 1:
            raise ValueError("fusion weights and threshold must lie in [0, 1]")
    hierarchical_projection = _shared_projection(
        hierarchical_checkpoint_paths, "hierarchical_eat_music"
    )
    segmental_projection = _shared_projection(
        segmental_checkpoint_paths, "segmental_eat_music"
    )
    if not np.array_equal(hierarchical_projection, segmental_projection):
        raise ValueError("hierarchical and segmental EAT projections differ")

    with np.load(hierarchical_statistics_path, allow_pickle=False) as values:
        hierarchical_ids = values["ids"].astype(str)
        hierarchical = predict_hierarchical_music(
            values["statistics"], values["view_mask"].astype(bool),
            hierarchical_checkpoint_paths, device=device,
        )
    with np.load(segmental_statistics_path, allow_pickle=False) as values:
        segmental_ids = values["ids"].astype(str)
        segmental = predict_segmental_music(
            values["statistics"], values["view_mask"].astype(bool),
            segmental_checkpoint_paths, device=device,
        )
    if len(set(hierarchical_ids)) != len(hierarchical_ids):
        raise ValueError("hierarchical statistics contain duplicate IDs")
    if len(set(segmental_ids)) != len(segmental_ids):
        raise ValueError("segmental statistics contain duplicate IDs")
    if set(hierarchical_ids) != set(segmental_ids):
        raise ValueError("hierarchical and segmental statistic IDs differ")
    segmental_by_id = dict(zip(segmental_ids, segmental))
    aligned_segmental = np.asarray([
        segmental_by_id[item_id] for item_id in hierarchical_ids
    ])
    expert = _sigmoid(
        (1 - expert_weight) * _logit(hierarchical)
        + expert_weight * _logit(aligned_segmental)
    )
    expert_by_id = dict(zip(hierarchical_ids, expert))

    phone_ids: set[str] = set()
    if telephone_ids_path is not None:
        with np.load(telephone_ids_path, allow_pickle=False) as telephone:
            phone_ids = set(telephone["ids"].astype(str))
    with submission_path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        columns = list(reader.fieldnames or [])
        rows = list(reader)
    if set(row["ID"] for row in rows) != set(expert_by_id):
        raise ValueError("submission and EAT statistic IDs differ")
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
