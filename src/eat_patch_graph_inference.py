"""Offline scoring and bounded v18 fusion for the EAT patch-graph expert."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import torch

try:  # package import in tests; flat import in the offline submission
    from .eat_patch_graph import EatPatchGraphHead
except ImportError:  # pragma: no cover
    from eat_patch_graph import EatPatchGraphHead


def _logit(value: np.ndarray | float) -> np.ndarray:
    clipped = np.clip(value, 1e-6, 1 - 1e-6)
    return np.log(clipped) - np.log1p(-clipped)


def _sigmoid(value: np.ndarray | float) -> np.ndarray:
    return 1 / (1 + np.exp(-np.clip(value, -60, 60)))


def _load_model(path: Path, device: torch.device):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if checkpoint.get("model_type") != "eat_patch_graph_multitask":
        raise ValueError(f"not an EAT patch-graph checkpoint: {path}")
    model = EatPatchGraphHead(**checkpoint["config"])
    model.load_state_dict(checkpoint["model"], strict=True)
    return model.eval().to(device), checkpoint


@torch.inference_mode()
def score_patch_graph_statistics(
    statistics_path: Path,
    checkpoint_paths: list[Path],
    device: str = "cuda",
    batch_size: int = 128,
) -> tuple[np.ndarray, np.ndarray]:
    if not checkpoint_paths:
        raise ValueError("at least one patch-graph checkpoint is required")
    if batch_size <= 0:
        raise ValueError("batch size must be positive")
    archive = np.load(statistics_path, allow_pickle=False)
    ids = archive["ids"].astype(str)
    temporal = archive["temporal"]
    spectral = archive["spectral"]
    mask = archive["view_mask"]
    if len(set(ids)) != len(ids):
        raise ValueError("patch-graph statistics contain duplicate IDs")
    torch_device = torch.device(device)
    members = []
    for path in checkpoint_paths:
        model, checkpoint = _load_model(path, torch_device)
        if (
            not np.array_equal(archive["projection"], checkpoint["projection"])
            or not np.array_equal(archive["layers"], checkpoint["eat_layers"])
        ):
            raise ValueError("patch-graph cache metadata differs from checkpoint")
        logits = []
        for offset in range(0, len(ids), batch_size):
            end = offset + batch_size
            inputs = (
                torch.from_numpy(temporal[offset:end]).to(torch_device),
                torch.from_numpy(spectral[offset:end]).to(torch_device),
                torch.from_numpy(mask[offset:end]).to(torch_device),
            )
            logits.append(model(*inputs).float().cpu().numpy())
        members.append(np.concatenate(logits))
        del model
    mean_logits = np.stack(members).mean(axis=0)
    # The deployed expert is trained with the same structural File relation.
    direct = _sigmoid(mean_logits)
    component_or = 1 - (1 - direct[:, 0]) * (1 - direct[:, 1])
    file_weight = float(checkpoint["config"]["file_component_weight"])
    direct[:, 2] = (
        (1 - file_weight) * direct[:, 2] + file_weight * component_or
    )
    return ids, direct


def apply_eat_patch_graph_fusion(
    submission_path: Path,
    statistics_path: Path,
    checkpoint_paths: list[Path],
    device: str = "cuda",
    file_weight: float = .05,
    music_weight: float = .05,
) -> None:
    """Fuse File and Music only; Voice and both presence columns stay exact."""
    if not 0 <= file_weight <= 1 or not 0 <= music_weight <= 1:
        raise ValueError("fusion weights must lie in [0, 1]")
    ids, expert = score_patch_graph_statistics(
        statistics_path, checkpoint_paths, device=device
    )
    expert_by_id = dict(zip(ids, expert))
    with submission_path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        columns = list(reader.fieldnames or [])
        rows = list(reader)
    submission_ids = [row["ID"] for row in rows]
    if set(submission_ids) != set(expert_by_id):
        raise ValueError("submission/patch-graph statistic IDs differ")
    for row in rows:
        probability = expert_by_id[row["ID"]]
        row["FILE_FAKE_PROB"] = round(float(_sigmoid(
            (1 - file_weight) * _logit(float(row["FILE_FAKE_PROB"]))
            + file_weight * _logit(probability[2])
        )), 10)
        row["MUSIC_FAKE_PROB"] = round(float(_sigmoid(
            (1 - music_weight) * _logit(float(row["MUSIC_FAKE_PROB"]))
            + music_weight * _logit(probability[1])
        )), 10)
    temporary = submission_path.with_suffix(".tmp.csv")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(submission_path)
