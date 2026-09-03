"""Inference and logit-space fusion for the SPEAR temporal joint expert."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import torch

try:  # package import in tests; flat import in the offline submission
    from .spear_temporal_joint_head import SpearTemporalJointHead
except ImportError:  # pragma: no cover - exercised by script.py
    from spear_temporal_joint_head import SpearTemporalJointHead


def _logit(values: np.ndarray | float) -> np.ndarray:
    values = np.clip(np.asarray(values, dtype=np.float64), 1e-5, 1 - 1e-5)
    return np.log(values) - np.log1p(-values)


def _sigmoid(values: np.ndarray | float) -> np.ndarray:
    return np.exp(-np.logaddexp(0, -np.asarray(values, dtype=np.float64)))


def _fuse(anchor: float, expert: float, weight: float) -> float:
    return float(_sigmoid((1 - weight) * _logit(anchor) + weight * _logit(expert)))


@torch.inference_mode()
def predict_temporal_joint(
    statistics_path: Path,
    checkpoint_path: Path,
    device: str = "cuda",
    batch_size: int = 64,
) -> tuple[np.ndarray, np.ndarray]:
    """Return IDs and ``[File, Voice, Music, VoicePresent, MusicPresent]``."""
    archive = np.load(statistics_path, allow_pickle=False)
    if not {"ids", "features", "mask"}.issubset(archive.files):
        raise ValueError("temporal-bin archive misses required fields")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state, config = checkpoint["model"], checkpoint["config"]
    model = SpearTemporalJointHead(
        int(config["feature_dimension"]), state["mean"], state["std"],
        hidden=int(config["hidden"]), dropout=float(config["dropout"]),
        temperature=float(config["temperature"]),
        minimum_presence_weight=float(config["minimum_presence_weight"]),
    ).to(device)
    model.load_state_dict(state, strict=True)
    model.eval()
    features = archive["features"]
    features = features.reshape(
        len(features), features.shape[1], features.shape[2], -1
    )
    mask = archive["mask"].astype(bool)
    target = torch.device(device)
    batches: list[np.ndarray] = []
    for offset in range(0, len(features), batch_size):
        _, outputs = model(
            torch.from_numpy(features[offset:offset + batch_size]).to(target),
            torch.from_numpy(mask[offset:offset + batch_size]).to(target),
        )
        batches.append(
            torch.stack(outputs, dim=-1).float().cpu().numpy()
        )
    if not batches:
        raise ValueError("temporal-bin archive is empty")
    return archive["ids"].astype(str), np.concatenate(batches)


def apply_spear_temporal_joint_fusion(
    submission_path: Path,
    statistics_path: Path,
    checkpoint_path: Path,
    device: str = "cuda",
    file_weight: float = 0.25,
    voice_weight: float = 0.30,
) -> None:
    """Fuse the joint expert into File and Voice without changing CPS/Music.

    The joint model uses local component-presence predictions as an internal
    soft router.  Its presence outputs are deliberately not written to the
    submission: the deployed EAT/PANNs CPS stack is stronger and independently
    leaderboard-validated.
    """
    for value, name in ((file_weight, "file_weight"), (voice_weight, "voice_weight")):
        if not 0 <= value <= 1:
            raise ValueError(f"{name} must be in [0, 1]")
    ids, probabilities = predict_temporal_joint(
        statistics_path, checkpoint_path, device=device
    )
    if probabilities.shape != (len(ids), 5):
        raise ValueError("joint expert must return five probabilities per file")
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate temporal-bin IDs")
    expert_by_id = dict(zip(ids, probabilities))
    with submission_path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        columns = list(reader.fieldnames or [])
        rows = list(reader)
    if set(row["ID"] for row in rows) != set(expert_by_id):
        raise ValueError("submission and temporal-bin IDs differ")
    for row in rows:
        expert = expert_by_id[row["ID"]]
        row["FILE_FAKE_PROB"] = round(_fuse(
            float(row["FILE_FAKE_PROB"]), float(expert[0]), file_weight
        ), 10)
        row["VOICE_FAKE_PROB"] = round(_fuse(
            float(row["VOICE_FAKE_PROB"]), float(expert[1]), voice_weight
        ), 10)
    temporary = submission_path.with_suffix(".tmp.csv")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(submission_path)
