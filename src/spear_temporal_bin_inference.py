"""Offline inference and conservative fusion for a SPEAR temporal-bin head."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import torch

try:  # package import in tests; flat import in the offline submission
    from .spear_temporal_bin_head import SpearTemporalBinHead
except ImportError:  # pragma: no cover - exercised by script.py
    from spear_temporal_bin_head import SpearTemporalBinHead


def _component_file_evidence(*args, **kwargs):
    """Load the v28-only dependency only when its fusion path is requested."""
    try:
        from .presence_weighted_file_fusion import component_file_evidence
    except ImportError:  # pragma: no cover - exercised by script.py
        from presence_weighted_file_fusion import component_file_evidence
    return component_file_evidence(*args, **kwargs)


def _logit(values: np.ndarray) -> np.ndarray:
    values = np.clip(np.asarray(values, dtype=np.float64), 1e-5, 1 - 1e-5)
    return np.log(values) - np.log1p(-values)


def _sigmoid(values: np.ndarray) -> np.ndarray:
    return np.exp(-np.logaddexp(0, -np.asarray(values, dtype=np.float64)))


@torch.inference_mode()
def predict_temporal_bin_music(
    statistics_path: Path, checkpoint_path: Path,
    device: str = "cuda", batch_size: int = 64,
) -> tuple[np.ndarray, np.ndarray]:
    archive = np.load(statistics_path, allow_pickle=False)
    if not {"ids", "features", "mask"}.issubset(archive.files):
        raise ValueError("temporal-bin archive misses required fields")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state, config = checkpoint["model"], checkpoint["config"]
    model = SpearTemporalBinHead(
        int(config["feature_dimension"]), state["mean"], state["std"],
        hidden=int(config["hidden"]), dropout=float(config["dropout"]),
        temperature=float(config["temperature"]),
        minimum_presence_weight=float(config["minimum_presence_weight"]),
    ).to(device)
    model.load_state_dict(state, strict=True); model.eval()
    features = archive["features"]
    features = features.reshape(
        len(features), features.shape[1], features.shape[2], -1
    )
    mask = archive["mask"].astype(bool)
    scores = []
    target = torch.device(device)
    for offset in range(0, len(features), batch_size):
        _, probability = model(
            torch.from_numpy(features[offset:offset + batch_size]).to(target),
            torch.from_numpy(mask[offset:offset + batch_size]).to(target),
        )
        scores.append(probability.float().cpu().numpy())
    return archive["ids"].astype(str), np.concatenate(scores)


def apply_spear_temporal_bin_fusion(
    submission_path: Path, statistics_path: Path, checkpoint_path: Path,
    device: str = "cuda", music_weight: float = 0.40,
    file_consistency_update_weight: float = 0.75,
    base_file_consistency_weight: float = 0.50,
    presence_logit_weight: float = 0.50,
) -> None:
    """Fuse Music and partially refresh an already-applied v28 File score."""
    for value, name in (
        (music_weight, "music_weight"),
        (file_consistency_update_weight, "file_consistency_update_weight"),
        (base_file_consistency_weight, "base_file_consistency_weight"),
        (presence_logit_weight, "presence_logit_weight"),
    ):
        if not 0 <= value <= 1:
            raise ValueError(f"{name} must be in [0, 1]")
    if base_file_consistency_weight >= 1:
        raise ValueError("base File consistency weight must be below one")
    ids, expert = predict_temporal_bin_music(
        statistics_path, checkpoint_path, device=device
    )
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate temporal-bin IDs")
    expert_by_id = dict(zip(ids, expert))
    with submission_path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        columns = list(reader.fieldnames or [])
        rows = list(reader)
    if set(row["ID"] for row in rows) != set(expert_by_id):
        raise ValueError("submission and temporal-bin IDs differ")
    for row in rows:
        old_file = float(row["FILE_FAKE_PROB"])
        anchor = float(row["MUSIC_FAKE_PROB"])
        score = float(expert_by_id[row["ID"]])
        fused = _sigmoid((1 - music_weight) * _logit(anchor) + music_weight * _logit(score))
        row["MUSIC_FAKE_PROB"] = round(float(fused), 10)
        old_evidence = _component_file_evidence(
            float(row["VOICE_FAKE_PROB"]), anchor,
            float(row["VOICE_PRESENT_PROB"]), float(row["MUSIC_PRESENT_PROB"]),
            presence_logit_weight,
        )
        # Invert the v28 blend to recover its pre-consistency File logit, then
        # recompute it with the updated Music evidence.  Retaining 25% of the
        # old v28 score was the development-tied conservative setting.
        pre_consistency_logit = (
            _logit(old_file)
            - base_file_consistency_weight * _logit(old_evidence)
        ) / (1 - base_file_consistency_weight)
        new_evidence = _component_file_evidence(
            float(row["VOICE_FAKE_PROB"]), float(fused),
            float(row["VOICE_PRESENT_PROB"]), float(row["MUSIC_PRESENT_PROB"]),
            presence_logit_weight,
        )
        recomputed = _sigmoid(
            (1 - base_file_consistency_weight) * pre_consistency_logit
            + base_file_consistency_weight * _logit(new_evidence)
        )
        updated_file = _sigmoid(
            (1 - file_consistency_update_weight) * _logit(old_file)
            + file_consistency_update_weight * _logit(recomputed)
        )
        row["FILE_FAKE_PROB"] = round(float(updated_file), 10)
    temporary = submission_path.with_suffix(".tmp.csv")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader(); writer.writerows(rows)
    temporary.replace(submission_path)


def apply_spear_temporal_bin_music_only(
    submission_path: Path, statistics_path: Path, checkpoint_path: Path,
    device: str = "cuda", music_weight: float = 0.50,
) -> None:
    """Fuse only Music, preserving File/Voice/Presence predictions exactly.

    This is intentionally separate from ``apply_spear_temporal_bin_fusion``:
    the latter also refreshes a v28-specific File-consistency term.  Exact-v18
    ablations need a component-only residual whose leaderboard effect can be
    attributed to Music EER alone.
    """
    if not 0 <= music_weight <= 1:
        raise ValueError("music_weight must be in [0, 1]")
    ids, expert = predict_temporal_bin_music(
        statistics_path, checkpoint_path, device=device
    )
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate temporal-bin IDs")
    expert_by_id = dict(zip(ids, expert))
    with submission_path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        columns = list(reader.fieldnames or [])
        rows = list(reader)
    if set(row["ID"] for row in rows) != set(expert_by_id):
        raise ValueError("submission and temporal-bin IDs differ")
    for row in rows:
        anchor = float(row["MUSIC_FAKE_PROB"])
        score = float(expert_by_id[row["ID"]])
        fused = _sigmoid(
            (1 - music_weight) * _logit(anchor)
            + music_weight * _logit(score)
        )
        row["MUSIC_FAKE_PROB"] = round(float(fused), 10)
    temporary = submission_path.with_suffix(".tmp.csv")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader(); writer.writerows(rows)
    temporary.replace(submission_path)
