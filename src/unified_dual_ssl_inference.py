"""Offline Music-only residual from the separation-free unified SSL head."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import torch

try:  # package import in tests; flat import in the offline submission
    from .unified_dual_ssl_head import UnifiedDualSSLHead
except ImportError:  # pragma: no cover - exercised by script.py
    from unified_dual_ssl_head import UnifiedDualSSLHead


def _logit(values) -> np.ndarray:
    values = np.clip(np.asarray(values, dtype=np.float64), 1e-5, 1 - 1e-5)
    return np.log(values) - np.log1p(-values)


def _sigmoid(values) -> np.ndarray:
    return np.exp(-np.logaddexp(0, -np.asarray(values, dtype=np.float64)))


def _metadata(checkpoint: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    required = {"model", "config", "eat_projection", "spear_projection",
                "spear_layers", "spear_bins"}
    missing = required.difference(checkpoint)
    if missing:
        raise ValueError(f"unified checkpoint is missing {sorted(missing)}")
    return (
        np.asarray(checkpoint["eat_projection"], dtype=np.float32),
        np.asarray(checkpoint["spear_projection"], dtype=np.float32),
        np.asarray(checkpoint["spear_layers"], dtype=np.int16),
        int(checkpoint["spear_bins"]),
    )


@torch.inference_mode()
def predict_unified_tasks(
    eat_statistics: np.ndarray,
    spear_features: np.ndarray,
    eat_mask: np.ndarray,
    spear_mask: np.ndarray,
    checkpoint_paths: list[Path],
    device: str = "cuda",
    batch_size: int = 128,
) -> tuple[np.ndarray, tuple[np.ndarray, np.ndarray, np.ndarray, int]]:
    """Return Voice/Music/File probabilities from an equal-logit ensemble.

    Voice and Music use their native task logits.  File first applies the
    head's trained component-OR calibration, then converts that probability
    back to a logit before checkpoint averaging.  This exactly preserves the
    historical Music path while exposing the other two expert decisions.
    """
    if not checkpoint_paths:
        raise ValueError("at least one unified checkpoint is required")
    if len(eat_statistics) != len(spear_features):
        raise ValueError("EAT and SPEAR feature counts differ")
    if eat_mask.shape != eat_statistics.shape[:2]:
        raise ValueError("EAT mask has incompatible shape")
    if spear_mask.shape != spear_features.shape[:3]:
        raise ValueError("SPEAR mask has incompatible shape")
    target = torch.device(device)
    member_logits = []
    expected_metadata = None
    for path in checkpoint_paths:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        metadata = _metadata(checkpoint)
        if expected_metadata is None:
            expected_metadata = metadata
        elif any(
            not np.array_equal(left, right)
            for left, right in zip(expected_metadata[:3], metadata[:3])
        ) or expected_metadata[3] != metadata[3]:
            raise ValueError("unified checkpoints use different projections")
        model = UnifiedDualSSLHead(**checkpoint["config"]).to(target)
        model.load_state_dict(checkpoint["model"], strict=True)
        model.eval()
        logits = []
        for offset in range(0, len(eat_statistics), batch_size):
            end = offset + batch_size
            task_logits, _, _ = model(
                torch.from_numpy(eat_statistics[offset:end]).to(target),
                torch.from_numpy(spear_features[offset:end]).to(target),
                torch.from_numpy(eat_mask[offset:end]).to(target),
                torch.from_numpy(spear_mask[offset:end]).to(target),
            )
            task_logits = task_logits.float().cpu().numpy()
            probabilities = model.probabilities(
                torch.from_numpy(task_logits)
            ).numpy()
            # Native logits are retained for the direct component heads.  The
            # calibrated File probability does not have a native equivalent.
            task_logits[:, 2] = _logit(probabilities[:, 2])
            logits.append(task_logits)
        member_logits.append(np.concatenate(logits))
        del model
    return _sigmoid(np.mean(member_logits, axis=0)), expected_metadata


def predict_unified_music(
    eat_statistics: np.ndarray,
    spear_features: np.ndarray,
    eat_mask: np.ndarray,
    spear_mask: np.ndarray,
    checkpoint_paths: list[Path],
    device: str = "cuda",
    batch_size: int = 128,
) -> tuple[np.ndarray, tuple[np.ndarray, np.ndarray, np.ndarray, int]]:
    """Backward-compatible Music-only view of :func:`predict_unified_tasks`."""
    probability, metadata = predict_unified_tasks(
        eat_statistics, spear_features, eat_mask, spear_mask,
        checkpoint_paths, device=device, batch_size=batch_size,
    )
    return probability[:, 1], metadata


def apply_unified_music_fusion(
    submission_path: Path,
    eat_statistics_path: Path,
    spear_features_path: Path,
    checkpoint_paths: list[Path],
    device: str = "cuda",
    music_weight: float = .20,
    mixed_music_weight: float | None = None,
    mixed_presence_threshold: float = .70,
    expert_output_path: Path | None = None,
) -> None:
    """Fuse only Music; File, Voice, and both presence columns stay bit-exact.

    ``mixed_music_weight`` enables the audited shallow content router.  A
    value of ``None`` selects the more generator-robust fixed soft MoE.
    """
    weights = [music_weight, mixed_presence_threshold]
    if mixed_music_weight is not None:
        weights.append(mixed_music_weight)
    if any(not 0 <= value <= 1 for value in weights):
        raise ValueError("fusion weights and threshold must lie in [0, 1]")
    with np.load(eat_statistics_path, allow_pickle=False) as archive:
        eat_ids = archive["ids"].astype(str)
        eat_statistics = archive["statistics"]
        eat_mask = archive["view_mask"].astype(bool)
        eat_projection = archive["projection"]
    with np.load(spear_features_path, allow_pickle=False) as archive:
        spear_ids = archive["ids"].astype(str)
        spear_features = archive["features"]
        spear_mask = archive["mask"].astype(bool)
        spear_projection = archive["projection"]
        spear_layers = archive["layers"]
        spear_bins = int(archive["bins"])
    if len(set(eat_ids)) != len(eat_ids) or len(set(spear_ids)) != len(spear_ids):
        raise ValueError("unified statistics contain duplicate IDs")
    if set(eat_ids) != set(spear_ids):
        raise ValueError("EAT and SPEAR statistic IDs differ")
    spear_by_id = {item: index for index, item in enumerate(spear_ids)}
    order = np.asarray([spear_by_id[item] for item in eat_ids], dtype=np.int64)
    task_expert, metadata = predict_unified_tasks(
        eat_statistics, spear_features[order], eat_mask, spear_mask[order],
        checkpoint_paths, device=device,
    )
    expert = task_expert[:, 1]
    if not np.array_equal(eat_projection, metadata[0]):
        raise ValueError("EAT cache projection differs from unified checkpoint")
    if not np.array_equal(spear_projection, metadata[1]):
        raise ValueError("SPEAR cache projection differs from unified checkpoint")
    if not np.array_equal(spear_layers, metadata[2]) or spear_bins != metadata[3]:
        raise ValueError("SPEAR cache layout differs from unified checkpoint")
    expert_by_id = dict(zip(eat_ids, expert))
    if expert_output_path is not None:
        expert_output_path = Path(expert_output_path)
        expert_output_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            expert_output_path,
            ids=eat_ids,
            probabilities=task_expert.astype(np.float32),
        )

    with submission_path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        columns = list(reader.fieldnames or [])
        rows = list(reader)
    if set(row["ID"] for row in rows) != set(expert_by_id):
        raise ValueError("submission and unified statistic IDs differ")
    for row in rows:
        weight = music_weight
        if mixed_music_weight is not None and (
            float(row["VOICE_PRESENT_PROB"]) >= mixed_presence_threshold
            and float(row["MUSIC_PRESENT_PROB"]) >= mixed_presence_threshold
        ):
            weight = mixed_music_weight
        row["MUSIC_FAKE_PROB"] = round(float(_sigmoid(
            (1 - weight) * _logit(float(row["MUSIC_FAKE_PROB"]))
            + weight * _logit(expert_by_id[row["ID"]])
        )), 10)
    temporary = submission_path.with_suffix(".tmp.csv")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(submission_path)
