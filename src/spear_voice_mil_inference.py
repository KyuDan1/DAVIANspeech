"""Inference and Voice-only fusion for the separation-free SPEAR MIL expert."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import torch

try:
    from .spear_voice_mil_head import SpearVoiceMILHead
except ImportError:  # pragma: no cover - offline flat import
    from spear_voice_mil_head import SpearVoiceMILHead


def _logit(values) -> np.ndarray:
    values = np.clip(np.asarray(values, dtype=np.float64), 1e-5, 1 - 1e-5)
    return np.log(values) - np.log1p(-values)


def _sigmoid(values) -> np.ndarray:
    return np.exp(-np.logaddexp(0, -np.asarray(values, dtype=np.float64)))


@torch.inference_mode()
def predict_spear_voice_mil(
    statistics_path: Path,
    checkpoint_paths: list[Path],
    *,
    device: str = "cuda",
    batch_size: int = 128,
) -> tuple[np.ndarray, np.ndarray]:
    """Return IDs and a uniform logit ensemble from fixed tiny heads."""
    if not checkpoint_paths or batch_size <= 0:
        raise ValueError("Voice MIL needs checkpoints and a positive batch size")
    archive = np.load(statistics_path, allow_pickle=False)
    required = {"ids", "features", "mask", "projection", "layers", "bins"}
    if missing := required.difference(archive.files):
        raise ValueError(f"Voice MIL statistics miss fields: {sorted(missing)}")
    ids = archive["ids"].astype(str)
    if len(set(ids)) != len(ids):
        raise ValueError("Voice MIL statistics contain duplicate IDs")
    features = archive["features"].reshape(
        len(ids), archive["features"].shape[1], archive["features"].shape[2], -1,
    )
    mask = archive["mask"].astype(bool)
    target = torch.device(device)
    member_probability: list[np.ndarray] = []
    for checkpoint_path in checkpoint_paths:
        checkpoint = torch.load(
            checkpoint_path, map_location="cpu", weights_only=False,
        )
        if checkpoint.get("model_type") != "spear_voice_mil_v1":
            raise ValueError("not a SPEAR Voice MIL checkpoint")
        for archive_key, checkpoint_key in (
            ("projection", "projection"), ("layers", "layers"), ("bins", "bins"),
        ):
            if not np.array_equal(archive[archive_key], checkpoint[checkpoint_key]):
                raise ValueError(f"Voice MIL cache {archive_key} differs")
        config = checkpoint["config"]
        state = checkpoint["model"]
        model = SpearVoiceMILHead(
            int(config["feature_dimension"]), state["mean"], state["std"],
            hidden=int(config["hidden"]), dropout=float(config["dropout"]),
            temperature=float(config["temperature"]),
            minimum_presence_weight=float(config["minimum_presence_weight"]),
        ).to(target)
        model.load_state_dict(state, strict=True)
        model.eval()
        batches = []
        for offset in range(0, len(ids), batch_size):
            probability, _, _ = model(
                torch.from_numpy(features[offset:offset + batch_size]).to(target),
                torch.from_numpy(mask[offset:offset + batch_size]).to(target),
            )
            batches.append(probability.float().cpu().numpy())
        member_probability.append(np.concatenate(batches))
    expert = _sigmoid(np.mean([_logit(value) for value in member_probability], axis=0))
    return ids, expert


def apply_spear_voice_mil_residual(
    submission_path: Path,
    statistics_path: Path,
    checkpoint_paths: list[Path],
    *,
    device: str = "cuda",
    voice_weight: float = 0.025,
) -> None:
    """Change only Voice authenticity after strict per-file ID alignment."""
    if not 0 <= voice_weight <= 1:
        raise ValueError("Voice MIL residual weight must lie in [0, 1]")
    ids, expert = predict_spear_voice_mil(
        statistics_path, checkpoint_paths, device=device,
    )
    by_id = dict(zip(ids, expert))
    with Path(submission_path).open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        columns = list(reader.fieldnames or [])
        rows = list(reader)
    row_ids = [row["ID"] for row in rows]
    if len(row_ids) != len(set(row_ids)) or set(row_ids) != set(by_id):
        raise ValueError("submission and Voice MIL IDs differ")
    for row in rows:
        anchor = float(row["VOICE_FAKE_PROB"])
        row["VOICE_FAKE_PROB"] = round(float(_sigmoid(
            (1 - voice_weight) * _logit(anchor)
            + voice_weight * _logit(by_id[row["ID"]])
        )), 10)
    temporary = Path(submission_path).with_suffix(".tmp.csv")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(submission_path)
