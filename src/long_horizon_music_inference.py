"""Frozen original-audio multi-view music expert and conservative routing.

The EAT and SPEAR statistics are already produced by the deployed pipeline.
This module projects those start/middle/end statistics with fixed, label-free
random matrices, evaluates two linear music heads, and combines them in logit
space.  Only the internal head mixture is routed: narrow-band calls use the
average-risk head, while other audio uses the frozen two-head MoE.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F


def _load_statistics(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    values = np.load(path, allow_pickle=False)
    required = {"ids", "statistics", "view_mask"}
    if not required.issubset(values.files):
        raise ValueError(f"Missing statistics fields in {path}")
    ids = values["ids"].astype(str)
    if len(set(ids)) != len(ids):
        raise ValueError(f"Duplicate statistic IDs in {path}")
    return ids, values["statistics"], values["view_mask"].astype(bool)


def _logit(values: np.ndarray) -> np.ndarray:
    values = np.clip(np.asarray(values, dtype=np.float64), 1e-5, 1 - 1e-5)
    return np.log(values) - np.log1p(-values)


def _sigmoid(values: np.ndarray) -> np.ndarray:
    return np.exp(-np.logaddexp(0.0, -np.asarray(values, dtype=np.float64)))


def _fuse(anchor: np.ndarray, expert: np.ndarray, weight) -> np.ndarray:
    weight = np.asarray(weight, dtype=np.float64)
    return _sigmoid((1 - weight) * _logit(anchor) + weight * _logit(expert))


def _validate_checkpoint(state) -> int:
    required = {
        "eat_matrix", "spear_matrix", "member_count", "member_logit_weights",
        "phone_member_logit_weights", "music_weight", "file_weight",
        "file_music_presence_threshold",
    }
    missing = required.difference(state.files)
    if missing:
        raise ValueError(f"Missing music checkpoint fields: {sorted(missing)}")
    member_count = int(state["member_count"])
    if member_count < 1:
        raise ValueError("Music checkpoint has no members")
    for index in range(member_count):
        for field in ("mean", "std", "weight", "bias"):
            key = f"member_{index}_{field}"
            if key not in state.files:
                raise ValueError(f"Missing music checkpoint field: {key}")
    for key in ("member_logit_weights", "phone_member_logit_weights"):
        weights = np.asarray(state[key], dtype=np.float64)
        if weights.shape != (member_count,) or np.any(weights < 0):
            raise ValueError(f"Invalid {key}")
        if not np.isclose(weights.sum(), 1.0):
            raise ValueError(f"{key} must sum to one")
    return member_count


@torch.inference_mode()
def predict_long_horizon_music(
    eat: np.ndarray,
    spear: np.ndarray,
    eat_mask: np.ndarray,
    spear_mask: np.ndarray,
    checkpoint_path: Path,
    device: str = "cuda",
    batch_size: int = 64,
    phone_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Predict one music-fake probability per aligned statistics row."""
    if eat.shape[:2] != spear.shape[:2]:
        raise ValueError("EAT/SPEAR statistic view shapes differ")
    if eat_mask.shape != spear_mask.shape or eat_mask.shape != eat.shape[:2]:
        raise ValueError("Statistic masks do not match view shapes")
    if eat.shape[2:] != (4, 768):
        raise ValueError(f"Unexpected EAT statistic shape: {eat.shape}")
    if spear.shape[2:] != (13, 4, 1280):
        raise ValueError(f"Unexpected SPEAR statistic shape: {spear.shape}")
    sample_count = len(eat)
    if phone_mask is None:
        phone_mask = np.zeros(sample_count, dtype=bool)
    phone_mask = np.asarray(phone_mask, dtype=bool)
    if phone_mask.shape != (sample_count,):
        raise ValueError("Telephone route mask has the wrong shape")

    state = np.load(checkpoint_path, allow_pickle=False)
    member_count = _validate_checkpoint(state)
    target_device = torch.device(device)
    eat_matrix = torch.from_numpy(
        np.asarray(state["eat_matrix"], dtype=np.float32)
    ).to(target_device)
    spear_matrix = torch.from_numpy(
        np.asarray(state["spear_matrix"], dtype=np.float32)
    ).to(target_device)
    if eat_matrix.shape[0] != 768 or spear_matrix.shape[0] != 1280:
        raise ValueError("Projection matrices have incompatible shapes")
    if eat_matrix.shape[1] != spear_matrix.shape[1]:
        raise ValueError("Projection matrices use different widths")

    member_parameters = []
    expected_features = 2 * (4 + 13 * 4) * eat_matrix.shape[1] + 1
    for index in range(member_count):
        mean = torch.from_numpy(
            np.asarray(state[f"member_{index}_mean"], dtype=np.float32)
        ).to(target_device)
        std = torch.from_numpy(
            np.asarray(state[f"member_{index}_std"], dtype=np.float32)
        ).to(target_device)
        weight = torch.from_numpy(
            np.asarray(state[f"member_{index}_weight"], dtype=np.float32)
        ).to(target_device).reshape(-1)
        bias = torch.as_tensor(
            float(state[f"member_{index}_bias"]),
            dtype=torch.float32, device=target_device,
        )
        if mean.shape != (expected_features,) or std.shape != mean.shape:
            raise ValueError("Music-head normalization has incompatible shape")
        if weight.shape != mean.shape or torch.any(std <= 0):
            raise ValueError("Music-head classifier has incompatible shape")
        member_parameters.append((mean, std, weight, bias))

    predictions = []
    valid_mask = np.asarray(eat_mask, dtype=bool) & np.asarray(spear_mask, dtype=bool)
    for offset in range(0, sample_count, batch_size):
        end = offset + batch_size
        eat_batch = torch.from_numpy(eat[offset:end]).to(
            target_device, dtype=torch.float32
        ).clamp_(-8, 8)
        spear_batch = torch.from_numpy(spear[offset:end]).to(
            target_device, dtype=torch.float32
        ).clamp_(-8, 8)
        eat_projected = F.layer_norm(
            eat_batch, (eat_batch.shape[-1],)
        ) @ eat_matrix
        spear_projected = F.layer_norm(
            spear_batch, (spear_batch.shape[-1],)
        ) @ spear_matrix
        values = torch.cat((
            eat_projected.flatten(2, 3),
            spear_projected.flatten(2, 4),
        ), dim=2)
        mask = torch.from_numpy(valid_mask[offset:end]).to(target_device)
        count = mask.sum(dim=1, keepdim=True).clamp_min(1).to(values.dtype)
        mask_weight = mask[:, :, None].to(values.dtype)
        mean_view = (values * mask_weight).sum(dim=1) / count
        second = (values.square() * mask_weight).sum(dim=1) / count
        dispersion = (second - mean_view.square()).clamp_min(0).sqrt()
        features = torch.cat((mean_view, dispersion, count / 3.0), dim=1)

        member_probabilities = []
        for mean, std, weight, bias in member_parameters:
            normalized = ((features - mean) / std).clamp_(-8, 8)
            member_probabilities.append(
                (normalized @ weight + bias).sigmoid().cpu().numpy()
            )
        member_probabilities = np.stack(member_probabilities, axis=1)
        normal_weights = np.asarray(state["member_logit_weights"], dtype=np.float64)
        phone_weights = np.asarray(
            state["phone_member_logit_weights"], dtype=np.float64
        )
        normal = _sigmoid(_logit(member_probabilities) @ normal_weights)
        phone = _sigmoid(_logit(member_probabilities) @ phone_weights)
        routed = np.where(phone_mask[offset:end], phone, normal)
        predictions.append(routed)
    return np.concatenate(predictions).astype(np.float64, copy=False)


def apply_long_horizon_music_fusion(
    submission_path: Path,
    eat_statistics_path: Path,
    spear_statistics_path: Path,
    checkpoint_path: Path,
    device: str = "cuda",
    batch_size: int = 64,
    telephone_ids_path: Path | None = None,
) -> None:
    """Fuse the frozen expert into Music and music-present File scores."""
    eat_ids, eat, eat_mask = _load_statistics(eat_statistics_path)
    spear_ids, spear, spear_mask = _load_statistics(spear_statistics_path)
    if set(eat_ids) != set(spear_ids):
        raise ValueError("EAT/SPEAR statistic IDs differ")
    spear_index = {sample_id: index for index, sample_id in enumerate(spear_ids)}
    order = np.asarray([spear_index[sample_id] for sample_id in eat_ids])
    spear, spear_mask = spear[order], spear_mask[order]

    phone_ids: set[str] = set()
    if telephone_ids_path is not None:
        routed = np.load(telephone_ids_path, allow_pickle=False)
        if "ids" not in routed.files:
            raise ValueError("Telephone route archive has no IDs")
        phone_ids = set(routed["ids"].astype(str))
        unknown = phone_ids.difference(eat_ids)
        if unknown:
            raise ValueError(f"Unknown telephone IDs: {sorted(unknown)[:5]}")
    phone_mask = np.asarray([sample_id in phone_ids for sample_id in eat_ids])
    expert = predict_long_horizon_music(
        eat, spear, eat_mask, spear_mask, checkpoint_path,
        device=device, batch_size=batch_size, phone_mask=phone_mask,
    )
    expert_by_id = dict(zip(eat_ids, expert))

    state = np.load(checkpoint_path, allow_pickle=False)
    music_weight = float(state["music_weight"])
    file_weight = float(state["file_weight"])
    presence_threshold = float(state["file_music_presence_threshold"])
    with submission_path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        columns = list(reader.fieldnames or [])
        rows = list(reader)
    row_ids = [row["ID"] for row in rows]
    if len(set(row_ids)) != len(row_ids):
        raise ValueError("Submission contains duplicate IDs")
    if set(row_ids) != set(expert_by_id):
        raise ValueError("Submission/statistic IDs differ")
    for row in rows:
        score = np.asarray([expert_by_id[row["ID"]]])
        row["MUSIC_FAKE_PROB"] = round(float(_fuse(
            np.asarray([float(row["MUSIC_FAKE_PROB"])]), score, music_weight,
        )[0]), 10)
        if float(row["MUSIC_PRESENT_PROB"]) >= presence_threshold:
            row["FILE_FAKE_PROB"] = round(float(_fuse(
                np.asarray([float(row["FILE_FAKE_PROB"])]), score, file_weight,
            )[0]), 10)
    temporary = submission_path.with_suffix(".tmp.csv")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(submission_path)
