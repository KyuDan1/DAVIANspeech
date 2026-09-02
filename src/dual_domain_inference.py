"""Offline ensemble inference and frozen anchor fusion for dual-domain heads."""

from __future__ import annotations

import csv
import gc
from pathlib import Path

import numpy as np
import torch

try:  # package import in tests; flat import in the offline submission
    from .dual_domain_head import DualDomainHead
except ImportError:  # pragma: no cover - exercised by script.py
    from dual_domain_head import DualDomainHead


def _load_statistics(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    values = np.load(path, allow_pickle=False)
    return (
        values["ids"].astype(str), values["statistics"], values["view_mask"]
    )


def _logit(value: np.ndarray) -> np.ndarray:
    value = np.clip(value, 1e-5, 1 - 1e-5)
    return np.log(value) - np.log1p(-value)


def _fuse(anchor: np.ndarray, expert: np.ndarray, weight: float) -> np.ndarray:
    mixed = (1 - weight) * _logit(anchor) + weight * _logit(expert)
    return 1 / (1 + np.exp(-mixed))


@torch.inference_mode()
def apply_dual_domain_fusion(
    submission_path: Path,
    eat_statistics_path: Path,
    spear_statistics_path: Path,
    checkpoint_paths: list[Path],
    device: str = "cuda",
    batch_size: int = 64,
    file_weight: float = 0.50,
    voice_weight: float = 0.30,
    music_weight: float = 0.50,
    routed_ids_path: Path | None = None,
) -> None:
    """Apply the frozen ensemble, optionally only to pre-routed file IDs."""
    eat_ids, eat, eat_mask = _load_statistics(eat_statistics_path)
    spear_ids, spear, spear_mask = _load_statistics(spear_statistics_path)
    if set(eat_ids) != set(spear_ids):
        raise ValueError("EAT/SPEAR statistic IDs differ")
    spear_index = {item: index for index, item in enumerate(spear_ids)}
    order = np.asarray([spear_index[item] for item in eat_ids])
    spear, spear_mask = spear[order], spear_mask[order]
    target_device = torch.device(device)
    member_predictions = []
    for path in checkpoint_paths:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        model = DualDomainHead(**checkpoint["config"]).to(target_device)
        model.load_state_dict(checkpoint["model"])
        model.eval()
        normal = checkpoint["normalization"]
        eat_mean = torch.from_numpy(normal["eat_mean"]).to(target_device)[None, None]
        eat_std = torch.from_numpy(normal["eat_std"]).to(target_device)[None, None]
        spear_mean = torch.from_numpy(normal["spear_mean"]).to(target_device)[None, None]
        spear_std = torch.from_numpy(normal["spear_std"]).to(target_device)[None, None]
        batches = []
        for offset in range(0, len(eat_ids), batch_size):
            eat_batch = torch.from_numpy(eat[offset:offset + batch_size]).to(
                target_device, dtype=torch.float32
            )
            spear_batch = torch.from_numpy(spear[offset:offset + batch_size]).to(
                target_device, dtype=torch.float32
            )
            eat_batch = ((eat_batch - eat_mean) / eat_std).clamp_(-8, 8)
            spear_batch = ((spear_batch - spear_mean) / spear_std).clamp_(-8, 8)
            eat_mask_batch = torch.from_numpy(
                eat_mask[offset:offset + batch_size]
            ).to(target_device)
            spear_mask_batch = torch.from_numpy(
                spear_mask[offset:offset + batch_size]
            ).to(target_device)
            with torch.autocast(
                device_type=target_device.type, dtype=torch.bfloat16,
                enabled=target_device.type == "cuda",
            ):
                task_logits, joint_logits = model(
                    eat_batch, spear_batch, eat_mask_batch, spear_mask_batch
                )
            batches.append(
                model.probabilities(task_logits.float(), joint_logits.float()).cpu()
            )
        member_predictions.append(torch.cat(batches).numpy())
        del model, checkpoint
        gc.collect()
        torch.cuda.empty_cache()
    expert = np.mean(member_predictions, axis=0)
    expert_by_id = {item: expert[index] for index, item in enumerate(eat_ids)}

    with submission_path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        columns = list(reader.fieldnames or [])
        rows = list(reader)
    if {row["ID"] for row in rows} != set(expert_by_id):
        raise ValueError("Submission/statistic IDs differ")
    routed_ids = None
    if routed_ids_path is not None:
        routed_ids = set(np.load(routed_ids_path, allow_pickle=False)["ids"].astype(str))
        unknown = routed_ids - set(expert_by_id)
        if unknown:
            raise ValueError(f"Unknown routed statistic IDs: {sorted(unknown)[:5]}")
    weights = {
        "VOICE_FAKE_PROB": (0, voice_weight),
        "MUSIC_FAKE_PROB": (1, music_weight),
        "FILE_FAKE_PROB": (2, file_weight),
    }
    for row in rows:
        if routed_ids is not None and row["ID"] not in routed_ids:
            continue
        score = expert_by_id[row["ID"]]
        for column, (index, weight) in weights.items():
            if weight <= 0:
                continue
            row[column] = round(float(_fuse(
                np.asarray([float(row[column])]),
                np.asarray([score[index]]), weight,
            )[0]), 10)
    temporary = submission_path.with_suffix(".tmp.csv")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(submission_path)
