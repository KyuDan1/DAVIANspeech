"""Add EAT AudioSet evidence to PANNs presence.

Presence and authenticity are deliberately decoupled by default.  Updating a
file-authenticity score through a hard presence threshold can propagate a small
component-presence error into ADS, while the competition scores presence
directly through CPS.
"""

from __future__ import annotations

import csv
import gc
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

try:  # package import in tests; flat import in the offline submission
    from .eat_presence import EatPresence, fuse_music_probe, fuse_presence
    from .pipeline import find_audio_files, load_audio, order_by_submission
    from .telephone_router import TelephoneRouter
except ImportError:  # pragma: no cover - exercised by script.py
    from eat_presence import EatPresence, fuse_music_probe, fuse_presence
    from pipeline import find_audio_files, load_audio, order_by_submission
    from telephone_router import TelephoneRouter


def latent_linear_probability(
    matrix: np.ndarray, mask: np.ndarray, checkpoint,
    temperature: float = 5.0,
) -> float:
    """Score one frozen EAT statistics matrix without importing sklearn."""
    values = matrix.astype(np.float32)
    valid = mask[:, None, None]
    count = max(int(mask.sum()), 1)
    view_mean = (values * valid).sum(axis=0) / count
    view_max = np.where(valid, values, -np.inf).max(axis=0)
    features = np.concatenate((view_mean, view_max), axis=0).reshape(-1)
    features = np.clip(
        (features - checkpoint["mean"]) / checkpoint["std"], -8, 8
    )
    decision = (
        features @ checkpoint["coefficient"].reshape(-1)
        + checkpoint["intercept"].item()
    ) / temperature
    return float(1 / (1 + np.exp(-np.clip(decision, -60, 60))))


def logit_fuse_presence(base: float, expert: float, weight: float) -> float:
    """Interpolate odds while retaining ranking resolution near 0 and 1."""
    base = float(np.clip(base, 1e-5, 1 - 1e-5))
    expert = float(np.clip(expert, 1e-5, 1 - 1e-5))
    mixed = (
        (1 - weight) * (np.log(base) - np.log1p(-base))
        + weight * (np.log(expert) - np.log1p(-expert))
    )
    return float(1 / (1 + np.exp(-mixed)))


def combine_with_gate(voice: float, music: float, voice_present: float,
                      music_present: float, gate: float = 0.60) -> float:
    active = []
    if voice_present >= gate:
        active.append(voice)
    if music_present >= gate:
        active.append(music)
    if active:
        return float(max(active))
    return float(voice if voice_present >= music_present else music)


def apply_eat_presence_fusion(
    test_dir: Path,
    submission_path: Path,
    eat_dir: Path,
    labels_dir: Path,
    device: str = "cuda",
    voice_weight: float = 0.35,
    music_weight: float = 0.90,
    file_gate: float = 0.60,
    update_file_score: bool = False,
    presence_head_path: Path | None = None,
    music_probe_weight: float = 0.40,
    statistics_output_path: Path | None = None,
    hierarchical_statistics_output_path: Path | None = None,
    hierarchical_checkpoint_path: Path | None = None,
    segmental_statistics_output_path: Path | None = None,
    segmental_checkpoint_path: Path | None = None,
    phone_voice_head_path: Path | None = None,
    telephone_router_path: Path | None = None,
    phone_voice_weight: float = 0.50,
    telephone_ids_output_path: Path | None = None,
) -> None:
    """Rewrite presence ranks and optionally refresh File score.

    ``update_file_score=False`` is the safe competition default: it preserves
    the verified ADS path exactly and prevents hard-gate error propagation.
    The opt-in path remains available for controlled ablations.
    """
    with submission_path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        columns = list(reader.fieldnames or [])
        rows = list(reader)
    required = {
        "ID", "FILE_FAKE_PROB", "VOICE_FAKE_PROB", "MUSIC_FAKE_PROB",
        "VOICE_PRESENT_PROB", "MUSIC_PRESENT_PROB",
    }
    if not required.issubset(columns):
        raise ValueError(f"Submission is missing {sorted(required - set(columns))}")
    for value in (
        voice_weight, music_weight, file_gate, music_probe_weight,
        phone_voice_weight,
    ):
        if not 0 <= value <= 1:
            raise ValueError("fusion weights and gate must lie in [0, 1]")
    if (phone_voice_head_path is None) != (telephone_router_path is None):
        raise ValueError("phone Voice head and telephone router must be provided together")
    if (hierarchical_statistics_output_path is None) != (hierarchical_checkpoint_path is None):
        raise ValueError(
            "hierarchical statistics output and checkpoint must be provided together"
        )
    if hierarchical_statistics_output_path is not None and statistics_output_path is None:
        raise ValueError(
            "hierarchical EAT statistics require the standard statistics output"
        )
    if (segmental_statistics_output_path is None) != (segmental_checkpoint_path is None):
        raise ValueError(
            "segmental statistics output and checkpoint must be provided together"
        )
    if segmental_statistics_output_path is not None:
        if hierarchical_statistics_output_path is None:
            raise ValueError("segmental statistics require hierarchical statistics")
        if statistics_output_path is None:
            raise ValueError("segmental statistics require the standard statistics output")

    audio_files = order_by_submission(find_audio_files(test_dir), rows)
    gc.collect()
    torch.cuda.empty_cache()
    detector = EatPresence(
        eat_dir, labels_dir, device=device,
        presence_head_path=presence_head_path,
    )
    phone_checkpoint = (
        np.load(phone_voice_head_path, allow_pickle=False)
        if phone_voice_head_path is not None else None
    )
    telephone_router = (
        TelephoneRouter(telephone_router_path)
        if telephone_router_path is not None else None
    )
    telephone_count = 0
    telephone_ids = []
    statistic_ids, statistics, statistic_masks = [], [], []
    hierarchical_values = []
    segmental_values, segmental_masks = [], []
    hierarchical_projection = None
    segmental_max_views = None
    if hierarchical_checkpoint_path is not None:
        checkpoint = torch.load(
            hierarchical_checkpoint_path, map_location="cpu", weights_only=False
        )
        if checkpoint.get("model_type") != "hierarchical_eat_music":
            raise ValueError("invalid hierarchical EAT checkpoint")
        hierarchical_projection = torch.from_numpy(
            np.asarray(checkpoint["projection"], dtype=np.float32)
        )
    if segmental_checkpoint_path is not None:
        segmental_checkpoint = torch.load(
            segmental_checkpoint_path, map_location="cpu", weights_only=False
        )
        if segmental_checkpoint.get("model_type") != "segmental_eat_music":
            raise ValueError("invalid segmental EAT checkpoint")
        segmental_projection = np.asarray(
            segmental_checkpoint["projection"], dtype=np.float32
        )
        if not np.array_equal(hierarchical_projection.numpy(), segmental_projection):
            raise ValueError("hierarchical and segmental projections differ")
        segmental_max_views = int(segmental_checkpoint["config"]["max_views"])
    file_batch_size = 4 if segmental_max_views is not None else 8
    for offset in tqdm(
        range(0, len(audio_files), file_batch_size), desc="EAT presence+stats"
    ):
        paths = audio_files[offset:offset + file_batch_size]
        audios = [load_audio(path) for path in paths]
        audio_set = [detector.predict_audio_set(audio) for audio in audios]
        latent = detector.latent_statistics_batch(
            audios, hierarchical_projection=hierarchical_projection,
            segmental_max_views=segmental_max_views,
        )
        for row, path, audio, (eat_voice, eat_music), latent_item in zip(
            rows[offset:offset + file_batch_size], paths, audios, audio_set, latent
        ):
            matrix, mask, probe_music = latent_item[:3]
            voice_present, music_present = fuse_presence(
                float(row["VOICE_PRESENT_PROB"]),
                float(row["MUSIC_PRESENT_PROB"]), eat_voice, eat_music,
                voice_weight=voice_weight, music_weight=music_weight,
            )
            if probe_music is not None:
                music_present = fuse_music_probe(
                    music_present, probe_music, music_probe_weight
                )
            if telephone_router is not None and telephone_router.is_narrowband(audio):
                phone_voice = latent_linear_probability(
                    matrix, mask, phone_checkpoint
                )
                voice_present = logit_fuse_presence(
                    voice_present, phone_voice, phone_voice_weight
                )
                telephone_count += 1
                telephone_ids.append(path.stem)
            if update_file_score:
                row["FILE_FAKE_PROB"] = round(combine_with_gate(
                    float(row["VOICE_FAKE_PROB"]), float(row["MUSIC_FAKE_PROB"]),
                    voice_present, music_present, gate=file_gate,
                ), 10)
            row["VOICE_PRESENT_PROB"] = round(voice_present, 10)
            row["MUSIC_PRESENT_PROB"] = round(music_present, 10)
            if statistics_output_path is not None:
                statistic_ids.append(path.stem)
                statistics.append(matrix)
                statistic_masks.append(mask)
            if hierarchical_statistics_output_path is not None:
                hierarchical_values.append(latent_item[3])
            if segmental_statistics_output_path is not None:
                segmental_values.append(latent_item[4])
                segmental_masks.append(latent_item[5])
    del detector
    gc.collect()
    torch.cuda.empty_cache()

    temporary = submission_path.with_suffix(".tmp.csv")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(submission_path)
    if statistics_output_path is not None:
        statistics_output_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            statistics_output_path,
            ids=np.asarray(statistic_ids),
            statistics=np.stack(statistics),
            view_mask=np.stack(statistic_masks),
            stream=np.asarray("eat"), channel=np.asarray("clean"),
        )
    if hierarchical_statistics_output_path is not None:
        if len(hierarchical_values) != len(statistic_ids):
            raise RuntimeError(
                "hierarchical EAT statistics require the standard statistics output"
            )
        hierarchical_statistics_output_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            hierarchical_statistics_output_path,
            ids=np.asarray(statistic_ids),
            statistics=np.stack(hierarchical_values),
            view_mask=np.stack(statistic_masks),
        )
    if segmental_statistics_output_path is not None:
        if len(segmental_values) != len(statistic_ids):
            raise RuntimeError(
                "segmental EAT statistics require the standard statistics output"
            )
        segmental_statistics_output_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            segmental_statistics_output_path,
            ids=np.asarray(statistic_ids),
            statistics=np.stack(segmental_values),
            view_mask=np.stack(segmental_masks),
        )
    if telephone_router is not None:
        print(f"telephone Voice presence routed {telephone_count}/{len(rows)} files")
        if telephone_ids_output_path is not None:
            telephone_ids_output_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez(
                telephone_ids_output_path,
                ids=np.asarray(telephone_ids),
            )
