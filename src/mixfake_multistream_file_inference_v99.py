"""Apply the frozen three-view MixFake File expert as a fixed soft MoE."""
from __future__ import annotations

import csv
import gc
import os
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

try:
    from .multistream_prompt_spectra import MultiStreamSpectraMultitask
    from .pipeline import find_audio_files, load_audio, order_by_submission
    from .wpt_spectra_inference import (
        _load_spectra, _logit, _preemphasis, _sigmoid, fixed_windows,
    )
except ImportError:  # flat imports inside the submission archive
    from multistream_prompt_spectra import MultiStreamSpectraMultitask
    from pipeline import find_audio_files, load_audio, order_by_submission
    from wpt_spectra_inference import (
        _load_spectra, _logit, _preemphasis, _sigmoid, fixed_windows,
    )


def load_model(model_dir, checkpoint_path, device, expected_task=None):
    checkpoint = torch.load(
        checkpoint_path, map_location="cpu", weights_only=False
    )
    if checkpoint.get("model_type") != "mixfake_multistream_prompt_v94":
        raise ValueError("not a completed MixFake multi-stream checkpoint")
    config = checkpoint.get("config", {})
    if expected_task is not None and config.get("task") != expected_task:
        raise ValueError(
            f"expected a {expected_task!r} checkpoint, got {config.get('task')!r}"
        )
    model = MultiStreamSpectraMultitask(
        _load_spectra(Path(model_dir), device),
        base_tokens=int(config["base_tokens"]),
        frequency_tokens=int(config["frequency_tokens"]),
        texture_tokens=int(config["texture_tokens"]),
        prompt_dropout=float(config["prompt_dropout"]),
        temperature=float(config["temperature"]),
    ).to(device)
    model.load_trainable_state_dict(checkpoint["state"])
    return model.eval(), config


@torch.inference_mode()
def predict_tasks(
    paths, model_dir, checkpoint_path, device="cuda", batch_size=12, views=3,
    expected_task=None,
):
    if batch_size <= 0 or views <= 0:
        raise ValueError("batch size and views must be positive")
    target = torch.device(device)
    model, config = load_model(
        model_dir, checkpoint_path, target, expected_task=expected_task
    )
    window = int(config["window"])
    results = []
    for offset in tqdm(
        range(0, len(paths), batch_size), desc="mixed-audio File expert"
    ):
        selected = paths[offset:offset + batch_size]
        windows = np.stack([
            fixed_windows(load_audio(path), window, views) for path in selected
        ])
        waveform = _preemphasis(torch.from_numpy(windows).to(target))
        with torch.autocast(
            device_type=target.type, dtype=torch.bfloat16,
            enabled=target.type == "cuda",
        ):
            logits = model(waveform)
        results.append(logits.float().sigmoid().cpu().numpy())
    del model
    gc.collect()
    if target.type == "cuda":
        torch.cuda.empty_cache()
    return np.concatenate(results).astype(np.float32)


def predict_file(paths, model_dir, checkpoint_path, device="cuda", batch_size=12, views=3):
    return predict_tasks(
        paths, model_dir, checkpoint_path, device=device,
        batch_size=batch_size, views=views, expected_task="file",
    )[:, 2]


def apply_mixfake_file_moe(
    test_dir, submission_path, model_dir, checkpoint_path,
    *, device="cuda", weight=.40, batch_size=12, views=3,
):
    if not 0 <= weight <= 1:
        raise ValueError("File expert weight must lie in [0, 1]")
    submission_path = Path(submission_path)
    with submission_path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError("empty submission")
    paths = order_by_submission(find_audio_files(Path(test_dir)), rows)
    expert = predict_file(
        paths, model_dir, checkpoint_path,
        device=device, batch_size=batch_size, views=views,
    )
    if len(expert) != len(rows) or not np.isfinite(expert).all():
        raise ValueError("invalid File expert output")
    fieldnames = list(rows[0])
    if fieldnames.count("FILE_FAKE_PROB") != 1:
        raise ValueError("submission must contain one FILE_FAKE_PROB column")
    temporary = submission_path.with_suffix(".mixfake-v99.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row, score in zip(rows, expert):
            current = float(row["FILE_FAKE_PROB"])
            fused = _sigmoid(
                (1 - weight) * _logit(current) + weight * _logit(float(score))
            )
            row["FILE_FAKE_PROB"] = repr(float(fused))
            writer.writerow(row)
    os.replace(temporary, submission_path)


def apply_mixfake_joint_music_moe(
    test_dir, submission_path, model_dir, joint_checkpoint_path,
    music_checkpoint_path, *, device="cuda", file_weight=.40,
    voice_weight=.12, music_weight=.28, batch_size=12, joint_views=3,
    music_views=1, file_or_weight=.15,
):
    """Fuse File/Voice from one joint pass and Music from an exact-codec pass."""
    weights = {
        "FILE_FAKE_PROB": float(file_weight),
        "VOICE_FAKE_PROB": float(voice_weight),
        "MUSIC_FAKE_PROB": float(music_weight),
    }
    if any(not 0 <= value <= 1 for value in (*weights.values(), file_or_weight)):
        raise ValueError("all expert weights must lie in [0, 1]")
    submission_path = Path(submission_path)
    with submission_path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError("empty submission")
    paths = order_by_submission(find_audio_files(Path(test_dir)), rows)
    joint = predict_tasks(
        paths, model_dir, joint_checkpoint_path, device=device,
        batch_size=batch_size, views=joint_views, expected_task="all",
    )
    music = predict_tasks(
        paths, model_dir, music_checkpoint_path, device=device,
        batch_size=batch_size, views=music_views, expected_task="music",
    )[:, 1]
    if joint.shape != (len(rows), 3) or music.shape != (len(rows),):
        raise ValueError("invalid expert output shape")
    if not np.isfinite(joint).all() or not np.isfinite(music).all():
        raise ValueError("nonfinite expert output")
    fieldnames = list(rows[0])
    if any(fieldnames.count(name) != 1 for name in weights):
        raise ValueError("submission authenticity columns are invalid")
    experts = {
        "VOICE_FAKE_PROB": joint[:, 0],
        "MUSIC_FAKE_PROB": music,
        "FILE_FAKE_PROB": joint[:, 2],
    }
    temporary = submission_path.with_suffix(".mixfake-v101.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for index, row in enumerate(rows):
            for name, weight in weights.items():
                current = float(row[name])
                fused = _sigmoid(
                    (1 - weight) * _logit(current)
                    + weight * _logit(float(experts[name][index]))
                )
                row[name] = repr(float(fused))
            # The ground-truth definition is FILE = VOICE OR MUSIC.  Retain
            # the independently trained File detector, but softly reconcile
            # it with the already-fused component probabilities.
            voice = float(row["VOICE_FAKE_PROB"])
            music_score = float(row["MUSIC_FAKE_PROB"])
            component_or = 1 - (1 - voice) * (1 - music_score)
            row["FILE_FAKE_PROB"] = repr(float(_sigmoid(
                (1 - file_or_weight) * _logit(float(row["FILE_FAKE_PROB"]))
                + file_or_weight * _logit(component_or)
            )))
            writer.writerow(row)
    os.replace(temporary, submission_path)
