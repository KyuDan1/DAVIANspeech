#!/usr/bin/env python3
"""Train one separation-free multi-stream model for all three ADS tasks.

The model shares a single original-mixture XLS-R/Spectra forward pass.  Its
objective follows the competition ADS weights directly: Voice .2, Music .3,
and File .5.  Clean and content-matched channel views receive the same labels
and a small logit-consistency penalty.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import random
import sys
import time

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from data_guard import assert_no_locked_eval_leakage  # noqa: E402
from multistream_prompt_spectra import (  # noqa: E402
    MultiStreamSpectraMultitask, warm_start_shared_backend,
)
from train_mixfake_multistream_prompt_v94 import (  # noqa: E402
    PairedMixtures, finite_eer, preemphasis, read_manifest, sample_weights,
    sha256_file, task_rank, worker_seed, load_spectra,
)


ADS_WEIGHTS = np.asarray([.2, .3, .5], np.float64)


@torch.inference_mode()
def evaluate(model, loader, device, *, use_changed=False):
    model.eval()
    scores, indices = [], []
    for clean, changed, _target, index in loader:
        waveform = changed if use_changed else clean
        waveform = preemphasis(waveform.to(device, non_blocking=True))
        with torch.autocast(
            device_type=device.type, dtype=torch.bfloat16,
            enabled=device.type == "cuda",
        ):
            logits = model(waveform)
        scores.append(logits.float().sigmoid().cpu().numpy())
        indices.extend(index.tolist())
    scores = np.concatenate(scores).astype(np.float32)
    labels = loader.dataset.frame.iloc[indices][
        ["VOICE_FAKE", "MUSIC_FAKE", "FILE_FAKE"]
    ].to_numpy(np.int64)
    eers = np.asarray([
        finite_eer(labels[:, task], scores[:, task]) for task in range(3)
    ])
    return float(np.dot(ADS_WEIGHTS, eers)), eers, scores


def weighted_task_loss(clean, phone, target, ranking_weight, consistency_weight):
    weights = clean.new_tensor(ADS_WEIGHTS)
    bce = clean.new_zeros(())
    rank = clean.new_zeros(())
    consistency = clean.new_zeros(())
    for task, weight in enumerate(weights):
        bce = bce + weight * .5 * (
            F.binary_cross_entropy_with_logits(clean[:, task], target[:, task])
            + F.binary_cross_entropy_with_logits(phone[:, task], target[:, task])
        )
        rank = rank + weight * .5 * (
            task_rank(clean[:, task], target[:, task])
            + task_rank(phone[:, task], target[:, task])
        )
        consistency = consistency + weight * F.smooth_l1_loss(
            phone[:, task], clean[:, task].detach()
        )
    return bce + ranking_weight * rank + consistency_weight * consistency


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-manifest", type=Path, required=True)
    parser.add_argument("--development-manifest", type=Path, required=True)
    parser.add_argument("--audio-root", type=Path, required=True)
    parser.add_argument("--paired-audio-root", type=Path, default=None)
    parser.add_argument("--model-dir", type=Path,
                        default=ROOT / "models/external/spectra_aasist")
    parser.add_argument("--warm-start", type=Path, default=(
        ROOT / "v50_v57m_challenger_v2/model/spectra-aasist/"
        "wpt_spectra_multitask.pt"
    ))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=20260934)
    parser.add_argument("--window", type=int, default=64600)
    parser.add_argument("--views", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--eval-batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--samples-per-epoch", type=int, default=4096)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--prompt-lr", type=float, default=2e-3)
    parser.add_argument("--head-lr", type=float, default=5e-4)
    parser.add_argument("--backend-lr", type=float, default=5e-5)
    parser.add_argument("--ranking-weight", type=float, default=.10)
    parser.add_argument("--consistency-weight", type=float, default=.10)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if "locked" in " ".join(map(str, (
        args.train_manifest, args.development_manifest, args.output,
    ))).lower():
        raise ValueError("locked data are forbidden during training")
    assert_no_locked_eval_leakage(
        args.train_manifest, ROOT / "configs/data_partitions.yaml"
    )
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    train = read_manifest(args.train_manifest, args.audio_root)
    development = read_manifest(args.development_manifest, args.audio_root)
    if args.paired_audio_root is not None:
        for frame, split in ((train, "train"), (development, "development")):
            directory = args.paired_audio_root / split
            paths = frame.ID.map(lambda value: directory / f"{value}.wav")
            missing = [str(path) for path in paths if not path.is_file()]
            if missing:
                raise FileNotFoundError(missing[0])
            frame["PHONE_PATH"] = paths.map(str)
    if set(train.ID) & set(development.ID):
        raise ValueError("train/development ID overlap")

    train_data = PairedMixtures(train, args.window, args.views, True, True)
    dev_clean = PairedMixtures(development, args.window, args.views, False, False)
    dev_phone = PairedMixtures(development, args.window, args.views, False, True)
    sampler = WeightedRandomSampler(
        sample_weights({
            "component_case": train.COMPONENT_CASE.to_numpy(),
            "music_generator": train.MUSIC_GENERATOR.to_numpy(),
        }),
        args.samples_per_epoch, replacement=True,
        generator=torch.Generator().manual_seed(args.seed),
    )
    common = dict(
        num_workers=args.workers, pin_memory=True,
        worker_init_fn=worker_seed, persistent_workers=args.workers > 0,
    )
    train_loader = DataLoader(
        train_data, batch_size=args.batch_size, sampler=sampler, **common
    )
    clean_loader = DataLoader(
        dev_clean, batch_size=args.eval_batch_size, shuffle=False, **common
    )
    phone_loader = DataLoader(
        dev_phone, batch_size=args.eval_batch_size, shuffle=False, **common
    )

    device = torch.device(args.device)
    model = MultiStreamSpectraMultitask(load_spectra(args.model_dir, device)).to(device)
    warm = torch.load(args.warm_start, map_location="cpu", weights_only=False)
    if warm.get("model_type") != "wpt_spectra_multitask":
        raise ValueError("joint v97 warm-start must be the current WPT checkpoint")
    warm_report = warm_start_shared_backend(model, warm["state"])
    prompts = list(model.prompt_encoder.prompts.parameters())
    head = list(model.task_head.parameters())
    excluded = {id(value) for value in prompts + head}
    backend = [value for value in model.parameters()
               if value.requires_grad and id(value) not in excluded]
    optimizer = torch.optim.AdamW([
        {"params": prompts, "lr": args.prompt_lr},
        {"params": head, "lr": args.head_lr},
        {"params": backend, "lr": args.backend_lr},
    ], weight_decay=5e-4)

    best, best_state, stale, history = float("inf"), None, 0, []
    started = time.monotonic()
    for epoch in range(args.epochs):
        model.train(); losses = []
        for step, (clean, phone, target, _) in enumerate(train_loader):
            target = target.to(device, non_blocking=True)
            clean = preemphasis(clean.to(device, non_blocking=True))
            phone = preemphasis(phone.to(device, non_blocking=True))
            with torch.autocast(
                device_type=device.type, dtype=torch.bfloat16,
                enabled=device.type == "cuda",
            ):
                clean_logit = model(clean)
                phone_logit = model(phone)
                loss = weighted_task_loss(
                    clean_logit, phone_logit, target,
                    args.ranking_weight, args.consistency_weight,
                )
            optimizer.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [value for value in model.parameters() if value.requires_grad],
                3., error_if_nonfinite=True,
            )
            optimizer.step(); losses.append(float(loss.detach()))
            if step and step % 100 == 0:
                print(json.dumps({"epoch": epoch, "step": step,
                                  "loss": float(np.mean(losses[-100:]))}), flush=True)
        clean_error, clean_eers, clean_scores = evaluate(model, clean_loader, device)
        phone_error, phone_eers, phone_scores = evaluate(
            model, phone_loader, device, use_changed=True
        )
        selection = .65 * (clean_error + phone_error) / 2 + .35 * max(
            clean_error, phone_error
        )
        record = {
            "epoch": epoch, "loss": float(np.mean(losses)),
            "clean_error": clean_error, "phone_error": phone_error,
            "clean_voice_eer": float(clean_eers[0]),
            "clean_music_eer": float(clean_eers[1]),
            "clean_file_eer": float(clean_eers[2]),
            "phone_voice_eer": float(phone_eers[0]),
            "phone_music_eer": float(phone_eers[1]),
            "phone_file_eer": float(phone_eers[2]),
            "selection_error": selection, "seconds": time.monotonic() - started,
        }
        history.append(record); print(json.dumps(record), flush=True)
        if selection < best - 1e-5:
            best, stale = selection, 0
            best_state = copy.deepcopy(model.trainable_state_dict())
            best_scores = {"clean": clean_scores, "phone": phone_scores}
            best_epoch = epoch
        else:
            stale += 1
            if stale >= args.patience:
                break
    if best_state is None:
        raise RuntimeError("no checkpoint selected")
    args.output.mkdir(parents=True)
    checkpoint = {
        "model_type": "mixfake_multistream_prompt_v94",
        "state": best_state,
        "config": {
            "temperature": model.temperature, "window": args.window,
            "views": args.views, "task": "all",
            "base_tokens": model.prompt_encoder.prompts.base_tokens,
            "frequency_tokens": model.prompt_encoder.prompts.frequency_tokens,
            "texture_tokens": model.prompt_encoder.prompts.texture_tokens,
            "prompt_dropout": model.prompt_encoder.prompts.dropout.p,
        },
        "best_epoch": best_epoch, "selection_error": best,
        "seed": args.seed, "warm_start_report": warm_report,
        "provenance": {
            "train_manifest": str(args.train_manifest),
            "train_manifest_sha256": sha256_file(args.train_manifest),
            "development_manifest": str(args.development_manifest),
            "development_manifest_sha256": sha256_file(args.development_manifest),
            "warm_start_sha256": sha256_file(args.warm_start),
            "locked_scores_read": False,
        },
    }
    torch.save(checkpoint, args.output / "multistream_prompt.pt")
    pd.DataFrame(history).to_csv(args.output / "history.csv", index=False)
    np.savez_compressed(
        args.output / "development_predictions.npz",
        ids=development.ID.to_numpy(dtype=str), **best_scores,
    )
    summary = {
        "best_epoch": best_epoch, "selection_error": best,
        "elapsed_seconds": time.monotonic() - started,
        "clean_eers": dict(zip(("voice", "music", "file"), [
            finite_eer(development.VOICE_FAKE, best_scores["clean"][:, 0]),
            finite_eer(development.MUSIC_FAKE, best_scores["clean"][:, 1]),
            finite_eer(development.FILE_FAKE, best_scores["clean"][:, 2]),
        ])),
        "phone_eers": dict(zip(("voice", "music", "file"), [
            finite_eer(development.VOICE_FAKE, best_scores["phone"][:, 0]),
            finite_eer(development.MUSIC_FAKE, best_scores["phone"][:, 1]),
            finite_eer(development.FILE_FAKE, best_scores["phone"][:, 2]),
        ])),
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({"status": "complete", "best_epoch": best_epoch,
                      "selection_error": best, "output": str(args.output)}), flush=True)


if __name__ == "__main__":
    main()
