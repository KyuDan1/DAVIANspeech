#!/usr/bin/env python3
"""Train the separation-free EAT-large to AASIST v56 expert.

Only ``train`` manifests from ``configs/data_partitions.yaml`` update weights.
Checkpoint selection uses an identity/generator-disjoint fold derived from that
role.  The complete ``development`` role is scored only after the checkpoint
has been selected.
"""

from __future__ import annotations

import argparse
import copy
import json
import random
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import transformers  # import before the local timm compatibility shim
import librosa
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from eat_large_aasist import EatLargeAASISTExpert, multitask_loss  # noqa: E402
from eat_large_aasist_data import (  # noqa: E402
    crop_targets, music_generator_group, partition_paths, split_plan,
)
from eat_large_aasist_inference import (  # noqa: E402
    SAMPLE_RATE, SAMPLES, _load_local_model, audio_views, crop_or_pad, fbank,
    sha256,
)
from evaluate_diagnostic import score_frame  # noqa: E402


AUDIO_EXTENSIONS = {".aac", ".flac", ".m4a", ".mp3", ".ogg", ".opus", ".wav", ".wma"}


def load_audio(path: Path) -> np.ndarray:
    audio, _ = librosa.load(path, sr=SAMPLE_RATE, mono=True, dtype=np.float32)
    if not len(audio) or not np.isfinite(audio).all():
        raise ValueError(f"invalid audio: {path}")
    return audio


PROBABILITY_COLUMNS = (
    "VOICE_FAKE_PROB", "MUSIC_FAKE_PROB", "FILE_FAKE_PROB",
)


def _audio_index(directory: Path) -> dict[str, Path]:
    result = {
        path.stem: path for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in AUDIO_EXTENSIONS
    }
    if not result:
        raise FileNotFoundError(f"no audio in {directory}")
    if len(result) != len(set(result)):
        raise ValueError(f"duplicate audio stems in {directory}")
    return result


def load_manifest(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, dtype={"ID": str})
    required = {
        "ID", "FILE_FAKE", "VOICE_FAKE", "MUSIC_FAKE",
        "VOICE_PRESENT", "MUSIC_PRESENT",
    }
    if missing := required.difference(frame):
        raise ValueError(f"{path} misses columns {sorted(missing)}")
    if frame.ID.isna().any() or frame.ID.duplicated().any():
        raise ValueError(f"invalid IDs in {path}")
    index = _audio_index(path.parent / "audio")
    missing_audio = set(frame.ID).difference(index)
    if missing_audio:
        raise FileNotFoundError(
            f"{path.parent.name}: {len(missing_audio)} rows have no audio"
        )
    frame = frame.copy()
    frame["PATH"] = frame.ID.map(index)
    frame["DATASET"] = path.parent.name
    frame["MANIFEST"] = str(path.relative_to(ROOT))
    return frame.reset_index(drop=True)


def load_role(config_path: Path, role: str) -> pd.DataFrame:
    frames = [load_manifest(path) for path in partition_paths(config_path, role)]
    result = pd.concat(frames, ignore_index=True, sort=False)
    keys = result[["DATASET", "ID"]].astype(str)
    if keys.duplicated().any():
        raise ValueError(f"duplicate dataset/ID rows in {role}")
    return result


def _random_start(length: int) -> int:
    return 0 if length <= SAMPLES else random.randrange(length - SAMPLES + 1)


def channel_augment(audio: np.ndarray) -> np.ndarray:
    """Cheap, label-symmetric codec/telephone stress for training only."""
    audio = np.asarray(audio, dtype=np.float32)
    mode = random.randrange(4)
    if mode == 0:  # companding proxy for G.711
        mu = 255.0
        peak = max(float(np.abs(audio).max()), 1e-5)
        normal = np.clip(audio / peak, -1, 1)
        encoded = np.sign(normal) * np.log1p(mu * np.abs(normal)) / np.log1p(mu)
        encoded = np.round((encoded + 1) * 127.5) / 127.5 - 1
        audio = np.sign(encoded) * np.expm1(np.abs(encoded) * np.log1p(mu)) / mu
        audio *= peak
    elif mode == 1:  # deterministic FFT telephone band
        spectrum = np.fft.rfft(audio)
        frequency = np.fft.rfftfreq(len(audio), 1 / SAMPLE_RATE)
        spectrum[(frequency < 280) | (frequency > 3_600)] = 0
        audio = np.fft.irfft(spectrum, n=len(audio)).astype(np.float32)
    elif mode == 2:  # narrowband down/up-sampling proxy
        low_length = max(2, len(audio) // 2)
        source = np.linspace(0, 1, len(audio), endpoint=False)
        low_axis = np.linspace(0, 1, low_length, endpoint=False)
        low = np.interp(low_axis, source, audio)
        audio = np.interp(source, low_axis, low).astype(np.float32)
    else:  # low-level channel noise
        snr = random.uniform(20, 40)
        rms = float(np.sqrt(np.mean(np.square(audio, dtype=np.float64))))
        if rms > 0:
            noise = np.random.standard_normal(len(audio)).astype(np.float32)
            noise *= rms / (10 ** (snr / 20) * max(float(noise.std()), 1e-6))
            audio = audio + noise
    gain = 10 ** (random.uniform(-3, 3) / 20)
    return np.clip(audio * gain, -1, 1).astype(np.float32)


class TrainDataset(Dataset):
    def __init__(self, frame: pd.DataFrame, augmentation_probability: float) -> None:
        self.frame = frame.reset_index(drop=True)
        self.augmentation_probability = float(augmentation_probability)

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int):
        row = self.frame.iloc[index]
        audio = load_audio(Path(row.PATH))
        start = _random_start(len(audio))
        view = crop_or_pad(audio, start)
        if random.random() < self.augmentation_probability:
            view = channel_augment(view)
        start_seconds = start / SAMPLE_RATE
        target, presence = crop_targets(
            row, start_seconds, start_seconds + SAMPLES / SAMPLE_RATE
        )
        return fbank(view), target, presence, np.int64(index)


class EvalDataset(Dataset):
    def __init__(self, frame: pd.DataFrame, maximum_views: int) -> None:
        self.frame = frame.reset_index(drop=True)
        self.maximum_views = int(maximum_views)

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int):
        audio = load_audio(Path(self.frame.iloc[index].PATH))
        if self.maximum_views == 1:
            start = max(0, (len(audio) - SAMPLES) // 2)
            values = fbank(crop_or_pad(audio, start))[None]
            mask = torch.ones(1, dtype=torch.bool)
        else:
            values, mask = audio_views(audio, self.maximum_views)
        return values, mask


def balanced_weights(frame: pd.DataFrame) -> np.ndarray:
    keys = []
    for _, row in frame.iterrows():
        voice_fake = int(float(row.VOICE_FAKE)) if pd.notna(row.VOICE_FAKE) else 0
        music_fake = int(float(row.MUSIC_FAKE)) if pd.notna(row.MUSIC_FAKE) else 0
        keys.append((
            str(row.DATASET),
            int(float(row.VOICE_PRESENT)), int(float(row.MUSIC_PRESENT)),
            voice_fake, music_fake,
            str(row.get("MIX_MODE", row.get("CONDITION", "unknown"))),
            str(row.get("CHANNEL", row.get("CODEC", "clean"))),
            music_generator_group(row),
        ))
    counts = Counter(keys)
    weight = np.asarray([1 / counts[key] for key in keys], dtype=np.float64)
    datasets = frame.DATASET.astype(str).to_numpy()
    for dataset in np.unique(datasets):
        selected = datasets == dataset
        weight[selected] /= weight[selected].sum()
    return (weight / weight.mean()).astype(np.float32)


@torch.inference_mode()
def predict(
    model: EatLargeAASISTExpert, frame: pd.DataFrame, device: torch.device,
    batch_size: int, workers: int, maximum_views: int,
) -> np.ndarray:
    loader = DataLoader(
        EvalDataset(frame, maximum_views), batch_size=batch_size,
        shuffle=False, num_workers=workers, pin_memory=device.type == "cuda",
        persistent_workers=workers > 0,
    )
    model.eval()
    output = []
    for features, mask in loader:
        features = features[:, :, None].to(device, non_blocking=True)
        mask = mask.to(device, non_blocking=True)
        with torch.autocast(
            device_type=device.type, dtype=torch.bfloat16,
            enabled=device.type == "cuda",
        ):
            logits = model(features, mask)
            probability = model.backend.probabilities(logits)
        output.append(probability.float().cpu().numpy())
    return np.concatenate(output)


def metrics_for(frame: pd.DataFrame, probability: np.ndarray) -> dict[str, float]:
    prediction = pd.DataFrame({
        column: probability[:, index]
        for index, column in enumerate(PROBABILITY_COLUMNS)
    }, index=frame.ID)
    return score_frame(frame.set_index("ID").join(prediction))


def _state(model: EatLargeAASISTExpert) -> dict[str, torch.Tensor]:
    return {
        name: value.detach().cpu().clone()
        for name, value in model.state_dict().items()
        if not name.startswith("eat.")
    }


def _load_small_state(
    model: EatLargeAASISTExpert, state: dict[str, torch.Tensor],
) -> None:
    incompatible = model.load_state_dict(state, strict=False)
    if incompatible.unexpected_keys or any(
        not name.startswith("eat.") for name in incompatible.missing_keys
    ):
        raise RuntimeError(f"incompatible selected state: {incompatible}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--partition-config", type=Path,
        default=ROOT / "configs/data_partitions.yaml",
    )
    parser.add_argument(
        "--model-dir", type=Path, default=ROOT / "models/eat-large-as2m-v56"
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--capture-layers", nargs="+", type=int,
                        default=[3, 7, 11, 15, 19, 20, 21, 22, 23])
    parser.add_argument("--adapter-blocks", nargs="+", type=int,
                        default=[20, 21, 22, 23])
    parser.add_argument("--bottleneck", type=int, default=32)
    parser.add_argument("--graph-width", type=int, default=64)
    parser.add_argument("--graph-hidden", type=int, default=32)
    parser.add_argument("--branches", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=.30)
    parser.add_argument("--temperature", type=float, default=5.)
    parser.add_argument("--file-component-weight", type=float, default=.50)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--samples-per-epoch", type=int, default=12_000)
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--eval-batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--ranking-weight", type=float, default=.10)
    parser.add_argument("--component-or-weight", type=float, default=.05)
    parser.add_argument("--augmentation-probability", type=float, default=.45)
    parser.add_argument("--folds", type=int, default=3)
    parser.add_argument("--held-out-fold", type=int, default=0)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260904)
    parser.add_argument("--final-dev-views", type=int, default=3)
    parser.add_argument("--max-train-examples", type=int)
    parser.add_argument("--max-validation-examples", type=int)
    parser.add_argument("--skip-final-development", action="store_true")
    args = parser.parse_args()
    if args.output_dir.exists():
        parser.error(f"refusing to overwrite {args.output_dir}")
    if not 0 <= args.augmentation_probability <= 1:
        parser.error("augmentation probability must lie in [0,1]")
    if args.final_dev_views not in (1, 2, 3):
        parser.error("final dev views must be 1, 2 or 3")
    args.output_dir.mkdir(parents=True)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    full_train = load_role(args.partition_config, "train")
    plan = split_plan(
        full_train, args.folds, args.held_out_fold, args.seed
    )
    train = full_train.iloc[plan.train_index].reset_index(drop=True)
    music_validation = full_train.iloc[
        plan.music_validation_index
    ].reset_index(drop=True)
    voice_validation = full_train.iloc[
        plan.voice_validation_index
    ].reset_index(drop=True)
    if args.max_train_examples:
        train = train.sample(
            min(args.max_train_examples, len(train)), random_state=args.seed
        ).reset_index(drop=True)
    if args.max_validation_examples:
        music_validation = music_validation.sample(
            min(args.max_validation_examples, len(music_validation)),
            random_state=args.seed
        ).reset_index(drop=True)
        voice_validation = voice_validation.sample(
            min(args.max_validation_examples, len(voice_validation)),
            random_state=args.seed + 1
        ).reset_index(drop=True)

    split_audit = dict(plan.audit)
    split_audit.update({
        "all_train_rows": len(full_train), "optimization_rows": len(train),
        "music_selection_rows_used": len(music_validation),
        "voice_selection_rows_used": len(voice_validation),
    })
    print(json.dumps({"split": split_audit}), flush=True)

    device = torch.device(args.device)
    eat = _load_local_model(args.model_dir, device)
    config = {
        "capture_layers": tuple(args.capture_layers),
        "adapter_blocks": tuple(args.adapter_blocks),
        "bottleneck": args.bottleneck, "graph_width": args.graph_width,
        "graph_hidden": args.graph_hidden, "branches": args.branches,
        "dropout": args.dropout, "temperature": args.temperature,
        "file_component_weight": args.file_component_weight,
    }
    model = EatLargeAASISTExpert(eat, **config).to(device)
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable, lr=args.learning_rate, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=args.learning_rate,
        total_steps=args.epochs * max(1, args.samples_per_epoch // args.batch_size),
        pct_start=.10,
    )
    weight = balanced_weights(train)
    sampler = WeightedRandomSampler(
        weight, num_samples=args.samples_per_epoch, replacement=True,
        generator=torch.Generator().manual_seed(args.seed),
    )
    loader = DataLoader(
        TrainDataset(train, args.augmentation_probability),
        batch_size=args.batch_size, sampler=sampler, num_workers=args.workers,
        pin_memory=device.type == "cuda", persistent_workers=args.workers > 0,
        drop_last=True,
    )

    best_selection, best_epoch, best_state, stale = -np.inf, -1, None, 0
    history = []
    start_time = time.time()
    for epoch in range(args.epochs):
        model.train()
        losses = []
        for features, target, presence, index in loader:
            features = features[:, None].to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            presence = presence.to(device, non_blocking=True)
            # The replacement sampler already applies the generator/cell
            # weights.  Reapplying them in the loss would square the correction.
            sample_weight = torch.ones(len(index), device=device)
            with torch.autocast(
                device_type=device.type, dtype=torch.bfloat16,
                enabled=device.type == "cuda",
            ):
                logits = model(features)
                loss, _ = multitask_loss(
                    model, logits, target, presence, sample_weight,
                    ranking_weight=args.ranking_weight,
                    component_or_weight=args.component_or_weight,
                )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 5.)
            optimizer.step()
            scheduler.step()
            losses.append(float(loss.detach()))

        music_probability = predict(
            model, music_validation, device, args.eval_batch_size,
            args.workers, maximum_views=1,
        )
        voice_probability = predict(
            model, voice_validation, device, args.eval_batch_size,
            args.workers, maximum_views=1,
        )
        music_metrics = metrics_for(music_validation, music_probability)
        voice_metrics = metrics_for(voice_validation, voice_probability)
        music_quality = .625 * (1 - music_metrics["FILE_EER"]) + .375 * (
            1 - music_metrics["MUSIC_EER"]
        )
        voice_quality = .625 * (1 - voice_metrics["FILE_EER"]) + .375 * (
            1 - voice_metrics["VOICE_EER"]
        )
        # File must survive both independently held-out component views.
        selection = .5 * min(music_quality, voice_quality) + .25 * (
            music_quality + voice_quality
        )
        row = {
            "epoch": epoch, "loss": float(np.mean(losses)),
            "selection": float(selection),
            "music_holdout_file_eer": float(music_metrics["FILE_EER"]),
            "music_holdout_music_eer": float(music_metrics["MUSIC_EER"]),
            "voice_holdout_file_eer": float(voice_metrics["FILE_EER"]),
            "voice_holdout_voice_eer": float(voice_metrics["VOICE_EER"]),
            "elapsed_seconds": time.time() - start_time,
        }
        history.append(row)
        print(json.dumps(row), flush=True)
        partial = args.output_dir / "history.partial.csv"
        temporary = args.output_dir / "history.partial.tmp.csv"
        pd.DataFrame(history).to_csv(temporary, index=False)
        temporary.replace(partial)
        if selection > best_selection + 1e-5:
            best_selection, best_epoch, best_state, stale = (
                float(selection), epoch, _state(model), 0
            )
        else:
            stale += 1
            if stale >= args.patience:
                break
    if best_state is None:
        raise RuntimeError("training did not produce a checkpoint")
    _load_small_state(model, best_state)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "model_type": "eat_large_aasist_v56",
        "model": best_state, "config": config,
        "base_eat_sha256": sha256(args.model_dir / "model.safetensors"),
        "base_eat_revision": "53b1251196e3dabb991dfa7629fa89394c7f8335",
        "partition_config": str(args.partition_config.relative_to(ROOT)),
        "best_epoch": best_epoch, "selection": best_selection,
        "seed": args.seed, "split_audit": split_audit,
        "train_manifests": [str(path.relative_to(ROOT)) for path in partition_paths(
            args.partition_config, "train"
        )],
    }
    torch.save(checkpoint, args.output_dir / "eat_large_aasist_v56.pt")
    pd.DataFrame(history).to_csv(args.output_dir / "history.csv", index=False)
    (args.output_dir / "split_audit.json").write_text(
        json.dumps(split_audit, indent=2) + "\n", encoding="utf-8"
    )

    if not args.skip_final_development:
        development = load_role(args.partition_config, "development")
        predictions, metric_rows = [], []
        for dataset, frame in development.groupby("DATASET", sort=False):
            frame = frame.reset_index(drop=True)
            probability = predict(
                model, frame, device, args.eval_batch_size,
                args.workers, maximum_views=args.final_dev_views,
            )
            metric_rows.append({"DATASET": dataset, **metrics_for(frame, probability)})
            current = pd.DataFrame({
                "DATASET": dataset, "ID": frame.ID,
                **{
                    column: probability[:, index]
                    for index, column in enumerate(PROBABILITY_COLUMNS)
                },
            })
            predictions.append(current)
            print(json.dumps(metric_rows[-1]), flush=True)
        pd.concat(predictions, ignore_index=True).to_csv(
            args.output_dir / "development_predictions.csv", index=False
        )
        pd.DataFrame(metric_rows).to_csv(
            args.output_dir / "development_metrics.csv", index=False
        )

    summary = {
        "best_epoch": best_epoch, "selection": best_selection,
        "trainable_parameters": sum(parameter.numel() for parameter in trainable),
        "base_parameters": sum(parameter.numel() for parameter in model.eat.parameters()),
        "elapsed_seconds": time.time() - start_time,
        **split_audit,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
