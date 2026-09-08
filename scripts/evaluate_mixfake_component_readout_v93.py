#!/usr/bin/env python3
"""Compare v93 WPT readouts and fixed task-wise residual ensembles on development."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from mixfake_component_readout_v93 import ComponentReadoutV93, lme  # noqa: E402
from train_mixfake_component_readout_v93 import (  # noqa: E402
    discover, eer, load_shards, metrics, sha256_file,
)


def logit(probability: np.ndarray) -> np.ndarray:
    values = np.clip(np.asarray(probability, np.float64), 1e-6, 1 - 1e-6)
    return np.log(values) - np.log1p(-values)


def sigmoid(value: np.ndarray) -> np.ndarray:
    return np.exp(-np.logaddexp(0, -np.asarray(value, np.float64)))


def load_model(path: Path, device: torch.device) -> ComponentReadoutV93:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if checkpoint.get("model_type") != "mixfake_component_readout_v93":
        raise ValueError(f"unexpected checkpoint: {path}")
    model = ComponentReadoutV93(**checkpoint["config"]).to(device)
    model.load_state_dict(checkpoint["state"], strict=True)
    return model.eval()


@torch.inference_mode()
def logits_and_residual(
    model: ComponentReadoutV93, cache: dict[str, np.ndarray],
    device: torch.device, batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    outputs, residuals = [], []
    for start in range(0, len(cache["ids"]), batch_size):
        stop = start + batch_size
        embeddings = torch.from_numpy(
            cache["embeddings"][start:stop].astype(np.float32)
        ).to(device)
        base_logits = torch.from_numpy(
            cache["base_logits"][start:stop].astype(np.float32)
        ).to(device)
        output, residual = model(embeddings, base_logits)
        outputs.append(output.cpu().numpy())
        residuals.append(residual.cpu().numpy())
    return np.concatenate(outputs), np.concatenate(residuals)


def base_logits(cache: dict[str, np.ndarray]) -> np.ndarray:
    tensor = torch.from_numpy(cache["base_logits"].astype(np.float32))
    return lme(tensor, 5.).numpy()


def objective(channel_eers: list[float]) -> float:
    quality = 1 - np.asarray(channel_eers, np.float64)
    return float(.65 * quality.mean() + .35 * quality.min())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--checkpoints", type=Path, nargs="+", required=True)
    parser.add_argument("--channels", nargs="+", default=[
        "clean", "paired_fast", "g711_ulaw", "g722_wb", "opus_nb_8k",
        "transcode_g711_opus",
    ])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=1024)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if "locked" in " ".join(map(str, [args.cache_root, args.output])).lower():
        raise ValueError("locked caches are forbidden during selection")
    device = torch.device(args.device)
    caches = {
        channel: load_shards(discover(args.cache_root, "development", channel))
        for channel in args.channels
    }
    reference_ids = next(iter(caches.values()))["ids"].astype(str)
    for channel, cache in caches.items():
        if not np.array_equal(cache["ids"].astype(str), reference_ids):
            raise ValueError(f"development cache ID mismatch: {channel}")
    bases = {channel: base_logits(cache) for channel, cache in caches.items()}
    expert_residuals: dict[str, dict[str, np.ndarray]] = {}
    checkpoint_meta = []
    for path in args.checkpoints:
        name = path.parent.name
        model = load_model(path, device)
        expert_residuals[name] = {}
        for channel, cache in caches.items():
            _, residual = logits_and_residual(model, cache, device, args.batch_size)
            expert_residuals[name][channel] = residual
        checkpoint_meta.append({"name": name, "path": str(path), "sha256": sha256_file(path)})

    families = {name: residual for name, residual in expert_residuals.items()}
    families["uniform_ensemble"] = {
        channel: np.mean([
            residual[channel] for residual in expert_residuals.values()
        ], axis=0)
        for channel in args.channels
    }
    targets = {
        channel: np.stack((cache["voice_fake"], cache["music_fake"], cache["file_fake"]), axis=1)
        for channel, cache in caches.items()
    }
    report = {
        "schema": "mixfake_component_readout_v93_development_comparison",
        "channels": args.channels, "checkpoints": checkpoint_meta,
        "locked_scores_read": False, "families": {},
    }
    weights = np.linspace(0, 1, 21)
    baseline_probabilities = {channel: sigmoid(values) for channel, values in bases.items()}
    report["baseline"] = {
        channel: metrics(caches[channel], probability)
        for channel, probability in baseline_probabilities.items()
    }
    for family, residual_by_channel in families.items():
        selected_weights = []
        task_sweeps = {}
        for task, task_name in enumerate(("voice", "music", "file")):
            entries = []
            for weight in weights:
                channel_eers = [
                    eer(
                        targets[channel][:, task],
                        sigmoid(bases[channel][:, task]
                                + weight * residual_by_channel[channel][:, task]),
                    )
                    for channel in args.channels
                ]
                entries.append({
                    "weight": float(weight), "objective": objective(channel_eers),
                    "mean_eer": float(np.mean(channel_eers)),
                    "worst_eer": float(np.max(channel_eers)),
                    "channel_eers": dict(zip(args.channels, channel_eers)),
                })
            best = max(entries, key=lambda row: (row["objective"], -row["weight"]))
            selected_weights.append(best["weight"])
            task_sweeps[task_name] = entries
        probabilities = {
            channel: sigmoid(
                bases[channel]
                + residual_by_channel[channel] * np.asarray(selected_weights)[None]
            )
            for channel in args.channels
        }
        channel_metrics = {
            channel: metrics(caches[channel], probability)
            for channel, probability in probabilities.items()
        }
        ads_values = [value["ads"] for value in channel_metrics.values()]
        report["families"][family] = {
            "selected_task_weights": dict(zip(("voice", "music", "file"), selected_weights)),
            "development_ads_objective": float(.65 * np.mean(ads_values) + .35 * np.min(ads_values)),
            "mean_ads": float(np.mean(ads_values)), "worst_ads": float(np.min(ads_values)),
            "metrics": channel_metrics, "task_sweeps": task_sweeps,
        }
    winner = max(report["families"], key=lambda key: report["families"][key]["development_ads_objective"])
    report["winner"] = winner
    report["winner_weights"] = report["families"][winner]["selected_task_weights"]
    args.output.mkdir(parents=True)
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    summary = []
    for name, value in report["families"].items():
        summary.append({
            "family": name, "ads_objective": value["development_ads_objective"],
            "mean_ads": value["mean_ads"], "worst_ads": value["worst_ads"],
            **{f"weight_{key}": weight for key, weight in value["selected_task_weights"].items()},
        })
    pd.DataFrame(summary).sort_values("ads_objective", ascending=False).to_csv(
        args.output / "summary.csv", index=False
    )
    print(json.dumps({"winner": winner, **report["families"][winner]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
