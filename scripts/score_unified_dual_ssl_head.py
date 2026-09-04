#!/usr/bin/env python3
"""Score one or more unified EAT/SPEAR heads on untouched audit banks."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.special import expit, logit

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from evaluate_diagnostic import score_frame  # noqa: E402
from train_spear_temporal_bin_mil import load_archive  # noqa: E402
from train_unified_dual_ssl_head import (  # noqa: E402
    load_block, load_eat, move_block_to_device, predict, truth_for,
)
from unified_dual_ssl_head import UnifiedDualSSLHead  # noqa: E402


DEFAULT_DATASETS = (
    "factorial_eval_1200_v2_holdout",
    "phone_factorial_1200_v1",
    "yue_cross_component_audit_v1",
    "source_disjoint_music_telephone_v1",
    "suno_vocals_v1",
    "suno_vocals_codec_stress_v1",
)


def load_model(path: Path, device: torch.device):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    model = UnifiedDualSSLHead(**checkpoint["config"])
    model.load_state_dict(checkpoint["model"])
    return model.to(device).eval(), checkpoint


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, nargs="+", required=True)
    parser.add_argument(
        "--eat-cache-root", type=Path,
        default=ROOT / "output/eat_hierarchical_stats_v1",
    )
    parser.add_argument(
        "--spear-cache-root", type=Path,
        default=ROOT / "reports/spear_temporal_attention_v1/cache",
    )
    parser.add_argument("--datasets", nargs="+", default=list(DEFAULT_DATASETS))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument(
        "--stream-mode", choices=("joint", "eat"), default="joint",
        help="Use the full dual-SSL expert or its EAT-only specialist path.",
    )
    parser.add_argument(
        "--save-latents", action="store_true",
        help="Save the task-wise pre-classifier embeddings for MoE routing.",
    )
    parser.add_argument(
        "--save-member-probs", action="store_true",
        help="Append each checkpoint's task probabilities for subset ensembles.",
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    models = [load_model(path, device) for path in args.checkpoint]
    reference = models[0][1]
    for _, checkpoint in models[1:]:
        for key in ("eat_projection", "spear_projection", "spear_layers"):
            if not np.array_equal(reference[key], checkpoint[key]):
                raise ValueError(f"checkpoint metadata differs for {key}")
    spear_cache = (
        load_archive(args.spear_cache_root) if args.stream_mode == "joint" else None
    )
    records, all_predictions = [], []
    for name in args.datasets:
        if args.stream_mode == "joint":
            frame, block = load_block(
                args.eat_cache_root, spear_cache, name, "audit"
            )
        else:
            frame = truth_for(name, "audit")
            frame["DATASET"] = name
            eat = load_eat(args.eat_cache_root, name, frame)
            config = reference["config"]
            count = len(frame)
            block = {
                **eat,
                "spear": np.zeros((
                    count, config["maximum_views"], config["maximum_bins"],
                    config["spear_layers"], config["spear_stats"],
                    config["spear_dimension"],
                ), dtype=np.float16),
                "spear_mask": np.zeros((
                    count, config["maximum_views"], config["maximum_bins"]
                ), dtype=bool),
            }
        block = move_block_to_device(block, device)
        fake_members, presence_members, latent_members = [], [], []
        for model, _ in models:
            if args.save_latents:
                fake_chunks, presence_chunks, latent_chunks = [], [], []
                for offset in range(0, len(frame), args.batch_size):
                    stop = min(offset + args.batch_size, len(frame))
                    with torch.inference_mode():
                        task, presence_logit, _joint, pooled = model.forward_with_embedding(
                            block["eat"][offset:stop],
                            block["spear"][offset:stop],
                            block["eat_mask"][offset:stop],
                            block["spear_mask"][offset:stop],
                        )
                    fake_chunks.append(model.probabilities(task).cpu().numpy())
                    presence_chunks.append(
                        presence_logit.sigmoid().cpu().numpy()
                    )
                    latent_chunks.append(
                        pooled.flatten(1).float().cpu().numpy()
                    )
                fake = np.concatenate(fake_chunks)
                presence = np.concatenate(presence_chunks)
                latent_members.append(np.concatenate(latent_chunks))
            else:
                fake, presence = predict(model, block, device, args.batch_size)
            fake_members.append(fake)
            presence_members.append(presence)
        fake = expit(np.mean([
            logit(np.clip(value, 1e-5, 1 - 1e-5)) for value in fake_members
        ], axis=0))
        presence = expit(np.mean([
            logit(np.clip(value, 1e-5, 1 - 1e-5)) for value in presence_members
        ], axis=0))
        prediction = pd.DataFrame({
            "FILE_FAKE_PROB": fake[:, 2],
            "VOICE_FAKE_PROB": fake[:, 0],
            "MUSIC_FAKE_PROB": fake[:, 1],
            "VOICE_PRESENT_PROB": presence[:, 0],
            "MUSIC_PRESENT_PROB": presence[:, 1],
        }, index=frame.ID)
        metrics = score_frame(frame.set_index("ID").join(prediction))
        record = {"DATASET": name, **metrics}
        record["FAKE_OVER_HALF"] = int((fake[:, 2] > 0.5).sum())
        record["MUSIC_OVER_HALF"] = int((fake[:, 1] > 0.5).sum())
        records.append(record)
        current = prediction.reset_index().rename(columns={"index": "ID"})
        current.insert(0, "DATASET", name)
        if args.save_member_probs:
            for member_index, member in enumerate(fake_members):
                for task_index, task in enumerate(("VOICE", "MUSIC", "FILE")):
                    current[
                        f"MEMBER_{member_index:02d}_{task}_FAKE_PROB"
                    ] = member[:, task_index]
        all_predictions.append(current)
        if args.save_latents:
            np.savez_compressed(
                args.output_dir / f"{name}_latent.npz",
                ids=frame.ID.to_numpy(dtype=str),
                latent=np.mean(latent_members, axis=0).astype(np.float16),
                latent_members=np.stack(latent_members).astype(np.float16),
            )
        print(json.dumps(record, allow_nan=True), flush=True)

    pd.DataFrame(records).to_csv(args.output_dir / "metrics.csv", index=False)
    pd.concat(all_predictions).to_csv(
        args.output_dir / "predictions.csv", index=False
    )


if __name__ == "__main__":
    main()
