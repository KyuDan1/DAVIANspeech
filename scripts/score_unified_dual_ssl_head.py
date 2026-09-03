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
from train_unified_dual_ssl_head import load_block, move_block_to_device, predict  # noqa: E402
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
    args = parser.parse_args()

    device = torch.device(args.device)
    models = [load_model(path, device) for path in args.checkpoint]
    reference = models[0][1]
    for _, checkpoint in models[1:]:
        for key in ("eat_projection", "spear_projection", "spear_layers"):
            if not np.array_equal(reference[key], checkpoint[key]):
                raise ValueError(f"checkpoint metadata differs for {key}")
    spear_cache = load_archive(args.spear_cache_root)
    records, all_predictions = [], []
    for name in args.datasets:
        frame, block = load_block(
            args.eat_cache_root, spear_cache, name, "audit"
        )
        block = move_block_to_device(block, device)
        fake_members, presence_members = [], []
        for model, _ in models:
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
        all_predictions.append(current)
        print(json.dumps(record, allow_nan=True), flush=True)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records).to_csv(args.output_dir / "metrics.csv", index=False)
    pd.concat(all_predictions).to_csv(
        args.output_dir / "predictions.csv", index=False
    )


if __name__ == "__main__":
    main()
