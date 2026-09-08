#!/usr/bin/env python3
"""Score the frozen v55 EAT Music adapter and its fixed anchor residual."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import transformers  # import before local timm compatibility modules


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from eat_detector import _load_local_model  # noqa: E402
from eat_music_adapter import EatMusicAdapter  # noqa: E402
from evaluate_diagnostic import official_eer, score_frame  # noqa: E402
from train_eat_music_adapter_v55 import (  # noqa: E402
    DEV_NAME, audio_index, deterministic_views,
)


RETROSPECTIVE = {
    "codec_mixed_blind_v5",
    "codec_mixed_blind_v6",
    "codec_mixed_blind_v7",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def load_model(checkpoint_path: Path, device: torch.device) -> EatMusicAdapter:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    eat = _load_local_model(ROOT / "models/eat-base-as2m", device)
    actual_hash = sha256(ROOT / "models/eat-base-as2m/model.safetensors")
    if actual_hash != checkpoint["base_eat_sha256"]:
        raise ValueError("base EAT hash differs from frozen checkpoint")
    model = EatMusicAdapter(eat_model=eat, **checkpoint["config"]).to(device)
    incompatible = model.load_state_dict(checkpoint["model"], strict=False)
    if incompatible.unexpected_keys or any(
        not name.startswith("eat.") for name in incompatible.missing_keys
    ):
        raise ValueError(f"incompatible adapter checkpoint: {incompatible}")
    return model.eval()


@torch.inference_mode()
def predict(
    model: EatMusicAdapter,
    paths: list[Path],
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    result = []
    for offset in range(0, len(paths), batch_size):
        blocks = [deterministic_views(path) for path in paths[offset : offset + batch_size]]
        features = torch.stack([block[0] for block in blocks])[:, :, None].to(device)
        mask = torch.stack([block[1] for block in blocks]).to(device)
        with torch.autocast(
            device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"
        ):
            logits = model(features, mask)
        result.append(logits.float().sigmoid().cpu().numpy())
    return np.concatenate(result)


def truth_for(name: str) -> pd.DataFrame:
    path = ROOT / "data/eval" / name / "truth.csv"
    return pd.read_csv(path, dtype={"ID": str})


def logit(values: np.ndarray) -> np.ndarray:
    values = np.clip(values.astype(np.float64), 1e-6, 1 - 1e-6)
    return np.log(values) - np.log1p(-values)


def sigmoid(values: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-np.clip(values, -40, 40)))


def score_dataset(
    model: EatMusicAdapter,
    name: str,
    anchor_path: Path,
    output_dir: Path,
    device: torch.device,
    batch_size: int,
    residual_weight: float,
) -> dict[str, float]:
    if name == "codec_mixed_blind_v8":
        raise ValueError("blind v8 is reserved and forbidden to v55")
    if name != DEV_NAME and name not in RETROSPECTIVE:
        raise ValueError("v55 scorer only permits v4 and retired v5-v7 banks")
    truth = truth_for(name)
    index = audio_index(name)
    missing = set(truth.ID).difference(index)
    if missing:
        raise FileNotFoundError(f"{name}: missing audio IDs {sorted(missing)[:5]}")
    paths = [index[item] for item in truth.ID]
    expert = predict(model, paths, device, batch_size)
    anchor = pd.read_csv(anchor_path, dtype={"ID": str}).set_index("ID").loc[truth.ID]
    candidate = anchor.copy()
    candidate["MUSIC_FAKE_PROB"] = sigmoid(
        (1 - residual_weight) * logit(anchor.MUSIC_FAKE_PROB.to_numpy())
        + residual_weight * logit(expert)
    )
    anchor_metrics = score_frame(truth.set_index("ID").join(anchor))
    candidate_metrics = score_frame(truth.set_index("ID").join(candidate))

    output_dir.mkdir(parents=True, exist_ok=True)
    prediction = candidate.reset_index()
    prediction["EAT_ADAPTER_MUSIC_PROB"] = expert
    prediction.to_csv(output_dir / "predictions.csv", index=False)
    rows = []
    group_columns = [
        column for column in ("CHANNEL", "EVAL_CELL", "MIX_MODE") if column in truth
    ]
    for column in group_columns:
        for value, block in truth.groupby(column, dropna=False):
            ids = block.ID
            if block.MUSIC_FAKE.dropna().nunique() < 2:
                continue
            base = score_frame(block.set_index("ID").join(anchor.loc[ids]))
            updated = score_frame(block.set_index("ID").join(candidate.loc[ids]))
            rows.append({
                "GROUP_COLUMN": column, "GROUP": str(value), "N": len(block),
                "ANCHOR_MUSIC_EER": base["MUSIC_EER"],
                "V55_MUSIC_EER": updated["MUSIC_EER"],
            })
    pd.DataFrame(rows).to_csv(output_dir / "subgroups.csv", index=False)
    summary = {
        "dataset": name,
        "residual_weight": residual_weight,
        "anchor_ads": float(anchor_metrics["ADS"]),
        "v55_ads": float(candidate_metrics["ADS"]),
        "anchor_music_eer": float(anchor_metrics["MUSIC_EER"]),
        "v55_music_eer": float(candidate_metrics["MUSIC_EER"]),
        "expert_music_eer": float(official_eer(
            truth.loc[truth.MUSIC_PRESENT.eq(1), "MUSIC_FAKE"].astype(int),
            expert[truth.MUSIC_PRESENT.eq(1).to_numpy()],
        )),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--anchor", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--residual-weight", type=float, default=.15)
    args = parser.parse_args()
    if not 0 <= args.residual_weight <= 1:
        parser.error("residual weight must lie in [0,1]")
    device = torch.device(args.device)
    model = load_model(args.checkpoint, device)
    summary = score_dataset(
        model, args.dataset, args.anchor, args.output_dir, device,
        args.batch_size, args.residual_weight,
    )
    summary["checkpoint_sha256"] = sha256(args.checkpoint)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
