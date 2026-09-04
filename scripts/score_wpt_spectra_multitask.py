#!/usr/bin/env python3
"""Score compact WPT/Spectra checkpoints on development or locked banks."""

from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from train_wpt_spectra_multitask import (  # noqa: E402
    AudioBankDataset, finite_eer, load_frame, load_spectra, preemphasis,
)
from multistream_prompt_spectra import MultiStreamSpectraMultitask  # noqa: E402
from wpt_spectra import WPTSpectraMultitask  # noqa: E402


DEFAULT_DATASETS = (
    "factorial_eval_1200_v2_holdout",
    "phone_factorial_1200_v1",
    "yue_cross_component_audit_v1",
)


def load_model(
    checkpoint_path: Path, model_dir: Path, device: torch.device,
) -> tuple[WPTSpectraMultitask, dict]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model_type = checkpoint.get("model_type")
    if model_type not in {
        "wpt_spectra_multitask", "multistream_prompt_spectra_multitask",
    }:
        raise ValueError(f"not a prompt/Spectra checkpoint: {checkpoint_path}")
    config = checkpoint["config"]
    base = load_spectra(model_dir, device)
    if model_type == "wpt_spectra_multitask":
        model = WPTSpectraMultitask(
            base,
            prompt_tokens=int(config["prompt_tokens"]),
            wavelet_tokens=int(config["wavelet_tokens"]),
            temperature=float(config["temperature"]),
        ).to(device)
    else:
        model = MultiStreamSpectraMultitask(
            base,
            base_tokens=int(config["base_tokens"]),
            frequency_tokens=int(config["frequency_tokens"]),
            texture_tokens=int(config["texture_tokens"]),
            prompt_dropout=float(config["prompt_dropout"]),
            temperature=float(config["temperature"]),
        ).to(device)
    model.load_trainable_state_dict(checkpoint["state"])
    model.eval()
    return model, checkpoint


@torch.inference_mode()
def score_loader(
    model: WPTSpectraMultitask, loader: DataLoader, device: torch.device,
    save_latents: bool,
) -> tuple[np.ndarray, np.ndarray | None]:
    logits, latents = [], []
    for windows, _target, _presence, _index in loader:
        windows = preemphasis(windows.to(device, non_blocking=True))
        with torch.autocast(
            device_type=device.type, dtype=torch.bfloat16,
            enabled=device.type == "cuda",
        ):
            view_logits, embedding = model.forward_windows_with_embedding(windows)
            output = model.aggregate(view_logits)
        logits.append(output.float().cpu().numpy())
        if save_latents:
            # Mean/std/max retain both stable content and view disagreement,
            # which lets a later router detect sequential mixtures.
            embedding = embedding.float()
            summary = torch.cat((
                embedding.mean(1),
                embedding.std(1, unbiased=False),
                embedding.amax(1),
            ), dim=-1)
            latents.append(summary.cpu().numpy().astype(np.float16))
    return (
        np.concatenate(logits),
        np.concatenate(latents) if save_latents else None,
    )


def metrics(frame: pd.DataFrame, probability: np.ndarray) -> dict[str, float]:
    voice = frame.VOICE_PRESENT.eq(1).to_numpy()
    music = frame.MUSIC_PRESENT.eq(1).to_numpy()
    eers = {
        "VOICE_EER": finite_eer(frame.loc[voice, "VOICE_FAKE"], probability[voice, 0]),
        "MUSIC_EER": finite_eer(frame.loc[music, "MUSIC_FAKE"], probability[music, 1]),
        "FILE_EER": finite_eer(frame.FILE_FAKE, probability[:, 2]),
    }
    eers["ADS"] = (
        .2 * (1 - eers["VOICE_EER"])
        + .3 * (1 - eers["MUSIC_EER"])
        + .5 * (1 - eers["FILE_EER"])
    )
    return eers


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoints", type=Path, nargs="+", required=True)
    parser.add_argument(
        "--model-dir", type=Path,
        default=ROOT / "models/external/spectra_aasist",
    )
    parser.add_argument("--datasets", nargs="+", default=list(DEFAULT_DATASETS))
    parser.add_argument(
        "--data-role", choices=("train", "eval"), default="eval",
        help="Use train split filtering for banks that contain multiple splits.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--num-workers", type=int, default=6)
    parser.add_argument("--views", type=int, default=3)
    parser.add_argument("--window", type=int, default=64_600)
    parser.add_argument("--save-latents", action="store_true")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    frames, loaders = {}, {}
    for name in args.datasets:
        frame = load_frame(name, args.data_role)
        frames[name] = frame
        loaders[name] = DataLoader(
            AudioBankDataset(
                frame, args.window, args.views,
                stochastic=False, augment=False,
            ),
            batch_size=args.batch_size, shuffle=False,
            num_workers=args.num_workers, pin_memory=True,
            persistent_workers=False,
        )

    all_logits: dict[str, list[np.ndarray]] = {name: [] for name in args.datasets}
    all_latents: dict[str, list[np.ndarray]] = {name: [] for name in args.datasets}
    provenance = []
    for checkpoint_path in args.checkpoints:
        model, checkpoint = load_model(checkpoint_path, args.model_dir, device)
        provenance.append({
            "checkpoint": str(checkpoint_path),
            "model_type": checkpoint.get("model_type"),
            "seed": checkpoint.get("seed"),
            "best_epoch": checkpoint.get("best_epoch"),
            "selection": checkpoint.get("selection"),
        })
        for name in args.datasets:
            logits, latent = score_loader(
                model, loaders[name], device, args.save_latents
            )
            all_logits[name].append(logits)
            if latent is not None:
                all_latents[name].append(latent)
        del model
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    rows, member_rows, predictions = [], [], []
    for name in args.datasets:
        # A logit ensemble preserves rank evidence from confident members.
        logits = np.mean(all_logits[name], axis=0)
        probability = np.exp(-np.logaddexp(0, -np.clip(logits, -30, 30)))
        frame = frames[name]
        rows.append({"DATASET": name, "N": len(frame), **metrics(frame, probability)})
        for member, member_logits in enumerate(all_logits[name]):
            member_probability = np.exp(-np.logaddexp(
                0, -np.clip(member_logits, -30, 30)
            ))
            member_rows.append({
                "MEMBER": member, "DATASET": name, "N": len(frame),
                **metrics(frame, member_probability),
            })
        predictions.append(pd.DataFrame({
            "DATASET": name,
            "ID": frame.ID.to_numpy(),
            "VOICE_FAKE_PROB": probability[:, 0],
            "MUSIC_FAKE_PROB": probability[:, 1],
            "FILE_FAKE_PROB": probability[:, 2],
        }))
        if args.save_latents:
            np.savez_compressed(
                args.output_dir / f"{name}_latent.npz",
                ids=frame.ID.to_numpy(dtype=str),
                latent=np.mean(all_latents[name], axis=0).astype(np.float16),
                latent_members=np.stack(all_latents[name]).astype(np.float16),
                logits=logits.astype(np.float32),
                logit_members=np.stack(all_logits[name]).astype(np.float32),
            )
    metrics_frame = pd.DataFrame(rows)
    metrics_frame.to_csv(args.output_dir / "metrics.csv", index=False)
    pd.DataFrame(member_rows).to_csv(
        args.output_dir / "member_metrics.csv", index=False
    )
    pd.concat(predictions, ignore_index=True).to_csv(
        args.output_dir / "predictions.csv", index=False
    )
    (args.output_dir / "provenance.json").write_text(
        json.dumps(provenance, indent=2) + "\n", encoding="utf-8"
    )
    print(metrics_frame.round(6).to_string(index=False))


if __name__ == "__main__":
    main()
