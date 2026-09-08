#!/usr/bin/env python3
"""Audit a frozen v94 multi-stream prompt checkpoint on reserved development data.

The script consumes original mixtures only.  It supports deterministic fast
telephone proxies and the exact ffmpeg codec paths used by the channel audit.
It never accepts a path containing ``locked`` so checkpoint/fusion selection
cannot accidentally inspect the final reserved split.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_curve


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from full_coverage_wpt import resolve_ffmpeg  # noqa: E402
from multistream_prompt_spectra import MultiStreamSpectraMultitask  # noqa: E402
from pipeline import load_audio  # noqa: E402
from telephone_channel import apply_channel  # noqa: E402
from train_mixfake_multistream_prompt_v94 import (  # noqa: E402
    TASK_INDEX, finite_eer, load_spectra, preemphasis,
)
from wpt_spectra_inference import fixed_windows  # noqa: E402


CHANNELS = (
    "clean", "paired_fast", "g711_ulaw", "g722_wb", "opus_nb_8k",
    "transcode_g711_opus",
)
FAST_CHANNELS = ("resample8k", "mulaw_numpy", "fft_narrowband", "lowpass_5k")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_key(value: str) -> int:
    return int.from_bytes(hashlib.sha256(value.encode()).digest()[:8], "little")


def select_label(frame: pd.DataFrame, task: str) -> np.ndarray:
    if task == "all":
        # Absent component labels are intentionally undefined in the broad
        # inventory.  Their filled values are never scored because task_mask
        # excludes those rows below.
        return frame[["VOICE_FAKE", "MUSIC_FAKE", "FILE_FAKE"]].fillna(0).to_numpy(np.int64)
    return frame[f"{task.upper()}_FAKE"].to_numpy(np.int64)


def task_mask(frame: pd.DataFrame, task: str) -> np.ndarray:
    if task == "file":
        return np.ones(len(frame), dtype=bool)
    present = f"{task.upper()}_PRESENT"
    if present not in frame:
        return np.ones(len(frame), dtype=bool)
    return pd.to_numeric(frame[present], errors="coerce").eq(1).to_numpy()


def attach_audio_paths(
    frame: pd.DataFrame, audio_root: Path, flat_audio_dir: Path | None = None,
) -> pd.DataFrame:
    frame = frame.copy()
    if flat_audio_dir is not None:
        by_id = {path.stem: path for path in flat_audio_dir.iterdir() if path.is_file()}
        missing = sorted(set(frame.ID).difference(by_id))
        if missing:
            raise FileNotFoundError(f"flat audio missing for {missing[:5]}")
        frame["PATH"] = frame.ID.map(lambda value: str(by_id[str(value)]))
    elif "PATH" in frame and frame.PATH.notna().all():
        frame["PATH"] = frame.PATH.astype(str)
    elif "ARCHIVE_MEMBER" in frame:
        frame["PATH"] = frame.ARCHIVE_MEMBER.map(lambda value: str(audio_root / str(value)))
    else:
        audio_dir = audio_root / "audio"
        by_id = {path.stem: path for path in audio_dir.iterdir() if path.is_file()}
        missing = sorted(set(frame.ID).difference(by_id))
        if missing:
            raise FileNotFoundError(f"audio missing for {missing[:5]}")
        frame["PATH"] = frame.ID.map(lambda value: str(by_id[str(value)]))
    return frame


def subgroup_eer(frame: pd.DataFrame, scores: np.ndarray, task: str) -> dict[str, float]:
    result: dict[str, float] = {}
    if task == "music":
        for voice_state, cases in {
            "voice_real": ("RR", "RF"),
            "voice_fake": ("FR", "FF"),
        }.items():
            mask = frame.COMPONENT_CASE.isin(cases).to_numpy()
            result[voice_state] = finite_eer(select_label(frame.iloc[mask], task), scores[mask])
    elif task == "voice":
        for music_state, cases in {
            "music_real": ("RR", "FR"),
            "music_fake": ("RF", "FF"),
        }.items():
            mask = frame.COMPONENT_CASE.isin(cases).to_numpy()
            result[music_state] = finite_eer(select_label(frame.iloc[mask], task), scores[mask])
    return result


def load_model(model_dir: Path, checkpoint_path: Path, device: torch.device):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint.get("model_type") != "mixfake_multistream_prompt_v94":
        raise ValueError(f"not a v94 checkpoint: {checkpoint_path}")
    config = checkpoint["config"]
    model = MultiStreamSpectraMultitask(
        load_spectra(model_dir, device),
        base_tokens=int(config["base_tokens"]),
        frequency_tokens=int(config["frequency_tokens"]),
        texture_tokens=int(config["texture_tokens"]),
        prompt_dropout=float(config["prompt_dropout"]),
        temperature=float(config["temperature"]),
    ).to(device)
    model.load_trainable_state_dict(checkpoint["state"])
    return model.eval(), checkpoint


@torch.inference_mode()
def predict_channel(
    model, frame: pd.DataFrame, audio_root: Path, channel: str,
    window: int, views: int, batch_size: int, device: torch.device,
    task_index: int | None,
) -> np.ndarray:
    needs_ffmpeg = channel in {
        "g711_ulaw", "g722_wb", "opus_nb_8k", "transcode_g711_opus",
    }
    ffmpeg = resolve_ffmpeg() if needs_ffmpeg else None
    outputs: list[np.ndarray] = []
    started = time.monotonic()
    for offset in range(0, len(frame), batch_size):
        rows = frame.iloc[offset:offset + batch_size]
        arrays = []
        for row in rows.itertuples(index=False):
            audio = load_audio(Path(row.PATH))
            key = stable_key(str(row.ID))
            selected = (
                FAST_CHANNELS[key % len(FAST_CHANNELS)]
                if channel == "paired_fast" else channel
            )
            if selected != "clean":
                audio = apply_channel(audio, selected, ffmpeg=ffmpeg, key=key % (2 ** 31))
            arrays.append(fixed_windows(audio, window, views))
        waveform = preemphasis(torch.from_numpy(np.stack(arrays)).to(device))
        with torch.autocast(
            device_type=device.type, dtype=torch.bfloat16,
            enabled=device.type == "cuda",
        ):
            logits = model(waveform)
            if task_index is not None:
                logits = logits[:, task_index]
        outputs.append(logits.float().sigmoid().cpu().numpy())
        if offset == 0 or offset + len(rows) == len(frame):
            print(json.dumps({
                "channel": channel, "done": offset + len(rows), "total": len(frame),
                "seconds": time.monotonic() - started,
            }), flush=True)
    return np.concatenate(outputs).astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--audio-root", type=Path, required=True)
    parser.add_argument(
        "--flat-audio-dir", type=Path, default=None,
        help="Optional already-rendered directory keyed by <ID>.<ext>.",
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, default=ROOT / "models/external/spectra_aasist")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--channels", nargs="+", choices=CHANNELS, default=list(CHANNELS))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--views", type=int, default=None)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if "locked" in " ".join(map(str, (args.manifest, args.checkpoint, args.output))).lower():
        raise ValueError("locked data are forbidden during candidate selection")
    frame = pd.read_csv(args.manifest, dtype={"ID": str})
    required = {"ID", "COMPONENT_CASE", "VOICE_FAKE", "MUSIC_FAKE", "FILE_FAKE"}
    if missing := required.difference(frame.columns):
        raise ValueError(f"manifest misses {sorted(missing)}")
    if frame.ID.duplicated().any():
        raise ValueError("manifest IDs are not unique")
    frame = attach_audio_paths(frame, args.audio_root, args.flat_audio_dir)
    device = torch.device(args.device)
    model, checkpoint = load_model(args.model_dir, args.checkpoint, device)
    config = checkpoint["config"]
    task = str(config["task"])
    if task not in (*TASK_INDEX, "all"):
        raise ValueError(f"unknown checkpoint task {task!r}")
    if task in ("voice", "music") and f"{task.upper()}_PRESENT" in frame:
        frame = frame.loc[frame[f"{task.upper()}_PRESENT"].eq(1)].reset_index(drop=True)
    task_index = None if task == "all" else TASK_INDEX[task]
    window = int(config["window"])
    views = int(config["views"] if args.views is None else args.views)
    label = select_label(frame, task)
    args.output.mkdir(parents=True)
    reports = []
    score_archive: dict[str, np.ndarray] = {"ids": frame.ID.to_numpy(dtype=str)}
    for channel in args.channels:
        scores = predict_channel(
            model, frame, args.audio_root, channel, window, views,
            args.batch_size, device, task_index,
        )
        score_archive[channel] = scores
        if task == "all":
            eers = {
                name: finite_eer(
                    label[task_mask(frame, name), index],
                    scores[task_mask(frame, name), index],
                )
                for index, name in enumerate(("voice", "music", "file"))
            }
            weighted_error = (
                .2 * eers["voice"] + .3 * eers["music"] + .5 * eers["file"]
            )
            subgroups = {
                name: subgroup_eer(frame, scores[:, index], name)
                for index, name in enumerate(("voice", "music", "file"))
            }
            mean_by_case = {
                case: {
                    name: float(scores[frame.COMPONENT_CASE.eq(case).to_numpy(), index].mean())
                    for index, name in enumerate(("voice", "music", "file"))
                }
                for case in ("RR", "RF", "FR", "FF")
            }
        else:
            eers = {task: finite_eer(label, scores)}
            weighted_error = eers[task]
            subgroups = subgroup_eer(frame, scores, task)
            mean_by_case = {
                case: float(scores[frame.COMPONENT_CASE.eq(case).to_numpy()].mean())
                for case in ("RR", "RF", "FR", "FF")
            }
        report = {
            "channel": channel,
            "eer": weighted_error,
            "task_eer": eers,
            "ads": 1 - weighted_error if task == "all" else None,
            "subgroup_eer": subgroups,
            "manifest_channel_eer": (
                {
                    str(group): (
                        {
                            name: finite_eer(
                                label[
                                    frame.CHANNEL.fillna("missing").astype(str).eq(str(group)).to_numpy()
                                    & task_mask(frame, name), index
                                ],
                                scores[
                                    frame.CHANNEL.fillna("missing").astype(str).eq(str(group)).to_numpy()
                                    & task_mask(frame, name), index
                                ],
                            )
                            for index, name in enumerate(("voice", "music", "file"))
                        }
                        if task == "all" else finite_eer(
                            label[frame.CHANNEL.fillna("missing").astype(str).eq(str(group)).to_numpy()],
                            scores[frame.CHANNEL.fillna("missing").astype(str).eq(str(group)).to_numpy()],
                        )
                    )
                    for group in sorted(frame.CHANNEL.fillna("missing").astype(str).unique())
                }
                if "CHANNEL" in frame else {}
            ),
            "layout_eer": (
                {
                    str(group): (
                        {
                            name: finite_eer(
                                label[
                                    frame.MIX_MODE.fillna("missing").astype(str).eq(str(group)).to_numpy()
                                    & task_mask(frame, name), index
                                ],
                                scores[
                                    frame.MIX_MODE.fillna("missing").astype(str).eq(str(group)).to_numpy()
                                    & task_mask(frame, name), index
                                ],
                            )
                            for index, name in enumerate(("voice", "music", "file"))
                        }
                        if task == "all" else finite_eer(
                            label[frame.MIX_MODE.fillna("missing").astype(str).eq(str(group)).to_numpy()],
                            scores[frame.MIX_MODE.fillna("missing").astype(str).eq(str(group)).to_numpy()],
                        )
                    )
                    for group in sorted(frame.MIX_MODE.fillna("missing").astype(str).unique())
                }
                if "MIX_MODE" in frame else {}
            ),
            "mean_score_by_case": mean_by_case,
        }
        reports.append(report)
        print(json.dumps(report), flush=True)
    np.savez_compressed(args.output / "predictions.npz", **score_archive)
    summary = {
        "task": task, "rows": len(frame), "views": views,
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": sha256_file(args.checkpoint),
        "manifest": str(args.manifest), "manifest_sha256": sha256_file(args.manifest),
        "channels": reports,
        "mean_eer": float(np.mean([item["eer"] for item in reports])),
        "worst_eer": float(max(item["eer"] for item in reports)),
        "locked_scores_read": False,
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    pd.DataFrame([{
        "channel": item["channel"], "eer": item["eer"],
        **{f"{name}_eer": value for name, value in item["task_eer"].items()},
    } for item in reports]).to_csv(args.output / "summary.csv", index=False)
    print(json.dumps({"status": "complete", "output": str(args.output),
                      "mean_eer": summary["mean_eer"],
                      "worst_eer": summary["worst_eer"]}), flush=True)


if __name__ == "__main__":
    main()
