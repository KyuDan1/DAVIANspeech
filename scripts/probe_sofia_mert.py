#!/usr/bin/env python3
"""Evaluate the official SOFIA MERT music expert on a labelled audio bank."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from evaluate_diagnostic import official_eer  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sofia-root", type=Path, default=Path("/tmp/sofia"))
    parser.add_argument("--truth", type=Path, required=True)
    parser.add_argument("--audio-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--audio-type", choices=["all", "music", "mixed", "voice"],
                        default="all")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--backend", choices=["official", "safe"], default="official")
    parser.add_argument("--release-padding", action="store_true")
    args = parser.parse_args()

    device = torch.device(args.device)
    checkpoint_path = args.sofia_root / "checkpoint/sofia_G1_mert/sofia_G1_mert.pt"
    if args.backend == "official":
        # The official repository is a Python package named ``sofia``.  Keep its
        # parent ahead of this repository so its released checkpoint code is used.
        sys.path.insert(0, str(args.sofia_root.parent))
        from sofia.predict_audio import load_sofia_models  # noqa: PLC0415
        from sofia.utils.audio import load_audio  # noqa: PLC0415
        from sofia.utils.config import load_config  # noqa: PLC0415

        config_path = args.sofia_root / "config/sofia_g1_mert.yaml"
        cfg = load_config(str(config_path))
        cfg["audio_branches"]["mert"]["weights"] = str(
            args.sofia_root / "third_party/mert"
        )
        encoders, fusion, head = load_sofia_models(
            cfg, device, str(checkpoint_path)
        )
        encoder = encoders["mert"]
        safe_detector = None
    else:
        from sofia_mert_detector import SofiaMertDetector  # noqa: PLC0415

        safe_detector = SofiaMertDetector(
            args.sofia_root / "third_party/mert", checkpoint_path,
            device=str(device), pad_to_release_length=args.release_padding,
        )

    truth = pd.read_csv(args.truth, dtype={"ID": str})
    if args.audio_type != "all":
        truth = truth[truth.AUDIO_TYPE.eq(args.audio_type)].copy()
    if args.limit:
        truth = truth.head(args.limit).copy()
    extensions = (".wav", ".flac", ".mp3", ".ogg", ".m4a", ".aac", ".opus")
    paths = {}
    for path in args.audio_dir.iterdir():
        if path.is_file() and path.suffix.lower() in extensions:
            paths[path.stem] = path

    rows, started = [], time.monotonic()
    with torch.inference_mode():
        for index, row in enumerate(truth.itertuples(index=False), start=1):
            path = paths.get(str(row.ID))
            if path is None:
                raise FileNotFoundError(f"No audio for {row.ID}")
            if args.backend == "official":
                waveform, sample_rate = load_audio(
                    str(path), target_sr=44100, mono=False, normalize=True
                )
                embedding = encoder(
                    waveform.unsqueeze(0).to(device), sample_rate=sample_rate
                )
                fused = fusion({"mert": embedding})
                logits = head(
                    fused, torch.zeros(1, dtype=torch.long, device=device)
                )
                score = float(logits.softmax(dim=-1)[0, 1].cpu())
            else:
                score = safe_detector.score_path(path)
            rows.append({"ID": str(row.ID), "SOFIA_MERT_FAKE_PROB": score})
            if index % 25 == 0:
                print(f"{index}/{len(truth)}", flush=True)
    predictions = pd.DataFrame(rows)
    merged = truth.merge(predictions, on="ID", validate="one_to_one")
    metrics = {
        "samples": len(merged),
        "elapsed_seconds": time.monotonic() - started,
    }
    if merged.FILE_FAKE.nunique() == 2:
        metrics["FILE_EER"] = official_eer(
            merged.FILE_FAKE, merged.SOFIA_MERT_FAKE_PROB
        )
    music = merged[merged.MUSIC_PRESENT.eq(1)]
    if len(music) and music.MUSIC_FAKE.nunique() == 2:
        metrics["MUSIC_EER"] = official_eer(
            music.MUSIC_FAKE, music.SOFIA_MERT_FAKE_PROB
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(args.output, index=False)
    args.output.with_suffix(".json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
