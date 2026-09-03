#!/usr/bin/env python3
"""Score the deployed, ADS-decoupled voice/music presence ensemble."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
from tqdm import tqdm


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from eat_presence import EatPresence, fuse_music_probe, fuse_presence  # noqa: E402
from eat_presence_fusion import (  # noqa: E402
    latent_linear_probability,
    logit_fuse_presence,
)
from pipeline import find_audio_files, load_audio  # noqa: E402
from presence import PannsPresence  # noqa: E402
from telephone_router import TelephoneRouter  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audio-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--eat-dir", type=Path, default=ROOT / "models/eat-base-as2m"
    )
    parser.add_argument("--panns-dir", type=Path, default=ROOT / "models/panns")
    parser.add_argument(
        "--music-head", type=Path,
        default=ROOT / "reports/presence_probe_v1/presence_head.npz",
    )
    parser.add_argument(
        "--phone-head", type=Path,
        default=ROOT / "reports/phone_presence_probe_v2/phone_voice_presence_head.npz",
    )
    parser.add_argument(
        "--telephone-router", type=Path,
        default=ROOT / "model_heads/telephone-router-narrowband-v1.npz",
    )
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()

    files = find_audio_files(args.audio_dir)
    if not files:
        raise FileNotFoundError(f"No audio files in {args.audio_dir}")
    panns = PannsPresence(args.panns_dir, device=args.device)
    eat = EatPresence(
        args.eat_dir, args.panns_dir, device=args.device,
        presence_head_path=args.music_head,
    )
    phone_checkpoint = np.load(args.phone_head, allow_pickle=False)
    router = TelephoneRouter(args.telephone_router)
    rows = []
    for offset in tqdm(range(0, len(files), args.batch_size), desc="presence"):
        paths = files[offset:offset + args.batch_size]
        audios = [load_audio(path) for path in paths]
        panns_scores = [panns.predict(audio) for audio in audios]
        eat_scores = [eat.predict_audio_set(audio) for audio in audios]
        latent = eat.latent_statistics_batch(audios)
        for path, audio, panns_score, eat_score, latent_item in zip(
            paths, audios, panns_scores, eat_scores, latent
        ):
            matrix, mask, music_probe = latent_item
            voice, music = fuse_presence(
                *panns_score, *eat_score, voice_weight=0.35, music_weight=0.90
            )
            if music_probe is not None:
                music = fuse_music_probe(music, music_probe, probe_weight=0.40)
            is_telephone = router.is_narrowband(audio)
            phone_voice = np.nan
            if is_telephone:
                phone_voice = latent_linear_probability(
                    matrix, mask, phone_checkpoint
                )
                voice = logit_fuse_presence(voice, phone_voice, 0.10)
            rows.append({
                "ID": path.stem,
                "PANNS_VOICE_PRESENT_PROB": panns_score[0],
                "PANNS_MUSIC_PRESENT_PROB": panns_score[1],
                "EAT_VOICE_PRESENT_PROB": eat_score[0],
                "EAT_MUSIC_PRESENT_PROB": eat_score[1],
                "EAT_MUSIC_PROBE_PROB": music_probe,
                "PHONE_VOICE_PROB": phone_voice,
                "IS_TELEPHONE": int(is_telephone),
                "VOICE_PRESENT_PROB": voice,
                "MUSIC_PRESENT_PROB": music,
            })

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved {len(rows)} scores to {args.output}")


if __name__ == "__main__":
    main()
