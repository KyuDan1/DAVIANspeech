#!/usr/bin/env python3
"""Render the shared speaker/content manifest with F5-TTS voice cloning."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
import soundfile as sf


SR = 16_000


def write_clip(audio, sample_rate: int, destination: Path) -> float:
    audio = np.asarray(audio, dtype=np.float32).reshape(-1)
    if sample_rate != SR:
        audio = librosa.resample(audio, orig_sr=sample_rate, target_sr=SR,
                                 res_type="soxr_hq")
    if len(audio) < 4 * SR:
        audio = np.pad(audio, (0, 4 * SR - len(audio)))
    audio = audio[:60 * SR]
    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    if peak > 0.99:
        audio *= 0.99 / peak
    sf.write(destination, audio, SR, format="FLAC", subtype="PCM_16")
    return len(audio) / SR


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pool-dir", type=Path, required=True)
    parser.add_argument("--output-manifest", type=Path, required=True)
    parser.add_argument("--model", default="F5TTS_v1_Base")
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260831)
    parser.add_argument("--nfe-step", type=int, default=32)
    args = parser.parse_args()

    from f5_tts.api import F5TTS

    jobs = pd.read_csv(args.pool_dir / "generation_manifest.csv", dtype=str)
    speakers = sorted(jobs.SPEAKER.unique())
    selected = set(speakers[args.shard_index::args.num_shards])
    jobs = jobs[jobs.SPEAKER.isin(selected)]
    print(f"Loading {args.model} for {len(jobs)} jobs", flush=True)
    model = F5TTS(model=args.model, device="cuda")
    rows = []
    for job in jobs.itertuples(index=False):
        generator = "f5tts_v1_base"
        sample_id = f"voice_fake_{generator}_{job.JOB_ID}"
        destination = args.pool_dir / "audio" / f"{sample_id}.flac"
        if destination.is_file():
            duration = sf.info(destination).duration
        else:
            seed = int(hashlib.sha256(
                f"{args.seed}|{generator}|{job.JOB_ID}".encode()
            ).hexdigest()[:8], 16)
            waveform, sample_rate, _ = model.infer(
                ref_file=job.REFERENCE_AUDIO,
                ref_text=job.REFERENCE_TEXT,
                gen_text=job.TEXT,
                seed=seed,
                nfe_step=args.nfe_step,
                progress=None,
                show_info=lambda *_: None,
            )
            duration = write_clip(waveform, sample_rate, destination)
        rows.append({
            "ID": sample_id, "FILE_FAKE": 1, "VOICE_FAKE": 1,
            "MUSIC_FAKE": "", "VOICE_PRESENT": 1, "MUSIC_PRESENT": 0,
            "AUDIO_TYPE": "voice", "SOURCE": "synthetic_tts",
            "GENERATOR": generator, "CONDITION": "speech_only",
            "SPLIT": job.SPLIT, "GROUP_ID": job.JOB_ID,
            "SPEAKER": job.SPEAKER, "CATEGORY": job.CATEGORY,
            "GENDER": job.GENDER, "AGE": job.AGE,
            "SOURCE_FILE": job.SOURCE_FILE, "DURATION": round(duration, 3),
        })
        print(f"  {sample_id} {duration:.2f}s", flush=True)
    args.output_manifest.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.output_manifest, index=False)
    print(f"Wrote {len(rows)} rows to {args.output_manifest}")


if __name__ == "__main__":
    main()
