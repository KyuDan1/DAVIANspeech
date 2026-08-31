#!/usr/bin/env python3
"""Render the multi-generator voice manifest with Qwen3-TTS checkpoints."""

from __future__ import annotations

import argparse
import gc
import hashlib
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
import soundfile as sf
import torch


SR = 16_000


def write_clip(audio, sample_rate: int, destination: Path) -> float:
    audio = np.asarray(audio, dtype=np.float32).reshape(-1)
    if sample_rate != SR:
        audio = librosa.resample(audio, orig_sr=sample_rate, target_sr=SR,
                                 res_type="soxr_hq")
    if len(audio) < 4 * SR:
        audio = np.pad(audio, (0, 4 * SR - len(audio)))
    audio = audio[:60 * SR]
    peak = float(np.max(np.abs(audio))) if len(audio) else 0.0
    if peak > 0.99:
        audio *= 0.99 / peak
    sf.write(destination, audio, SR, format="FLAC", subtype="PCM_16")
    return len(audio) / SR


def fake_row(job, sample_id: str, generator: str, duration: float) -> dict:
    return {
        "ID": sample_id, "FILE_FAKE": 1, "VOICE_FAKE": 1,
        "MUSIC_FAKE": "", "VOICE_PRESENT": 1, "MUSIC_PRESENT": 0,
        "AUDIO_TYPE": "voice", "SOURCE": "synthetic_tts",
        "GENERATOR": generator, "CONDITION": "speech_only",
        "SPLIT": job.SPLIT, "GROUP_ID": job.JOB_ID,
        "SPEAKER": job.SPEAKER, "CATEGORY": job.CATEGORY,
        "GENDER": job.GENDER, "AGE": job.AGE,
        "SOURCE_FILE": job.SOURCE_FILE, "DURATION": round(duration, 3),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pool-dir", type=Path, required=True)
    parser.add_argument("--backend", choices=["finetuned", "custom"], required=True)
    parser.add_argument("--custom-model", default="Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice")
    parser.add_argument("--custom-speaker", default="Sohee")
    parser.add_argument("--output-manifest", type=Path, required=True)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260831)
    args = parser.parse_args()

    from qwen_tts import Qwen3TTSModel

    jobs = pd.read_csv(args.pool_dir / "generation_manifest.csv", dtype=str)
    speakers = sorted(jobs.SPEAKER.unique())
    selected_speakers = set(speakers[args.shard_index::args.num_shards])
    jobs = jobs[jobs.SPEAKER.isin(selected_speakers)]
    rows = []
    if args.backend == "custom":
        groups = [(args.custom_model, jobs)]
        generator = "qwen3tts_custom_0.6b"
    else:
        groups = list(jobs.groupby("QWEN_MODEL", sort=True))
        generator = "qwen3tts_finetuned_1.7b"

    for model_path, group in groups:
        print(f"Loading {model_path} for {len(group)} jobs", flush=True)
        model = Qwen3TTSModel.from_pretrained(
            model_path, device_map="cuda:0", dtype=torch.bfloat16,
            attn_implementation="sdpa",
        )
        for job in group.itertuples(index=False):
            sample_id = f"voice_fake_{generator}_{job.JOB_ID}"
            destination = args.pool_dir / "audio" / f"{sample_id}.flac"
            if destination.is_file():
                duration = sf.info(destination).duration
            else:
                digest = int(hashlib.sha256(
                    f"{args.seed}|{generator}|{job.JOB_ID}".encode()
                ).hexdigest()[:8], 16)
                torch.manual_seed(digest)
                speaker = args.custom_speaker if args.backend == "custom" else job.SPEAKER
                wavs, sample_rate = model.generate_custom_voice(
                    text=job.TEXT, language="Korean", speaker=speaker,
                )
                duration = write_clip(wavs[0], sample_rate, destination)
            rows.append(fake_row(job, sample_id, generator, duration))
            print(f"  {sample_id} {duration:.2f}s", flush=True)
        del model
        gc.collect()
        torch.cuda.empty_cache()

    args.output_manifest.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.output_manifest, index=False)
    print(f"Wrote {len(rows)} rows to {args.output_manifest}")


if __name__ == "__main__":
    main()
