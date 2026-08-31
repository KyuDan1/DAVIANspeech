#!/usr/bin/env python3
"""Render the shared speaker/content manifest with multilingual CosyVoice 3."""

from __future__ import annotations

import argparse
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
    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    if peak > 0.99:
        audio *= 0.99 / peak
    sf.write(destination, audio, SR, format="FLAC", subtype="PCM_16")
    return len(audio) / SR


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pool-dir", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output-manifest", type=Path, required=True)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260831)
    args = parser.parse_args()

    from cosyvoice.cli.cosyvoice import AutoModel
    import cosyvoice.cli.frontend as cosy_frontend

    # Recent torchaudio delegates decoding to torchcodec, which requires a
    # system FFmpeg shared-library installation.  Read the prompt directly so
    # generation remains independent of that optional runtime dependency.
    def load_prompt(path, target_sr):
        audio, sample_rate = sf.read(path, dtype="float32", always_2d=True)
        audio = audio.mean(axis=1)
        if sample_rate != target_sr:
            audio = librosa.resample(audio, orig_sr=sample_rate,
                                     target_sr=target_sr, res_type="soxr_hq")
        return torch.from_numpy(np.asarray(audio, dtype=np.float32)).unsqueeze(0)

    cosy_frontend.load_wav = load_prompt

    jobs = pd.read_csv(args.pool_dir / "generation_manifest.csv", dtype=str)
    speakers = sorted(jobs.SPEAKER.unique())
    selected = set(speakers[args.shard_index::args.num_shards])
    jobs = jobs[jobs.SPEAKER.isin(selected)]
    print(f"Loading {args.model_dir} for {len(jobs)} jobs", flush=True)
    model = AutoModel(model_dir=str(args.model_dir), fp16=True)
    generator = "cosyvoice3_0.5b"
    rows = []
    for job in jobs.itertuples(index=False):
        sample_id = f"voice_fake_{generator}_{job.JOB_ID}"
        destination = args.pool_dir / "audio" / f"{sample_id}.flac"
        if destination.is_file():
            duration = sf.info(destination).duration
        else:
            seed = int(hashlib.sha256(
                f"{args.seed}|{generator}|{job.JOB_ID}".encode()
            ).hexdigest()[:8], 16)
            torch.manual_seed(seed)
            outputs = list(model.inference_zero_shot(
                job.TEXT,
                "You are a helpful assistant.<|endofprompt|>" + job.REFERENCE_TEXT,
                job.REFERENCE_AUDIO,
                stream=False,
                text_frontend=False,
            ))
            if not outputs:
                raise RuntimeError(f"CosyVoice returned no audio for {job.JOB_ID}")
            waveform = torch.cat([item["tts_speech"].cpu() for item in outputs], dim=1)
            duration = write_clip(waveform.numpy(), model.sample_rate, destination)
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
