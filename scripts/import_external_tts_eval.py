#!/usr/bin/env python3
"""Import Typecast/API TTS outputs into the shared evaluation manifest."""

from __future__ import annotations

import argparse
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
import soundfile as sf


SR = 16_000


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pool-dir", type=Path, required=True)
    parser.add_argument("--import-manifest", type=Path, required=True,
                        help="CSV with JOB_ID,AUDIO_FILE and optional GENERATOR")
    parser.add_argument("--output-manifest", type=Path, required=True)
    parser.add_argument("--default-generator", default="typecast")
    args = parser.parse_args()

    jobs = pd.read_csv(args.pool_dir / "generation_manifest.csv", dtype=str)
    imported = pd.read_csv(args.import_manifest, dtype=str)
    required = {"JOB_ID", "AUDIO_FILE"}
    if not required.issubset(imported):
        raise ValueError(f"Import manifest needs {sorted(required)}")
    if imported.JOB_ID.duplicated().any():
        raise ValueError("Duplicate JOB_ID in import manifest")
    unknown = set(imported.JOB_ID) - set(jobs.JOB_ID)
    if unknown:
        raise ValueError(f"Unknown JOB_ID values: {sorted(unknown)[:5]}")
    merged = imported.merge(jobs, on="JOB_ID", validate="one_to_one")
    rows = []
    for row in merged.itertuples(index=False):
        generator_value = getattr(row, "GENERATOR", None)
        generator = (args.default_generator if pd.isna(generator_value)
                     or not str(generator_value).strip() else str(generator_value))
        generator_slug = "".join(char if char.isalnum() else "_"
                                 for char in generator.lower()).strip("_")
        sample_id = f"voice_fake_{generator_slug}_{row.JOB_ID}"
        source = Path(row.AUDIO_FILE).expanduser().resolve()
        if not source.is_file():
            raise FileNotFoundError(source)
        audio, _ = librosa.load(source, sr=SR, mono=True, dtype=np.float32)
        if audio.size < 4 * SR:
            audio = np.pad(audio, (0, 4 * SR - audio.size))
        audio = audio[:60 * SR]
        peak = float(np.max(np.abs(audio))) if audio.size else 0.0
        if peak > 0.99:
            audio *= 0.99 / peak
        destination = args.pool_dir / "audio" / f"{sample_id}.flac"
        sf.write(destination, audio, SR, format="FLAC", subtype="PCM_16")
        rows.append({
            "ID": sample_id, "FILE_FAKE": 1, "VOICE_FAKE": 1,
            "MUSIC_FAKE": "", "VOICE_PRESENT": 1, "MUSIC_PRESENT": 0,
            "AUDIO_TYPE": "voice", "SOURCE": "external_tts",
            "GENERATOR": generator, "CONDITION": "speech_only",
            "SPLIT": row.SPLIT, "GROUP_ID": row.JOB_ID,
            "SPEAKER": row.SPEAKER, "CATEGORY": row.CATEGORY,
            "GENDER": row.GENDER, "AGE": row.AGE,
            "SOURCE_FILE": str(source), "DURATION": round(audio.size / SR, 3),
        })
    pd.DataFrame(rows).sort_values("ID").to_csv(args.output_manifest, index=False)
    print(f"Imported {len(rows)} files into {args.output_manifest}")


if __name__ == "__main__":
    main()
