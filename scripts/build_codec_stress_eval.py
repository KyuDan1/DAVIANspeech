"""Create deterministic codec/channel variants of an existing audio eval set."""

from __future__ import annotations

import argparse
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
import soundfile as sf

import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from data_guard import identity_tokens  # noqa: E402


VARIANTS = ("wav", "flac", "mp3", "ogg", "telephone8k")
EXTENSIONS = {"wav": ".wav", "flac": ".flac", "mp3": ".mp3", "ogg": ".ogg", "telephone8k": ".wav"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--truth", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--variants", nargs="+", choices=VARIANTS, default=list(VARIANTS))
    parser.add_argument("--exclude-truth", type=Path, nargs="+", default=[])
    args = parser.parse_args()
    truth = pd.read_csv(args.truth, dtype={"ID": str})
    excluded = set()
    for path in args.exclude_truth:
        excluded.update(identity_tokens(pd.read_csv(path, dtype=str)))
    if excluded:
        truth = truth[~truth.ID.astype(str).isin(excluded)].copy()
    source = {path.stem: path for path in args.input_dir.iterdir() if path.is_file()}
    audio_dir = args.output_dir / "audio"; audio_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for row in truth.to_dict("records"):
        audio, _ = librosa.load(source[row["ID"]], sr=16_000, mono=True, dtype=np.float32)
        for variant in args.variants:
            sample_id = f"{row['ID']}__{variant}"
            destination = audio_dir / f"{sample_id}{EXTENSIONS[variant]}"
            transformed = audio
            if variant == "telephone8k":
                transformed = librosa.resample(audio, orig_sr=16_000, target_sr=8_000)
                transformed = librosa.resample(transformed, orig_sr=8_000, target_sr=16_000)
            format_name = {"mp3": "MP3", "ogg": "OGG"}.get(variant)
            sf.write(destination, transformed, 16_000, format=format_name)
            rows.append({**row, "ID": sample_id, "PARENT_ID": row["ID"], "STRESS_VARIANT": variant})
    args.output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.output_dir / "truth.csv", index=False)
    sample = pd.DataFrame({"ID": [row["ID"] for row in rows]})
    for column in ("FILE_FAKE_PROB", "VOICE_FAKE_PROB", "MUSIC_FAKE_PROB",
                   "VOICE_PRESENT_PROB", "MUSIC_PRESENT_PROB"):
        sample[column] = 0.5
    sample.to_csv(args.output_dir / "sample_submission.csv", index=False)
    print(f"Created {len(rows)} variants in {audio_dir}")


if __name__ == "__main__":
    main()
