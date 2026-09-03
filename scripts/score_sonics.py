"""Score a directory with the local dependency-free SONICS implementation."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pipeline import load_audio  # noqa: E402
from sonics_detector import SonicsMusicDetector  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audio-dir", type=Path, required=True)
    parser.add_argument("--sample-submission", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    sample = pd.read_csv(args.sample_submission, dtype={"ID": str})
    paths = {path.stem: path for path in args.audio_dir.iterdir() if path.is_file()}
    missing = [sample_id for sample_id in sample.ID if sample_id not in paths]
    if missing:
        raise ValueError(f"Missing audio for {missing[:5]}")
    model = SonicsMusicDetector.from_checkpoint(args.model_dir, device=args.device)

    probabilities = []
    for sample_id in tqdm(sample.ID, desc="SONICS"):
        probabilities.append(model.fake_probability(
            load_audio(paths[sample_id]), device=args.device
        ))
    sample["MUSIC_FAKE_PROB"] = probabilities
    sample["FILE_FAKE_PROB"] = probabilities
    sample["VOICE_FAKE_PROB"] = 0.0
    sample["VOICE_PRESENT_PROB"] = 0.0
    sample["MUSIC_PRESENT_PROB"] = 1.0
    args.output.parent.mkdir(parents=True, exist_ok=True)
    sample.to_csv(args.output, index=False, lineterminator="\r\n")
    print(f"Saved {len(sample)} predictions to {args.output}")


if __name__ == "__main__":
    main()
