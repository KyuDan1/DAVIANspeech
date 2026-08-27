"""Score an evaluation directory with ArtifactNet v9.4."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import librosa
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from artifactnet_detector import ArtifactNetMusicDetector  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audio-dir", type=Path, required=True)
    parser.add_argument("--sample-submission", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--provider", default="CPUExecutionProvider")
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    args = parser.parse_args()

    sample = pd.read_csv(args.sample_submission, dtype={"ID": str})
    if not 0 <= args.shard_index < args.num_shards:
        parser.error("--shard-index must satisfy 0 <= index < --num-shards")
    sample = sample.iloc[args.shard_index::args.num_shards].copy()
    paths = {path.stem: path for path in args.audio_dir.iterdir() if path.is_file()}
    detector = ArtifactNetMusicDetector(args.model_dir, providers=[args.provider])
    scores = []
    for sample_id in tqdm(sample.ID, desc="artifactnet"):
        audio, _ = librosa.load(paths[sample_id], sr=16_000, mono=True, dtype="float32")
        scores.append(detector.fake_probability(audio))
    sample["FILE_FAKE_PROB"] = scores
    sample["VOICE_FAKE_PROB"] = 0.0
    sample["MUSIC_FAKE_PROB"] = scores
    sample["VOICE_PRESENT_PROB"] = 0.0
    sample["MUSIC_PRESENT_PROB"] = 1.0
    args.output.parent.mkdir(parents=True, exist_ok=True)
    sample.to_csv(args.output, index=False, lineterminator="\r\n")
    print(f"Saved {len(sample)} predictions to {args.output}")


if __name__ == "__main__":
    main()
