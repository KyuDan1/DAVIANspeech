"""Materialize a compact, balanced EchoFake open-set speech stress test."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq


PREDICTION_COLUMNS = [
    "FILE_FAKE_PROB", "VOICE_FAKE_PROB", "MUSIC_FAKE_PROB",
    "VOICE_PRESENT_PROB", "MUSIC_PRESENT_PROB",
]
FAKE_LABELS = {"fake", "replay_fake"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parquet", type=Path, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--per-condition", type=int, default=200)
    parser.add_argument("--seed", type=int, default=20260828)
    args = parser.parse_args()

    if args.output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output_dir}")
    audio_dir = args.output_dir / "audio"
    audio_dir.mkdir(parents=True)

    tables = [pq.read_table(path) for path in args.parquet]
    frame = pd.concat([table.to_pandas() for table in tables], ignore_index=True)
    records = []
    for condition, block in frame.groupby("label", sort=True):
        if len(block) < args.per_condition:
            raise ValueError(f"Only {len(block)} rows for {condition}")
        selected = block.sample(n=args.per_condition, random_state=args.seed)
        for _, row in selected.iterrows():
            sample_id = f"echofake_{condition}_{len(records):04d}"
            suffix = Path(row["path"]["path"]).suffix.lower() or ".wav"
            destination = audio_dir / f"{sample_id}{suffix}"
            destination.write_bytes(row["path"]["bytes"])
            synthesis = row["synthesis_details"]
            replay = row["replay_details"]
            is_fake = int(condition in FAKE_LABELS)
            records.append({
                "ID": sample_id, "FILE_FAKE": is_fake,
                "VOICE_FAKE": is_fake, "MUSIC_FAKE": pd.NA,
                "VOICE_PRESENT": 1, "MUSIC_PRESENT": 0,
                "AUDIO_TYPE": "voice", "SOURCE": "EchoFake-open-set",
                "CONDITION": condition, "GENERATOR": synthesis.get("model"),
                "SOURCE_ID": row["source"], "SPEAKER_ID": row["source_speaker_id"],
                "PLAYER": replay.get("player"), "RECORDER": replay.get("recorder"),
                "DISTANCE": replay.get("distance"),
            })

    truth = pd.DataFrame(records)
    truth.to_csv(args.output_dir / "truth.csv", index=False)
    submission = pd.DataFrame({"ID": truth.ID})
    for column in PREDICTION_COLUMNS:
        submission[column] = 0.5
    submission.to_csv(args.output_dir / "sample_submission.csv", index=False)
    print(truth.groupby(["VOICE_FAKE", "CONDITION"]).size().to_string())
    print(f"Wrote {len(truth)} files to {args.output_dir}")


if __name__ == "__main__":
    main()
