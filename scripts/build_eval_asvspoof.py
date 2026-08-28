"""Build a deterministic, competition-shaped speech diagnostic set.

Input is one or more parquet shards from the benchmark-ready ASVspoof 2021 DF
release. Only clips in the competition's 4--60 second duration range are used.
The result contains balanced bona-fide/spoof labels and keeps codec, source,
attack and vocoder metadata for breakdown reporting.
"""

from __future__ import annotations

import argparse
import io
import json
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq
import soundfile as sf

PREDICTION_COLUMNS = [
    "FILE_FAKE_PROB", "VOICE_FAKE_PROB", "MUSIC_FAKE_PROB",
    "VOICE_PRESENT_PROB", "MUSIC_PRESENT_PROB",
]


def read_candidates(paths):
    records = []
    for parquet_path in paths:
        table = pq.read_table(parquet_path, columns=["path", "audio", "label", "notes"])
        for row in table.to_pylist():
            audio_bytes = row["audio"]["bytes"]
            info = sf.info(io.BytesIO(audio_bytes))
            duration = info.frames / info.samplerate
            if not 4 <= duration <= 60:
                continue
            metadata = json.loads(row["notes"])
            records.append({
                "source_path": row["path"], "audio_bytes": audio_bytes,
                "FILE_FAKE": int(row["label"]), "DURATION": duration,
                "SOURCE": metadata["source"], "GENERATOR": metadata["attack_id"],
                "CODEC": metadata["codec"], "VOCODER": metadata["vocoder"],
                "SPEAKER": metadata["speaker_id"],
            })
    return records


def balanced_sample(records, per_class, seed):
    frame = pd.DataFrame(records)
    counts = frame.groupby("FILE_FAKE").size()
    if set(counts.index) != {0, 1} or counts.min() < per_class:
        raise ValueError(f"Need {per_class} clips per class, available: {counts.to_dict()}")
    # Sort first so sampling is stable even if parquet arguments change order.
    frame = frame.sort_values("source_path")
    selected = frame.groupby("FILE_FAKE", group_keys=False).sample(
        n=per_class, random_state=seed
    )
    return selected.sort_values(["FILE_FAKE", "source_path"]).reset_index(drop=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("parquet", type=Path, nargs="+")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--per-class", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260827)
    args = parser.parse_args()

    selected = balanced_sample(read_candidates(args.parquet), args.per_class, args.seed)
    audio_dir = args.output_dir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)

    truth_rows = []
    for index, row in selected.iterrows():
        sample_id = f"voice_{index:04d}"
        (audio_dir / f"{sample_id}.flac").write_bytes(row.pop("audio_bytes"))
        truth_rows.append({
            "ID": sample_id,
            "FILE_FAKE": row["FILE_FAKE"],
            "VOICE_FAKE": row["FILE_FAKE"],
            "MUSIC_FAKE": pd.NA,
            "VOICE_PRESENT": 1,
            "MUSIC_PRESENT": 0,
            "AUDIO_TYPE": "voice",
            "CONDITION": "speech_only",
            **{key: row[key] for key in [
                "SOURCE", "GENERATOR", "CODEC", "VOCODER", "SPEAKER", "DURATION"
            ]},
        })

    truth = pd.DataFrame(truth_rows)
    truth.to_csv(args.output_dir / "truth.csv", index=False)
    sample = pd.DataFrame({"ID": truth["ID"]})
    for column in PREDICTION_COLUMNS:
        sample[column] = 0.0
    sample.to_csv(args.output_dir / "sample_submission.csv", index=False)
    print(f"Built {len(truth)} clips at {args.output_dir}")
    print(truth.groupby(["FILE_FAKE", "CODEC"]).size().unstack(fill_value=0))


if __name__ == "__main__":
    main()
