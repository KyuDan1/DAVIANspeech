"""Build a Korean voice-deepfake eval set from FLEURS ko_kr plus synthetic fakes.

The existing voice eval is ASVspoof, which XLS-R-2B-AntiDeepfake was post-trained
on -- measuring it there measures its training distribution. This set is Korean,
which is what the competition audio actually is, and pairs each fake with the real
utterance it was generated from so the detector cannot separate them on content.

Fakes are produced by generator scripts that write into
``<pool>/fake/<generator>/<ID>.wav``; this script only assembles what it finds, so
adding a generator means dropping in another directory.

    python scripts/build_eval_korean.py --pool korean_eval/pool \
        --output-dir korean_eval/sets/korean_v1
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf

PREDICTION_COLUMNS = [
    "FILE_FAKE_PROB", "VOICE_FAKE_PROB", "MUSIC_FAKE_PROB",
    "VOICE_PRESENT_PROB", "MUSIC_PRESENT_PROB",
]

# Every real utterance and all fakes derived from it share a split, so a
# generator never sees the same source sentence on both sides of a boundary.
SPLITS = (("calibration", 0.34), ("validation", 0.33), ("holdout", 0.33))


def assign_splits(source_ids, seed: int) -> dict[str, str]:
    ordered = sorted(set(source_ids))
    rng = np.random.default_rng(seed)
    rng.shuffle(ordered)
    out, start = {}, 0
    for index, (name, fraction) in enumerate(SPLITS):
        end = len(ordered) if index == len(SPLITS) - 1 else start + int(len(ordered) * fraction)
        for source_id in ordered[start:end]:
            out[source_id] = name
        start = end
    return out


def probe(path: Path) -> tuple[float, int]:
    info = sf.info(path)
    return float(info.duration), int(info.samplerate)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pool", type=Path, required=True,
                        help="Directory holding real/ , fake/<generator>/ and meta.json")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--min-duration", type=float, default=4.0)
    parser.add_argument("--max-duration", type=float, default=60.0)
    args = parser.parse_args()

    meta = {m["id"]: m for m in json.loads((args.pool / "meta.json").read_text("utf-8"))}
    splits = assign_splits(meta.keys(), args.seed)

    rows = []
    for path in sorted((args.pool / "real").glob("*.wav")):
        source_id = path.stem
        if source_id not in meta:
            continue
        duration, sample_rate = probe(path)
        rows.append({
            "ID": f"{source_id}_real", "source_path": str(path.resolve()),
            "FILE_FAKE": 0, "VOICE_FAKE": 0, "MUSIC_FAKE": pd.NA,
            "VOICE_PRESENT": 1, "MUSIC_PRESENT": 0,
            "AUDIO_TYPE": "voice", "SOURCE": "fleurs_ko", "GENERATOR": "bonafide",
            "CODEC": "none", "CHANNEL": "clean", "FORMAT": "wav",
            "CONDITION": "korean_read", "SPLIT": splits[source_id],
            "SOURCE_ID": source_id, "SPEAKER_GENDER": meta[source_id]["gender"],
            "DURATION": round(duration, 2), "SAMPLE_RATE": sample_rate,
        })

    fake_root = args.pool / "fake"
    for generator_dir in sorted(p for p in fake_root.glob("*") if p.is_dir()):
        generator = generator_dir.name
        for path in sorted(generator_dir.glob("*.wav")):
            source_id = path.stem
            if source_id not in meta:
                continue
            duration, sample_rate = probe(path)
            if not args.min_duration <= duration <= args.max_duration:
                continue
            rows.append({
                "ID": f"{source_id}_{generator}", "source_path": str(path.resolve()),
                "FILE_FAKE": 1, "VOICE_FAKE": 1, "MUSIC_FAKE": pd.NA,
                "VOICE_PRESENT": 1, "MUSIC_PRESENT": 0,
                "AUDIO_TYPE": "voice", "SOURCE": "fleurs_ko", "GENERATOR": generator,
                "CODEC": "none", "CHANNEL": "clean", "FORMAT": "wav",
                "CONDITION": "korean_read", "SPLIT": splits[source_id],
                "SOURCE_ID": source_id, "SPEAKER_GENDER": meta[source_id]["gender"],
                "DURATION": round(duration, 2), "SAMPLE_RATE": sample_rate,
            })

    truth = pd.DataFrame(rows)
    if truth.empty:
        raise SystemExit(f"No audio found under {args.pool}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    truth.to_csv(args.output_dir / "truth.csv", index=False)

    sample = pd.DataFrame({"ID": truth["ID"]})
    for column in PREDICTION_COLUMNS:
        sample[column] = 0.5
    sample.to_csv(args.output_dir / "sample_submission.csv", index=False)

    print(f"Built {len(truth)} clips at {args.output_dir}")
    print(truth.groupby(["GENERATOR", "SPLIT"]).size().unstack(fill_value=0).to_string())
    print(f"\nreal={int((truth.FILE_FAKE == 0).sum())}  fake={int((truth.FILE_FAKE == 1).sum())}")


if __name__ == "__main__":
    main()
