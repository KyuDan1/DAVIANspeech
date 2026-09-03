"""Build a 1,200-file source-disjoint telephone-channel factorial audit set.

The same balanced Voice/Music/Mixed source files are passed through four call
channels.  Channel variants are therefore paired nuisance transformations,
not extra independent examples, and this set is never used for fitting.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import soundfile as sf
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from pipeline import find_audio_files, load_audio  # noqa: E402
from telephone_channel import apply_channel  # noqa: E402

VARIANTS = ("resample8k", "g711_ulaw", "g726_24k", "opus_nb_8k")


def balanced_rows(dataset: str, count_per_cell: int) -> pd.DataFrame:
    truth = pd.read_csv(ROOT / "data" / "eval" / dataset / "truth.csv",
                        dtype={"ID": str})
    if dataset == "source_disjoint_mixed_equal_v1":
        cells = ["VOICE_FAKE", "MUSIC_FAKE"]
    elif dataset == "source_disjoint_music_v1":
        cells = ["MUSIC_FAKE"]
    elif dataset == "asvspoof_voice_v1":
        cells = ["VOICE_FAKE"]
    else:
        raise ValueError(dataset)
    return pd.concat(
        [group.sort_values("ID").head(count_per_cell)
         for _, group in truth.groupby(cells, dropna=False, sort=True)],
        ignore_index=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir", type=Path,
        default=ROOT / "data" / "eval" / "phone_factorial_1200_v1",
    )
    parser.add_argument(
        "--ffmpeg", type=Path,
        default=ROOT.parent / "conda_envs" / "envs" / "davianspeech" / "bin" / "ffmpeg",
    )
    args = parser.parse_args()
    audio_dir = args.output_dir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)

    selections = [
        ("source_disjoint_mixed_equal_v1", balanced_rows(
            "source_disjoint_mixed_equal_v1", 25)),
        ("source_disjoint_music_v1", balanced_rows(
            "source_disjoint_music_v1", 50)),
        ("asvspoof_voice_v1", balanced_rows("asvspoof_voice_v1", 50)),
    ]
    # 100 mixed + 100 music + 100 voice, each under four paired channels.
    assert sum(len(frame) for _, frame in selections) == 300

    records = []
    for dataset, truth in selections:
        files = {path.stem: path for path in find_audio_files(
            ROOT / "data" / "eval" / dataset / "audio"
        )}
        for row in tqdm(truth.to_dict("records"), desc=dataset):
            audio = load_audio(files[str(row["ID"])])
            for variant in VARIANTS:
                sample_id = f"{row['ID']}__phonefact_{variant}"
                transformed = apply_channel(
                    audio, variant, ffmpeg=args.ffmpeg,
                    key=sum(sample_id.encode("utf-8")),
                )
                sf.write(audio_dir / f"{sample_id}.flac", transformed, 16_000,
                         subtype="PCM_16")
                record = dict(row)
                record.update({
                    "ID": sample_id, "PARENT_ID": str(row["ID"]),
                    "CHANNEL": variant, "SOURCE_DATASET": dataset,
                })
                records.append(record)

    result = pd.DataFrame(records)
    result.to_csv(args.output_dir / "truth.csv", index=False)
    submission = pd.DataFrame({"ID": result.ID})
    for column in (
        "FILE_FAKE_PROB", "VOICE_FAKE_PROB", "MUSIC_FAKE_PROB",
        "VOICE_PRESENT_PROB", "MUSIC_PRESENT_PROB",
    ):
        submission[column] = 0.5
    submission.to_csv(args.output_dir / "sample_submission.csv", index=False)
    print(result.groupby(["AUDIO_TYPE", "CHANNEL"]).size().unstack(fill_value=0))
    print(f"Saved {len(result)} rows to {args.output_dir}")


if __name__ == "__main__":
    main()
