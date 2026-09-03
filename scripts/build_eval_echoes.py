"""Build an external, generator-diverse music holdout from Echoes.

Echoes contains generated tracks only, so the bona-fide side is taken from the
MusicCaps half of our semantic-pair suite.  This is intentionally an external
stress test: none of the Echoes providers are used by ``build_eval_musiccaps``.
The selected archive members and labels are written to a fixed manifest so the
same files are used in every experiment.
"""

from __future__ import annotations

import argparse
import shutil
import zipfile
from pathlib import Path

import pandas as pd


PREDICTION_COLUMNS = [
    "FILE_FAKE_PROB", "VOICE_FAKE_PROB", "MUSIC_FAKE_PROB",
    "VOICE_PRESENT_PROB", "MUSIC_PRESENT_PROB",
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--echoes-zip", type=Path, required=True)
    parser.add_argument("--real-suite", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--fake-per-generator", type=int, default=50)
    parser.add_argument("--real-count", type=int, default=200)
    parser.add_argument("--seed", type=int, default=20260828)
    args = parser.parse_args()

    if args.output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output_dir}")
    audio_dir = args.output_dir / "audio"
    audio_dir.mkdir(parents=True)

    with zipfile.ZipFile(args.echoes_zip) as archive:
        with archive.open("Echoes/dataset_manifest.csv") as handle:
            manifest = pd.read_csv(handle)

        fake_parts = []
        for generator, block in manifest.groupby("generator", sort=True):
            if len(block) < args.fake_per_generator:
                raise ValueError(f"Only {len(block)} tracks for {generator}")
            # Preserve both text- and audio-conditioned attacks where present.
            selected_types = []
            for _, typed in block.groupby("type"):
                count = max(
                    1, round(args.fake_per_generator * len(typed) / len(block))
                )
                selected_types.append(
                    typed.sample(n=count, random_state=args.seed)
                )
            sampled = pd.concat(selected_types)
            sampled = sampled.head(args.fake_per_generator)
            if len(sampled) < args.fake_per_generator:
                remaining = block.drop(sampled.index).sample(
                    n=args.fake_per_generator - len(sampled), random_state=args.seed
                )
                sampled = pd.concat([sampled, remaining])
            sampled["generator"] = generator
            fake_parts.append(sampled)
        fakes = pd.concat(fake_parts, ignore_index=True)

        records = []
        for index, row in fakes.iterrows():
            sample_id = f"echoes_fake_{index:04d}"
            member = "Echoes/" + row.path_in_dataset
            suffix = Path(row.path_in_dataset).suffix.lower()
            destination = audio_dir / f"{sample_id}{suffix}"
            with archive.open(member) as source, destination.open("wb") as target:
                shutil.copyfileobj(source, target)
            records.append({
                "ID": sample_id, "FILE_FAKE": 1, "VOICE_FAKE": pd.NA,
                "MUSIC_FAKE": 1, "VOICE_PRESENT": 0, "MUSIC_PRESENT": 1,
                "AUDIO_TYPE": "music", "SOURCE": "Echoes",
                "GENERATOR": row["generator"],
                "GENERATION_TYPE": row["type"], "GENRE": row["genre"],
                "GROUP_ID": row["original_audio"], "DURATION": row["duration"],
            })

    real_truth = pd.read_csv(args.real_suite / "truth.csv", dtype={"ID": str})
    reals = real_truth[real_truth.FILE_FAKE == 0].sample(
        n=args.real_count, random_state=args.seed
    )
    for index, row in enumerate(reals.itertuples(index=False)):
        source_matches = list((args.real_suite / "audio").glob(f"{row.ID}.*"))
        if len(source_matches) != 1:
            raise ValueError(f"Expected one real source for {row.ID}")
        sample_id = f"echoes_real_{index:04d}"
        destination = audio_dir / f"{sample_id}{source_matches[0].suffix.lower()}"
        shutil.copy2(source_matches[0], destination)
        records.append({
            "ID": sample_id, "FILE_FAKE": 0, "VOICE_FAKE": pd.NA,
            "MUSIC_FAKE": 0, "VOICE_PRESENT": 0, "MUSIC_PRESENT": 1,
            "AUDIO_TYPE": "music", "SOURCE": "MusicCaps",
            "GENERATOR": "real", "GENERATION_TYPE": "real",
            "GENRE": pd.NA, "GROUP_ID": row.PAIR_ID, "DURATION": row.DURATION,
        })

    truth = pd.DataFrame(records)
    truth.to_csv(args.output_dir / "truth.csv", index=False)
    submission = pd.DataFrame({"ID": truth.ID})
    for column in PREDICTION_COLUMNS:
        submission[column] = 0.5
    submission.to_csv(args.output_dir / "sample_submission.csv", index=False)
    print(truth.groupby(["FILE_FAKE", "GENERATOR"]).size().to_string())
    print(f"Wrote {len(truth)} files to {args.output_dir}")


if __name__ == "__main__":
    main()
