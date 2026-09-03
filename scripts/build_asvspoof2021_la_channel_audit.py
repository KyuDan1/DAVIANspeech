#!/usr/bin/env python3
"""Build a codec-balanced ASVspoof2021-LA channel audit subset.

ASVspoof2021-LA was included in AntiDeepfake XLS-R post-training, so this bank
must not be reported as an unseen authenticity benchmark.  Its purpose is to
audit the telephone router and the relative stability of downstream heads on
real PSTN/VoIP transmissions with official codec metadata.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil
import tarfile

import pandas as pd
import soundfile as sf


CODECS = ("alaw", "ulaw", "gsm", "pstn", "g722", "opus")
COLUMNS = (
    "SPEAKER", "SOURCE_ID", "CODEC", "TRANSMISSION", "ATTACK",
    "LABEL", "TRIM", "PARTITION",
)


def select_balanced_attack(group: pd.DataFrame, count: int, seed: int) -> pd.DataFrame:
    if len(group) < count:
        raise ValueError(f"Cannot select {count} rows from {len(group)}")
    if set(group.LABEL) != {"spoof"}:
        return group.sample(n=count, random_state=seed)
    attacks = sorted(group.ATTACK.unique())
    shuffled = {
        attack: group.loc[group.ATTACK.eq(attack)].sample(
            frac=1, random_state=seed + index
        ).reset_index(drop=True)
        for index, attack in enumerate(attacks)
    }
    offsets = {attack: 0 for attack in attacks}
    pieces: list[pd.DataFrame] = []
    selected = 0
    # Round-robin selection is exactly balanced when possible and degrades
    # gracefully when duration filtering leaves one legacy attack sparse.
    while selected < count:
        progressed = False
        for attack in attacks:
            offset = offsets[attack]
            if offset >= len(shuffled[attack]):
                continue
            pieces.append(shuffled[attack].iloc[[offset]])
            offsets[attack] += 1
            selected += 1
            progressed = True
            if selected == count:
                break
        if not progressed:
            raise RuntimeError("Attack-balanced selection exhausted unexpectedly")
    return pd.concat(pieces, ignore_index=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audio-root", type=Path, required=True)
    parser.add_argument(
        "--archive", type=Path,
        help="Optionally extract only balanced candidates from this tar.gz first.",
    )
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--per-label-codec", type=int, default=100)
    parser.add_argument("--candidate-multiplier", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260903)
    args = parser.parse_args()
    if args.candidate_multiplier < 1:
        parser.error("--candidate-multiplier must be positive")
    if args.output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output_dir}")

    metadata = pd.read_csv(
        args.metadata, sep=r"\s+", names=COLUMNS, dtype=str,
    )
    metadata = metadata.loc[
        metadata.PARTITION.eq("eval") & metadata.CODEC.isin(CODECS)
    ].copy()
    if args.archive is not None:
        args.audio_root.mkdir(parents=True, exist_ok=True)
        candidate_pieces = []
        for codec_index, codec in enumerate(CODECS):
            for label_index, label in enumerate(("bonafide", "spoof")):
                group = metadata.loc[
                    metadata.CODEC.eq(codec) & metadata.LABEL.eq(label)
                ]
                candidate_pieces.append(select_balanced_attack(
                    group, args.per_label_codec * args.candidate_multiplier,
                    args.seed + 10_000 + codec_index * 100 + label_index,
                ))
        candidate_ids = set(pd.concat(candidate_pieces).SOURCE_ID)
        existing_ids = {path.stem for path in args.audio_root.glob("*.flac")}
        needed = candidate_ids - existing_ids
        if needed:
            with tarfile.open(args.archive, "r:gz") as archive:
                for member in archive:
                    sample_id = Path(member.name).stem
                    if not member.isfile() or sample_id not in needed:
                        continue
                    source = archive.extractfile(member)
                    if source is None:
                        raise RuntimeError(f"Could not read archive member {member.name}")
                    destination = args.audio_root / f"{sample_id}.flac"
                    with source, destination.open("wb") as handle:
                        shutil.copyfileobj(source, handle, length=8 * 1024 * 1024)
                    needed.remove(sample_id)
                    if not needed:
                        break
        if needed:
            raise RuntimeError(f"Archive is missing {len(needed)} selected files")
        print(f"Extracted {len(candidate_ids)} balanced candidates", flush=True)
    audio_by_id = {
        path.stem: path for path in args.audio_root.rglob("*.flac")
    }
    metadata = metadata.loc[metadata.SOURCE_ID.isin(audio_by_id)].copy()
    if metadata.empty:
        raise ValueError(f"No metadata audio found below {args.audio_root}")

    # Filter to the competition duration range before balanced selection.
    duration = {}
    for sample_id in metadata.SOURCE_ID:
        info = sf.info(audio_by_id[sample_id])
        duration[sample_id] = info.frames / info.samplerate
    metadata["DURATION"] = metadata.SOURCE_ID.map(duration)
    metadata = metadata.loc[metadata.DURATION.between(4.0, 60.0)].copy()

    pieces = []
    for codec_index, codec in enumerate(CODECS):
        for label_index, label in enumerate(("bonafide", "spoof")):
            group = metadata.loc[
                metadata.CODEC.eq(codec) & metadata.LABEL.eq(label)
            ]
            if len(group) < args.per_label_codec:
                raise ValueError(f"Too few {codec}/{label}: {len(group)}")
            pieces.append(select_balanced_attack(
                group, args.per_label_codec,
                args.seed + codec_index * 100 + label_index,
            ))
    selected = pd.concat(pieces, ignore_index=True).sort_values(
        ["CODEC", "LABEL", "SOURCE_ID"]
    ).reset_index(drop=True)

    audio_dir = args.output_dir / "audio"
    audio_dir.mkdir(parents=True)
    for row in selected.itertuples(index=False):
        os.symlink(audio_by_id[row.SOURCE_ID].resolve(), audio_dir / f"{row.SOURCE_ID}.flac")

    selected["ID"] = selected.SOURCE_ID
    selected["FILE_FAKE"] = selected.LABEL.eq("spoof").astype(int)
    selected["VOICE_FAKE"] = selected.FILE_FAKE
    selected["MUSIC_FAKE"] = pd.NA
    selected["VOICE_PRESENT"] = 1
    selected["MUSIC_PRESENT"] = 0
    selected["AUDIO_TYPE"] = "voice"
    selected["ROLE"] = "channel_stress_only_xlsr_pretraining_overlap"
    truth_columns = [
        "ID", "FILE_FAKE", "VOICE_FAKE", "MUSIC_FAKE",
        "VOICE_PRESENT", "MUSIC_PRESENT", "AUDIO_TYPE", "CODEC",
        "TRANSMISSION", "ATTACK", "SPEAKER", "DURATION", "ROLE",
    ]
    selected[truth_columns].to_csv(args.output_dir / "truth.csv", index=False)
    submission = pd.DataFrame({"ID": selected.ID})
    for column in (
        "FILE_FAKE_PROB", "VOICE_FAKE_PROB", "MUSIC_FAKE_PROB",
        "VOICE_PRESENT_PROB", "MUSIC_PRESENT_PROB",
    ):
        submission[column] = 0.5
    submission.to_csv(args.output_dir / "sample_submission.csv", index=False)
    print(selected.groupby(["CODEC", "LABEL"]).size().unstack(fill_value=0))
    print("attacks", selected.loc[selected.LABEL.eq("spoof"), "ATTACK"].value_counts().to_dict())
    print(f"Built {len(selected)} channel-audit files in {args.output_dir}")


if __name__ == "__main__":
    main()
