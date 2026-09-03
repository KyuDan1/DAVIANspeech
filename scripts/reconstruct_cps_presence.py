#!/usr/bin/env python3
"""Reconstruct the deployed v13+ CPS scores from cached EAT statistics."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from eat_presence_fusion import (  # noqa: E402
    latent_linear_probability,
    logit_fuse_presence,
)
from pipeline import find_audio_files, load_audio  # noqa: E402
from telephone_router import TelephoneRouter  # noqa: E402


def read_many(paths: list[Path]) -> pd.DataFrame:
    frame = pd.concat(pd.read_csv(path, dtype={"ID": str}) for path in paths)
    if frame.ID.duplicated().any():
        raise ValueError("Duplicate IDs in CSV shards")
    return frame


def read_statistics(paths: list[Path]) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    result = {}
    for path in paths:
        shard = np.load(path, allow_pickle=False)
        for identifier, matrix, mask in zip(
            shard["ids"].astype(str), shard["statistics"], shard["view_mask"]
        ):
            if identifier in result:
                raise ValueError(f"Duplicate EAT statistics for {identifier}")
            result[identifier] = (matrix, mask)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--anchor", type=Path, required=True)
    parser.add_argument("--eat-presence", type=Path, nargs="+", required=True)
    parser.add_argument("--music-probe", type=Path, required=True)
    parser.add_argument("--eat-stats", type=Path, nargs="+", required=True)
    parser.add_argument("--audio-dir", type=Path, required=True)
    parser.add_argument("--phone-head", type=Path, required=True)
    parser.add_argument("--telephone-router", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--phone-weight", type=float, default=0.10)
    args = parser.parse_args()

    anchor = pd.read_csv(args.anchor, dtype={"ID": str})
    eat = read_many(args.eat_presence)
    probe = pd.read_csv(args.music_probe, dtype={"ID": str})[
        ["ID", "PROBE_MUSIC_PRESENT_PROB"]
    ]
    frame = anchor.merge(eat, on="ID", validate="one_to_one").merge(
        probe, on="ID", validate="one_to_one"
    )
    stats = read_statistics(args.eat_stats)
    checkpoint = np.load(args.phone_head, allow_pickle=False)
    router = TelephoneRouter(args.telephone_router)
    audio_paths = {path.stem: path for path in find_audio_files(args.audio_dir)}

    frame["VOICE_PRESENT_PROB"] = (
        0.65 * frame.VOICE_PRESENT_PROB
        + 0.35 * frame.EAT_VOICE_PRESENT_PROB
    )
    base_music = (
        0.10 * frame.MUSIC_PRESENT_PROB
        + 0.90 * frame.EAT_MUSIC_PRESENT_PROB
    )
    frame["MUSIC_PRESENT_PROB"] = (
        0.60 * base_music + 0.40 * frame.PROBE_MUSIC_PRESENT_PROB
    )

    routed = 0
    for index, row in frame.iterrows():
        identifier = row.ID
        path = audio_paths.get(identifier)
        if path is None:
            raise FileNotFoundError(f"Missing audio for {identifier}")
        audio = load_audio(path)
        if not router.is_narrowband(audio):
            continue
        matrix, mask = stats[identifier]
        phone_voice = latent_linear_probability(matrix, mask, checkpoint)
        frame.at[index, "VOICE_PRESENT_PROB"] = logit_fuse_presence(
            row.VOICE_PRESENT_PROB, phone_voice, args.phone_weight
        )
        routed += 1

    columns = [
        "ID", "FILE_FAKE_PROB", "VOICE_FAKE_PROB", "MUSIC_FAKE_PROB",
        "VOICE_PRESENT_PROB", "MUSIC_PRESENT_PROB",
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    frame[columns].to_csv(args.output, index=False)
    print(f"Wrote {len(frame)} rows; telephone routed {routed}/{len(frame)}")


if __name__ == "__main__":
    main()
