"""Build semantically paired real/fake music diagnostics from FakeMusicCaps.

FakeMusicCaps regenerates MusicCaps captions under five open text-to-music
models and names each output with the source MusicCaps YouTube ID. This script
range-reads only selected members of the 12.9 GB archive, downloads the paired
real 10-second MusicCaps segment, and transcodes both sides identically.

The datasets are research-only/CC BY-NC; review their terms before use.
"""

from __future__ import annotations

import argparse
import random
import subprocess
import tempfile
from pathlib import Path

import pandas as pd
from remotezip import RemoteZip

ARCHIVE_URL = "https://zenodo.org/api/records/15063698/files/FakeMusicCaps.zip/content"
GENERATORS = ["audioldm2", "MusicGen_medium", "musicldm", "mustango", "stable_audio_open"]
PREDICTION_COLUMNS = [
    "FILE_FAKE_PROB", "VOICE_FAKE_PROB", "MUSIC_FAKE_PROB",
    "VOICE_PRESENT_PROB", "MUSIC_PRESENT_PROB",
]


def run(command):
    subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def transcode(source: Path, destination: Path):
    run([
        "ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-y",
        "-i", str(source), "-ac", "1", "-ar", "16000", str(destination),
    ])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--musiccaps-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--pairs", type=int, default=50)
    parser.add_argument("--seed", type=int, default=20260827)
    args = parser.parse_args()

    metadata = pd.read_csv(args.musiccaps_csv).set_index("ytid")
    audio_dir = args.output_dir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    truth_rows = []

    with RemoteZip(ARCHIVE_URL) as archive:
        members = set(archive.namelist())
        candidates = sorted({
            Path(name).stem for name in members
            if name.endswith(".wav") and not name.startswith("__MACOSX/")
            and Path(name).stem in metadata.index
        })
        random.Random(args.seed).shuffle(candidates)

        with tempfile.TemporaryDirectory(prefix="musiccaps_eval_") as temporary:
            temporary = Path(temporary)
            completed = 0
            for ytid in candidates:
                if completed >= args.pairs:
                    break
                generator = GENERATORS[completed % len(GENERATORS)]
                member = f"{generator}/{ytid}.wav"
                if member not in members:
                    continue
                row = metadata.loc[ytid]
                real_raw = temporary / f"{ytid}_real.%(ext)s"
                fake_raw = temporary / f"{ytid}_fake.wav"
                try:
                    run([
                        "yt-dlp", "--no-playlist", "--quiet", "--no-warnings",
                        "-f", "bestaudio/best", "--download-sections",
                        f"*{row.start_s}-{row.end_s}", "-o", str(real_raw),
                        f"https://www.youtube.com/watch?v={ytid}",
                    ])
                    downloaded = next(temporary.glob(f"{ytid}_real.*"))
                    fake_raw.write_bytes(archive.read(member))
                    pair_id = f"music_{completed:04d}"
                    real_id, fake_id = f"{pair_id}_real", f"{pair_id}_fake"
                    transcode(downloaded, audio_dir / f"{real_id}.flac")
                    transcode(fake_raw, audio_dir / f"{fake_id}.flac")
                except (subprocess.CalledProcessError, StopIteration, OSError) as error:
                    print(f"skip {ytid}: {error}", flush=True)
                    continue

                common = {
                    "VOICE_FAKE": pd.NA, "VOICE_PRESENT": 0,
                    "MUSIC_PRESENT": 1, "AUDIO_TYPE": "music",
                    "CONDITION": "semantic_pair", "PAIR_ID": pair_id,
                    "SOURCE": "MusicCaps/FakeMusicCaps", "CODEC": "flac16k",
                    "DURATION": float(row.end_s - row.start_s),
                }
                truth_rows.extend([
                    {"ID": real_id, "FILE_FAKE": 0, "MUSIC_FAKE": 0,
                     "GENERATOR": "real", **common},
                    {"ID": fake_id, "FILE_FAKE": 1, "MUSIC_FAKE": 1,
                     "GENERATOR": generator, **common},
                ])
                completed += 1
                print(f"pair {completed}/{args.pairs}: {ytid} ({generator})", flush=True)

    if completed < args.pairs:
        raise RuntimeError(f"Only built {completed}/{args.pairs} requested pairs")
    truth = pd.DataFrame(truth_rows)
    truth.to_csv(args.output_dir / "truth.csv", index=False)
    sample = pd.DataFrame({"ID": truth["ID"]})
    for column in PREDICTION_COLUMNS:
        sample[column] = 0.0
    sample.to_csv(args.output_dir / "sample_submission.csv", index=False)
    print(f"Built {len(truth)} clips at {args.output_dir}")


if __name__ == "__main__":
    main()
