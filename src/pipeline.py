"""DAVIANspeech AI-generated audio detection pipeline.

    INPUT AUDIO
    |
    +-- PANNs Cnn14 ------> VOICE_PRESENT_PROB (VP), MUSIC_PRESENT_PROB (MP)
    |
    +-- Separator --------> voice stem  --> XLS-R-2B --> VOICE_FAKE_PROB (VF)
        (HTDemucs |         music stem  --> XLS-R-2B -----+--> MUSIC_FAKE_PROB (MF)
         SAM-Audio)         full audio  --> ArtifactNet --+

    FILE_FAKE_PROB = max(fake scores for components whose presence >= 0.7)

Relative to the competition baseline the presence stage is untouched, while
DF-Arena 1B is replaced by XLS-R-2B-AntiDeepfake and HTDemucs is optionally
replaced by SAM-Audio.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import librosa
import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))

from presence import PannsPresence, extract_segment, segment_starts  # noqa: E402
from separation import build_separator  # noqa: E402
from artifactnet_detector import ArtifactNetMusicDetector  # noqa: E402
from xlsr_antideepfake import XlsrAntiDeepfake  # noqa: E402

AUDIO_SR = 16_000
SILENCE_RMS = 1e-5
PRESENCE_GATE = 0.7
XLSR_MUSIC_WEIGHT = 0.5
ARTIFACTNET_WEIGHT = 0.5

PREDICTION_COLUMNS = [
    "FILE_FAKE_PROB",
    "VOICE_FAKE_PROB",
    "MUSIC_FAKE_PROB",
    "VOICE_PRESENT_PROB",
    "MUSIC_PRESENT_PROB",
]

AUDIO_EXTENSIONS = {".aac", ".flac", ".m4a", ".mp3", ".ogg", ".opus", ".wav", ".wma"}


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------

def find_audio_files(test_dir: Path) -> list[Path]:
    if not test_dir.is_dir():
        raise FileNotFoundError(f"Test directory not found: {test_dir}")
    files = sorted(
        (p for p in test_dir.iterdir()
         if p.is_file() and p.suffix.lower() in AUDIO_EXTENSIONS),
        key=lambda p: p.stem,
    )
    if not files:
        raise FileNotFoundError(f"No audio files found in {test_dir}")
    if len({p.stem for p in files}) != len(files):
        raise ValueError("Audio IDs must be unique")
    return files


def read_sample_submission(csv_path: Path):
    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        columns, rows = reader.fieldnames, list(reader)
    if not columns or not rows:
        raise ValueError(f"Invalid sample submission: {csv_path}")
    missing = [c for c in ["ID", *PREDICTION_COLUMNS] if c not in columns]
    if missing:
        raise ValueError(f"Sample submission is missing columns: {missing}")
    for row in rows:
        row["ID"] = str(row["ID"]).strip()
    return columns, rows


def order_by_submission(audio_files, rows):
    by_id = {p.stem: p for p in audio_files}
    ids = [r["ID"] for r in rows]
    missing = [i for i in ids if i not in by_id]
    extra = [i for i in by_id if i not in set(ids)]
    if missing or extra:
        raise ValueError(
            f"Test audio and submission IDs disagree. "
            f"Missing: {missing[:5]}, Extra: {extra[:5]}"
        )
    return [by_id[i] for i in ids]


def load_audio(path: Path) -> np.ndarray:
    audio, _ = librosa.load(path, sr=AUDIO_SR, mono=True, dtype=np.float32)
    if audio.size == 0 or not np.isfinite(audio).all():
        raise ValueError(f"Invalid audio: {path}")
    return audio


# ---------------------------------------------------------------------------
# Spoof scoring
# ---------------------------------------------------------------------------

def fake_probability(detector, audio, device, window, batch_size):
    """Max P(fake) over fixed windows, batched onto the GPU.

    A file is fake if *any* part of it is, so windows are pooled with max --
    the same rule the baseline used for DF-Arena.
    """
    if audio.size == 0:
        return 0.0
    rms = float(np.sqrt(np.mean(np.square(audio, dtype=np.float64))))
    if rms < SILENCE_RMS:
        # Nothing to judge: an empty stem must not create fake evidence.
        return 0.0

    windows = np.stack([
        extract_segment(audio, start, window)
        for start in segment_starts(audio.size, window)
    ])

    best = 0.0
    for offset in range(0, len(windows), batch_size):
        chunk = torch.from_numpy(windows[offset:offset + batch_size]).to(device)
        best = max(best, float(detector.fake_probability(chunk).max()))
    return best


def combine(voice_fake, music_fake, voice_present, music_present):
    """File fake probability from component-conditional fake probabilities.

    Presence scores are ranking scores trained for CPS, not calibrated
    probabilities. Multiplying by them damaged file EER, especially when
    voice and music occur sequentially. Use them only to suppress clearly
    absent components, then apply the competition's logical OR as a max.
    """
    active = []
    if voice_present >= PRESENCE_GATE:
        active.append(voice_fake)
    if music_present >= PRESENCE_GATE:
        active.append(music_fake)
    if active:
        return max(active)
    return voice_fake if voice_present >= music_present else music_fake


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def run(args):
    device = torch.device(args.device)

    audio_files = find_audio_files(args.test_dir)
    columns, rows = read_sample_submission(args.sample_submission)
    audio_files = order_by_submission(audio_files, rows)

    if args.limit:
        audio_files, rows = audio_files[:args.limit], rows[:args.limit]

    if args.num_shards > 1:
        # Round-robin rather than contiguous blocks, so every shard gets a
        # similar mix of file durations and finishes at about the same time.
        selected = range(args.shard_index, len(audio_files), args.num_shards)
        audio_files = [audio_files[i] for i in selected]
        rows = [rows[i] for i in selected]
        print(f"shard {args.shard_index}/{args.num_shards}: {len(audio_files)} files",
              flush=True)
        if not audio_files:
            print("nothing to do for this shard")
            return

    print(f"[1/3] PANNs presence over {len(audio_files)} files", flush=True)
    presence_model = PannsPresence(args.panns_dir, device=args.device)
    presence = {}
    for path in tqdm(audio_files, desc="presence"):
        presence[path.stem] = presence_model.predict(load_audio(path))
    del presence_model
    torch.cuda.empty_cache()

    print(f"[2/3] Separation ({args.separator}) + XLS-R-2B spoof scoring", flush=True)
    separator_kwargs = {}
    if args.separator == "precomputed":
        separator_kwargs["stems_dir"] = args.stems_dir
    if args.separator == "htdemucs" and args.htdemucs_repo:
        # Offline submissions must read the bag from model/htdemucs instead of
        # reaching out to dl.fbaipublicfiles.com.
        separator_kwargs["repo"] = Path(args.htdemucs_repo)
    separator = build_separator(args.separator, device=args.device, **separator_kwargs)
    detector = XlsrAntiDeepfake.from_checkpoint(args.xlsr_dir, device=device)
    artifact_detector = ArtifactNetMusicDetector(args.artifactnet_dir)

    for row, path in zip(rows, tqdm(audio_files, desc="detect")):
        voice_audio, music_audio = separator.separate(path)
        voice_fake = fake_probability(
            detector, voice_audio, device, args.window, args.batch_size
        )
        music_fake_xlsr = fake_probability(
            detector, music_audio, device, args.window, args.batch_size
        )
        original_audio = load_audio(path)
        music_fake_artifact = artifact_detector.fake_probability(original_audio)
        music_fake = (
            XLSR_MUSIC_WEIGHT * music_fake_xlsr
            + ARTIFACTNET_WEIGHT * music_fake_artifact
        )
        voice_present, music_present = presence[path.stem]

        row["FILE_FAKE_PROB"] = round(
            combine(voice_fake, music_fake, voice_present, music_present), 10
        )
        row["VOICE_FAKE_PROB"] = round(voice_fake, 10)
        row["MUSIC_FAKE_PROB"] = round(music_fake, 10)
        row["VOICE_PRESENT_PROB"] = round(voice_present, 10)
        row["MUSIC_PRESENT_PROB"] = round(music_present, 10)

    print(f"[3/3] Writing {args.output}", flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved {len(rows)} predictions to {args.output}")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test-dir", type=Path, default=Path("data/test"))
    parser.add_argument("--sample-submission", type=Path,
                        default=Path("data/sample_submission.csv"))
    parser.add_argument("--output", type=Path, default=Path("output/submission.csv"))
    parser.add_argument("--panns-dir", type=Path, default=Path("models/panns"))
    parser.add_argument("--xlsr-dir", type=Path,
                        default=Path("models/xls-r-2b-anti-deepfake"))
    parser.add_argument("--artifactnet-dir", type=Path,
                        default=Path("models/artifactnet"))
    parser.add_argument("--separator",
                        choices=["htdemucs", "sam-audio", "precomputed"],
                        default="htdemucs")
    parser.add_argument("--stems-dir", type=Path, default=None,
                        help="Directory of <ID>_voice.wav / <ID>_music.wav "
                             "for --separator precomputed.")
    parser.add_argument("--htdemucs-repo", type=Path, default=None,
                        help="Local demucs bag directory (offline inference).")
    parser.add_argument("--device", default="cuda")
    # 4 s at 16 kHz. The detector was post-trained on 10-13 s crops, but the
    # baseline's 64600-sample window is kept configurable for comparison.
    parser.add_argument("--window", type=int, default=64_000)
    # XLS-R-2B needs about 8.2 GiB for one 4-second window but over 70 GiB at
    # batch 8. Keep the submission-safe default for the 22.4 GiB L4 grader.
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--limit", type=int, default=0,
                        help="Score only the first N files (smoke tests).")
    parser.add_argument("--num-shards", type=int, default=1,
                        help="Split the file list across this many processes.")
    parser.add_argument("--shard-index", type=int, default=0,
                        help="Which shard this process handles (0-based).")
    args = parser.parse_args(argv)
    if args.separator == "precomputed" and args.stems_dir is None:
        parser.error("--separator precomputed requires --stems-dir")
    if not 0 <= args.shard_index < args.num_shards:
        parser.error("--shard-index must satisfy 0 <= index < --num-shards")
    return args


if __name__ == "__main__":
    run(parse_args())
