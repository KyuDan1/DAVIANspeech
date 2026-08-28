"""DAVIANspeech AI-generated audio detection pipeline.

    INPUT AUDIO
    |
    +-- PANNs Cnn14 ------> VOICE_PRESENT_PROB (VP), MUSIC_PRESENT_PROB (MP)
    |
    +-- original audio -----------------> XLS-R-2B --------+--> voice/music votes
    +-- original audio -----------------> EAT -------------+--> music votes
    +-- Separator --------> voice stem  --> XLS-R-2B ------+

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
from eat_detector import EatMusicDetector  # noqa: E402
from spear_detector import SpearMusicDetector  # noqa: E402
from xlsr_antideepfake import XlsrAntiDeepfake  # noqa: E402

AUDIO_SR = 16_000
SILENCE_RMS = 1e-5
PRESENCE_GATE = 0.7
# Source separation can erase generator traces and can also introduce traces
# of its own.  Cross-suite maximin validation therefore gives the original
# mixture most of the XLS-R vote and retains stems as complementary evidence.
RAW_VOICE_WEIGHT = 0.6
STEM_VOICE_WEIGHT = 0.4
LEGACY_VOICE_ENSEMBLE_WEIGHT = 0.7
ECHOFAKE_VOICE_WEIGHT = 0.3
# Cross-dataset maximin weights. The original heads preserve performance on
# FakeMusicCaps, while Echoes heads cover twelve newer commercial/open models.
EAT_MUSIC_WEIGHT = 0.225
XLSR_ADAPTED_MUSIC_WEIGHT = 0.09
EAT_ECHOES_MUSIC_WEIGHT = 0.225
XLSR_ECHOES_MUSIC_WEIGHT = 0.36
SPEAR_MUSIC_WEIGHT = 0.10
MIXTURE_GATE = 0.80
MIXTURE_VOICE_WEIGHT = 0.20
MIXTURE_MUSIC_WEIGHT = 0.20

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

def pool_window_scores(scores: np.ndarray, method: str, temperature: float) -> float:
    """Reduce per-window P(fake) to one file-level score.

    ``max`` is the obvious operator -- a file is fake if any window is -- but it
    is badly length-biased. Measured on genuine Korean speech, mean max rises
    0.033 -> 0.316 going from a 4 s file to a 60 s one, purely because 15 draws
    give the tail 15 chances instead of one. Entries here span 4-60 s, so under
    max the score partly encodes duration rather than authenticity, and that
    costs real EER wherever the classes overlap.

    ``logmeanexp`` is a soft max: it still reacts to a single confident window,
    but averages over the rest, which cut the same length drift to 0.033 -> 0.087
    with no loss of separation on either whole-file or 4 s spliced-in fakes.
    """
    if scores.size == 0:
        return 0.0
    if method == "max":
        return float(scores.max())
    if method == "mean":
        return float(scores.mean())
    if method == "topk":
        k = min(3, scores.size)
        return float(np.sort(scores)[-k:].mean())
    if method == "logmeanexp":
        scaled = temperature * np.clip(scores, 0.0, 1.0)
        peak = scaled.max()
        return float((peak + np.log(np.mean(np.exp(scaled - peak)))) / temperature)
    raise ValueError(f"Unknown pooling method: {method}")


def fake_probability(detector, audio, device, window, batch_size,
                     pooling="max", temperature=5.0):
    """P(fake) for one stem, pooled over fixed windows."""
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

    scores = []
    for offset in range(0, len(windows), batch_size):
        chunk = torch.from_numpy(windows[offset:offset + batch_size]).to(device)
        scores.extend(detector.fake_probability(chunk).tolist())
    return pool_window_scores(np.asarray(scores), pooling, temperature)


def fake_probability_and_embedding(detector, audio, device, window, batch_size):
    """Return the released head score and mean pooled file representation.

    The raw-audio XLS-R pass is shared by the speech head and the adapted
    music head, so adding the latter has no extra 2B-encoder inference cost.
    """
    windows = np.stack([
        extract_segment(audio, start, window)
        for start in segment_starts(audio.size, window)
    ])
    best, embeddings = 0.0, []
    for offset in range(0, len(windows), batch_size):
        chunk = torch.from_numpy(windows[offset:offset + batch_size]).to(device)
        with torch.inference_mode():
            pooled = detector.embedding(detector.normalize(chunk))
            probabilities = torch.softmax(
                detector.proj_fc(pooled).float(), dim=-1
            )[:, 0]
        best = max(best, float(probabilities.max()))
        embeddings.append(pooled.float().cpu())
    return best, torch.cat(embeddings).mean(dim=0)


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
    xlsr_head = np.load(args.xlsr_music_head)
    xlsr_music_weight = torch.from_numpy(xlsr_head["weight"]).to(device)
    xlsr_music_bias = torch.as_tensor(xlsr_head["bias"], device=device)
    xlsr_echoes_head = np.load(args.xlsr_echoes_music_head)
    xlsr_echoes_weight = torch.from_numpy(xlsr_echoes_head["weight"]).to(device)
    xlsr_echoes_bias = torch.as_tensor(xlsr_echoes_head["bias"], device=device)
    xlsr_echofake_head = np.load(args.xlsr_echofake_voice_head)
    xlsr_echofake_weight = torch.from_numpy(
        xlsr_echofake_head["weight"]
    ).to(device)
    xlsr_echofake_bias = torch.as_tensor(
        xlsr_echofake_head["bias"], device=device
    )
    eat_detector = EatMusicDetector(
        args.eat_dir, args.eat_head, device=args.device,
        extra_head_path=args.eat_echoes_head,
    )
    spear_detector = SpearMusicDetector(
        args.spear_dir, args.spear_music_head, device=args.device,
        extra_head_paths=(
            args.spear_mixed_voice_head, args.spear_mixed_music_head,
            args.spear_mixture_present_head,
        ),
    )

    for row, path in zip(rows, tqdm(audio_files, desc="detect")):
        original_audio = load_audio(path)
        voice_audio, _ = separator.separate(path)
        voice_fake_stem = fake_probability(
            detector, voice_audio, device, args.window, args.batch_size
        )
        raw_fake_xlsr, raw_xlsr_embedding = fake_probability_and_embedding(
            detector, original_audio, device, args.window, args.batch_size
        )
        music_fake_xlsr_adapted = float(torch.sigmoid(
            raw_xlsr_embedding.to(device) @ xlsr_music_weight + xlsr_music_bias
        ))
        music_fake_xlsr_echoes = float(torch.sigmoid(
            raw_xlsr_embedding.to(device) @ xlsr_echoes_weight + xlsr_echoes_bias
        ))
        voice_fake_xlsr_echofake = float(torch.sigmoid(
            raw_xlsr_embedding.to(device) @ xlsr_echofake_weight
            + xlsr_echofake_bias
        ))
        music_fake_eat, music_fake_eat_echoes = (
            eat_detector.fake_probabilities(original_audio)
        )
        (music_fake_spear, mixed_voice_fake, mixed_music_fake,
         mixture_present) = (
            spear_detector.fake_probabilities(original_audio)
        )
        legacy_voice_fake = (
            STEM_VOICE_WEIGHT * voice_fake_stem
            + RAW_VOICE_WEIGHT * raw_fake_xlsr
        )
        voice_fake = (
            LEGACY_VOICE_ENSEMBLE_WEIGHT * legacy_voice_fake
            + ECHOFAKE_VOICE_WEIGHT * voice_fake_xlsr_echofake
        )
        music_fake = (
            EAT_MUSIC_WEIGHT * music_fake_eat
            + XLSR_ADAPTED_MUSIC_WEIGHT * music_fake_xlsr_adapted
            + EAT_ECHOES_MUSIC_WEIGHT * music_fake_eat_echoes
            + XLSR_ECHOES_MUSIC_WEIGHT * music_fake_xlsr_echoes
            + SPEAR_MUSIC_WEIGHT * music_fake_spear
        )
        if args.artifactnet_weight > 0:
            music_fake_artifact = artifact_detector.fake_probability(original_audio)
            music_fake = ((1 - args.artifactnet_weight) * music_fake
                          + args.artifactnet_weight * music_fake_artifact)
        voice_present, music_present = presence[path.stem]
        # A dedicated raw-mixture expert recovers artifacts masked by the
        # other component. It is never applied to obvious single-component
        # audio, where its training distribution would be inappropriate.
        if mixture_present >= MIXTURE_GATE:
            voice_fake = (
                (1 - MIXTURE_VOICE_WEIGHT) * voice_fake
                + MIXTURE_VOICE_WEIGHT * mixed_voice_fake
            )
            music_fake = (
                (1 - MIXTURE_MUSIC_WEIGHT) * music_fake
                + MIXTURE_MUSIC_WEIGHT * mixed_music_fake
            )

        # The high-specificity SPEAR router is substantially more reliable at
        # identifying mixtures than thresholding two independently calibrated
        # PANNs ranking scores. For a routed mixture both components are known
        # to exist, so implement the competition's logical OR directly.
        file_fake = (
            max(voice_fake, music_fake)
            if mixture_present >= MIXTURE_GATE
            else combine(voice_fake, music_fake, voice_present, music_present)
        )
        row["FILE_FAKE_PROB"] = round(file_fake, 10)
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
    parser.add_argument("--xlsr-music-head", type=Path,
                        default=Path("model_heads/xlsr-music-head.npz"))
    parser.add_argument("--xlsr-echoes-music-head", type=Path,
                        default=Path("model_heads/xlsr-echoes-music-head.npz"))
    parser.add_argument("--xlsr-echofake-voice-head", type=Path,
                        default=Path("model_heads/xlsr-echofake-voice-head.npz"))
    parser.add_argument("--eat-dir", type=Path, default=Path("models/eat-base-as2m"))
    parser.add_argument("--eat-head", type=Path,
                        default=Path("model_heads/eat-music-head.npz"))
    parser.add_argument("--eat-echoes-head", type=Path,
                        default=Path("model_heads/eat-echoes-music-head.npz"))
    parser.add_argument("--spear-dir", type=Path,
                        default=Path("models/spear-xlarge-speech-audio-v2"))
    parser.add_argument("--spear-music-head", type=Path,
                        default=Path("model_heads/spear-v3-music-head.npz"))
    parser.add_argument("--spear-mixed-voice-head", type=Path,
                        default=Path("model_heads/spear-mixed-voice_fake-head.npz"))
    parser.add_argument("--spear-mixed-music-head", type=Path,
                        default=Path("model_heads/spear-mixed-music_fake-head.npz"))
    parser.add_argument("--spear-mixture-present-head", type=Path,
                        default=Path("model_heads/spear-mixture-present-head.npz"))
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
    parser.add_argument("--pooling", choices=["max", "mean", "topk", "logmeanexp"],
                        default="max",
                        help="How per-window scores become a file score.")
    parser.add_argument("--temperature", type=float, default=5.0,
                        help="logmeanexp sharpness; higher approaches max.")
    parser.add_argument("--music-source", choices=["stem", "original"],
                        default="stem",
                        help="What the XLS-R music score is computed on.")
    parser.add_argument("--artifactnet-weight", type=float,
                        default=ARTIFACTNET_WEIGHT,
                        help="Blend weight for ArtifactNet; 0 disables it.")
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
