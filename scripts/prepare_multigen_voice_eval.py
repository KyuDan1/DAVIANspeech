#!/usr/bin/env python3
"""Prepare speaker-disjoint real speech and generation jobs for a TTS eval.

The source tree contains 16 Korean speakers spanning news, lectures, elderly
dialects, and children.  Each speaker belongs to exactly one of dev, holdout,
or locked.  A target utterance is kept as bona-fide audio and its text is also
sent to every TTS family, so real/fake content cannot become a shortcut.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
import soundfile as sf


SR = 16_000
SPLIT_BY_SPEAKER = {
    "dohyun": "dev", "eunju": "dev", "mansu": "dev",
    "seoa": "dev", "siwoo": "dev",
    "minji": "holdout", "jihye": "holdout", "byungchul": "holdout",
    "jian": "holdout", "hayoon": "holdout",
    "seoyeon": "locked", "jungwoo": "locked", "sunja": "locked",
    "youngsoon": "locked", "junseo": "locked", "hajun": "locked",
}
SOURCE_ROOTS = {
    "아나운서": "aihub_news/work",
    "강의": "aihub_lecture/work",
    "노년방언": "aihub_dialect/work",
    "아동": "aihub_child/work",
}


def stable_rank(seed: int, *parts: str) -> str:
    return hashlib.sha256((str(seed) + "|" + "|".join(parts)).encode()).hexdigest()


def read_candidates(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            audio = Path(row["audio"])
            text = " ".join(str(row["text"]).split())
            duration = float(row.get("duration", 0.0))
            if audio.is_file() and 4.0 <= duration <= 12.0 and 12 <= len(text) <= 120:
                row["audio"] = str(audio.resolve())
                row["text"] = text
                rows.append(row)
    return rows


def normalize_audio(source: Path, destination: Path) -> float:
    audio, _ = librosa.load(source, sr=SR, mono=True, dtype=np.float32)
    peak = float(np.max(np.abs(audio))) if len(audio) else 0.0
    if peak > 0.99:
        audio = audio * (0.99 / peak)
    sf.write(destination, audio, SR, format="FLAC", subtype="PCM_16")
    return len(audio) / SR


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--voices-json", type=Path, required=True)
    parser.add_argument("--speech-root", type=Path, required=True)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--targets-per-speaker", type=int, default=3)
    parser.add_argument("--extra-real-per-speaker", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260831)
    args = parser.parse_args()

    voices = json.loads(args.voices_json.read_text("utf-8"))
    declared = {voice["speaker"] for voice in voices}
    if declared != set(SPLIT_BY_SPEAKER):
        raise ValueError("Voice inventory changed; update the explicit split assignment")

    audio_dir = args.output_dir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    jobs, real_rows = [], []
    for voice in voices:
        speaker = voice["speaker"]
        source_dir = (
            args.speech_root / SOURCE_ROOTS[voice["category"]]
            / voice["source_speaker_id"]
        )
        candidates = read_candidates(source_dir / "train.jsonl")
        candidates.sort(key=lambda row: stable_rank(args.seed, speaker, row["audio"]))
        needed = 1 + args.targets_per_speaker + args.extra_real_per_speaker
        if len(candidates) < needed:
            raise RuntimeError(f"Only {len(candidates)} usable clips for {speaker}")
        reference, targets = candidates[0], candidates[1:1 + args.targets_per_speaker]
        extras = candidates[1 + args.targets_per_speaker:needed]
        split = SPLIT_BY_SPEAKER[speaker]

        for index, row in enumerate(targets):
            source_id = f"{speaker}_target_{index:02d}"
            real_id = f"voice_real_{source_id}"
            destination = audio_dir / f"{real_id}.flac"
            duration = normalize_audio(Path(row["audio"]), destination)
            real_rows.append({
                "ID": real_id, "FILE_FAKE": 0, "VOICE_FAKE": 0,
                "MUSIC_FAKE": "", "VOICE_PRESENT": 1, "MUSIC_PRESENT": 0,
                "AUDIO_TYPE": "voice", "SOURCE": "AIHub",
                "GENERATOR": "bonafide", "CONDITION": "speech_only",
                "SPLIT": split, "GROUP_ID": source_id, "SPEAKER": speaker,
                "CATEGORY": voice["category"], "GENDER": voice["gender"],
                "AGE": voice["age"], "SOURCE_FILE": str(Path(row["audio"]).resolve()),
                "DURATION": round(duration, 3),
            })
            jobs.append({
                "JOB_ID": source_id, "SPLIT": split, "TEXT": row["text"],
                "SOURCE_REAL_ID": real_id, "SOURCE_FILE": str(Path(row["audio"]).resolve()),
                "REFERENCE_AUDIO": reference["audio"],
                "REFERENCE_TEXT": reference["text"], "SPEAKER": speaker,
                "CATEGORY": voice["category"], "GENDER": voice["gender"],
                "AGE": voice["age"],
                "QWEN_MODEL": str((args.model_root / voice["dir"]).resolve()),
            })

        for index, row in enumerate(extras):
            source_id = f"{speaker}_extra_{index:02d}"
            real_id = f"voice_real_{source_id}"
            destination = audio_dir / f"{real_id}.flac"
            duration = normalize_audio(Path(row["audio"]), destination)
            real_rows.append({
                "ID": real_id, "FILE_FAKE": 0, "VOICE_FAKE": 0,
                "MUSIC_FAKE": "", "VOICE_PRESENT": 1, "MUSIC_PRESENT": 0,
                "AUDIO_TYPE": "voice", "SOURCE": "AIHub",
                "GENERATOR": "bonafide", "CONDITION": "speech_only",
                "SPLIT": split, "GROUP_ID": source_id, "SPEAKER": speaker,
                "CATEGORY": voice["category"], "GENDER": voice["gender"],
                "AGE": voice["age"], "SOURCE_FILE": str(Path(row["audio"]).resolve()),
                "DURATION": round(duration, 3),
            })

    jobs_frame = pd.DataFrame(jobs).sort_values(["SPLIT", "SPEAKER", "JOB_ID"])
    truth_frame = pd.DataFrame(real_rows).sort_values("ID")
    jobs_frame.to_csv(args.output_dir / "generation_manifest.csv", index=False)
    truth_frame.to_csv(args.output_dir / "real_truth.csv", index=False)
    print(f"Prepared {len(jobs_frame)} generation jobs and {len(truth_frame)} real clips")
    print(jobs_frame.groupby(["SPLIT", "CATEGORY"]).size().unstack(fill_value=0))


if __name__ == "__main__":
    main()
