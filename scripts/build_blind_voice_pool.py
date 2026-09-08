#!/usr/bin/env python3
"""Replace a mixed-bank voice recipe with source/speaker-disjoint voices.

Real speech comes from the official LibriSpeech test-clean split. Fake speech
comes from uncompressed ASVspoof 2021 DF evaluation examples. This utility is
metadata/audio construction only and never imports an authenticity detector.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
from io import BytesIO
import json
from pathlib import Path
import shutil
import sys

import pandas as pd
import soundfile as sf
import yaml


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from data_guard import identity_tokens  # noqa: E402


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def md5_file(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_rank(seed: int, *parts: object) -> str:
    value = "|".join((str(seed), *(str(part) for part in parts)))
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def protected_tokens(config_path: Path) -> set[str]:
    config = yaml.safe_load(config_path.read_text("utf-8")) or {}
    root = config_path.resolve().parent.parent
    tokens: set[str] = set()
    for paths in config.values():
        if not isinstance(paths, list):
            continue
        for relative in paths:
            path = root / relative
            if not path.is_file():
                raise FileNotFoundError(path)
            tokens.update(identity_tokens(pd.read_csv(path, dtype=str)))
    return tokens


def audio_duration_from_bytes(payload: bytes) -> float:
    return float(sf.info(BytesIO(payload)).duration)


def select_real(
    root: Path, needed: int, seed: int, protected: set[str],
    max_per_speaker: int = 1,
) -> list[dict[str, object]]:
    if max_per_speaker < 1:
        raise ValueError("max_per_speaker must be positive")
    candidates = []
    for path in sorted(root.rglob("*.flac")):
        relative = path.relative_to(root)
        speaker_raw = relative.parts[0]
        source_id = f"librispeech-test-clean:{path.stem}"
        speaker = f"librispeech-test-clean:{speaker_raw}"
        duration = float(sf.info(path).duration)
        if not 4.0 <= duration <= 60.0:
            continue
        if source_id in protected or speaker in protected:
            continue
        candidates.append({
            "source_id": source_id,
            "speaker": speaker,
            "generator": "bonafide",
            "source_dataset": "LibriSpeech_test-clean",
            "source_path": path,
            "duration": duration,
        })
    # One recording per speaker prevents a small number of readers from
    # dominating the real half of the blind bank.
    candidates.sort(
        key=lambda row: stable_rank(seed, row["speaker"], row["source_id"])
    )
    selected: list[dict[str, object]] = []
    selected_ids: set[str] = set()
    speaker_counts: Counter[str] = Counter()
    # Fill one utterance per speaker before permitting a second.  This retains
    # the original maximum-speaker-diversity policy when the protected pool no
    # longer contains ``needed`` distinct speakers.
    for quota in range(1, max_per_speaker + 1):
        for row in candidates:
            if row["source_id"] in selected_ids:
                continue
            if speaker_counts[row["speaker"]] >= quota:
                continue
            selected.append(row)
            selected_ids.add(row["source_id"])
            speaker_counts[row["speaker"]] += 1
            if len(selected) == needed:
                break
        if len(selected) == needed:
            break
    if len(selected) != needed:
        raise ValueError(
            f"need {needed} real sources with <= {max_per_speaker}/speaker, "
            f"found {len(selected)}"
        )
    return selected


def select_fake(
    parquet_paths: list[Path], needed: int, seed: int, protected: set[str],
    max_per_speaker: int = 2,
) -> list[dict[str, object]]:
    if max_per_speaker < 1:
        raise ValueError("max_per_speaker must be positive")
    candidates = []
    for parquet_path in parquet_paths:
        frame = pd.read_parquet(
            parquet_path, columns=["path", "audio", "label", "notes"]
        )
        for row in frame.itertuples(index=False):
            if int(row.label) != 1:
                continue
            notes = json.loads(row.notes)
            speaker_raw = str(notes.get("speaker_id", ""))
            # LA speakers overlap the 2019 evaluation voices used throughout
            # the repository. VCC/DF speakers are retained with an explicit
            # corpus namespace.
            if not speaker_raw or speaker_raw.startswith("LA_"):
                continue
            if str(notes.get("codec")) != "nocodec":
                continue
            payload = bytes(row.audio["bytes"])
            source_id = f"asvspoof2021-df:{Path(row.path).stem}"
            speaker = f"asvspoof2021-df:{speaker_raw}"
            generator = str(notes.get("attack_id", "unknown"))
            if source_id in protected or speaker in protected:
                continue
            duration = audio_duration_from_bytes(payload)
            if not 4.0 <= duration <= 60.0:
                continue
            candidates.append({
                "source_id": source_id,
                "speaker": speaker,
                "generator": generator,
                "source_dataset": "ASVspoof2021_DF_eval_nocodec",
                "payload": payload,
                "duration": duration,
                "source": str(notes.get("source", "unknown")),
                "vocoder": str(notes.get("vocoder", "unknown")),
                "parquet": parquet_path,
            })
    candidates.sort(
        key=lambda row: stable_rank(
            seed, row["speaker"], row["generator"], row["source_id"]
        )
    )
    selected = []
    speaker_counts: Counter[str] = Counter()
    generators: set[str] = set()
    # First pass maximizes attack diversity and caps each speaker at two.
    for row in candidates:
        if (
            speaker_counts[row["speaker"]] >= max_per_speaker
            or row["generator"] in generators
        ):
            continue
        selected.append(row)
        speaker_counts[row["speaker"]] += 1
        generators.add(row["generator"])
        if len(selected) == needed:
            break
    # A deterministic fallback preserves the speaker cap if fewer unique
    # attack IDs are present in the locally materialized parquet shards.
    selected_ids = {row["source_id"] for row in selected}
    for row in candidates:
        if len(selected) == needed:
            break
        if (
            row["source_id"] in selected_ids
            or speaker_counts[row["speaker"]] >= max_per_speaker
        ):
            continue
        selected.append(row)
        selected_ids.add(row["source_id"])
        speaker_counts[row["speaker"]] += 1
    if len(selected) != needed:
        raise ValueError(f"need {needed} fake voices, found {len(selected)}")
    return selected


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-reservation", type=Path, required=True)
    parser.add_argument("--base-provenance", type=Path, required=True)
    parser.add_argument("--partition-config", type=Path, required=True)
    parser.add_argument("--librispeech-root", type=Path, required=True)
    parser.add_argument("--librispeech-archive", type=Path, required=True)
    parser.add_argument("--librispeech-md5", required=True)
    parser.add_argument("--fake-parquet", type=Path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260907)
    parser.add_argument("--max-real-per-speaker", type=int, default=1)
    parser.add_argument("--max-fake-per-speaker", type=int, default=2)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite {args.output_dir}")
    for path in (
        args.base_reservation, args.base_provenance, args.partition_config,
        args.librispeech_root, args.librispeech_archive, *args.fake_parquet,
    ):
        if not path.exists():
            raise FileNotFoundError(path)
    if md5_file(args.librispeech_archive) != args.librispeech_md5:
        raise ValueError("LibriSpeech archive MD5 mismatch")
    base = pd.read_csv(args.base_reservation, dtype=str).fillna("")
    needed = int(base.VOICE_FAKE.astype(int).value_counts().min())
    if set(base.VOICE_FAKE.astype(int)) != {0, 1} or len(base) != 2 * needed:
        raise ValueError("base reservation must have balanced real/fake voices")
    protected = protected_tokens(args.partition_config)
    real = select_real(
        args.librispeech_root, needed, args.seed + 1, protected,
        max_per_speaker=args.max_real_per_speaker,
    )
    fake = select_fake(
        args.fake_parquet, needed, args.seed + 2, protected,
        max_per_speaker=args.max_fake_per_speaker,
    )
    pools = {0: iter(real), 1: iter(fake)}

    args.output_dir.mkdir(parents=True)
    audio_dir = args.output_dir / "voice_pool"
    audio_dir.mkdir()
    records = []
    result = base.copy()
    for index, row in result.iterrows():
        label = int(row.VOICE_FAKE)
        item = next(pools[label])
        filename = hashlib.sha256(item["source_id"].encode()).hexdigest()[:20] + ".flac"
        destination = audio_dir / filename
        if label:
            destination.write_bytes(item["payload"])
        else:
            shutil.copy2(item["source_path"], destination)
        info = sf.info(destination)
        if info.samplerate != 16_000 or not 4.0 <= info.duration <= 60.0:
            raise ValueError(f"invalid selected voice audio: {destination}")
        result.at[index, "VOICE_SOURCE_ID"] = item["source_id"]
        result.at[index, "VOICE_SPEAKER"] = item["speaker"]
        result.at[index, "VOICE_GENERATOR"] = item["generator"]
        result.at[index, "VOICE_ARCHIVE_ID"] = item["source_id"]
        result.at[index, "VOICE_ARCHIVE_MEMBER"] = f"voice_pool/{filename}"
        records.append({
            "VOICE_FAKE": label,
            "VOICE_SOURCE_ID": item["source_id"],
            "VOICE_SPEAKER": item["speaker"],
            "VOICE_GENERATOR": item["generator"],
            "VOICE_SOURCE_DATASET": item["source_dataset"],
            "VOICE_ARCHIVE_MEMBER": f"voice_pool/{filename}",
            "DURATION": info.duration,
            "AUDIO_SHA256": sha256_file(destination),
        })
    result["VOICE_SOURCE_DATASET"] = [
        record["VOICE_SOURCE_DATASET"] for record in records
    ]
    result["SOURCE_DATASET"] = "external_voice_disjoint+MixFake_music"
    if result.VOICE_SOURCE_ID.nunique() != len(result):
        raise ValueError("voice source reuse in final reservation")
    if set(result.VOICE_SPEAKER) & protected:
        raise ValueError("protected speaker survived final reservation")
    if identity_tokens(result) & protected:
        overlap = sorted(identity_tokens(result) & protected)
        raise ValueError(f"protected identity survived: {overlap[:5]}")

    reservation_path = args.output_dir / "reservation.csv"
    voice_manifest_path = args.output_dir / "voice_manifest.csv"
    result.to_csv(reservation_path, index=False)
    pd.DataFrame(records).to_csv(voice_manifest_path, index=False)
    base_provenance = json.loads(args.base_provenance.read_text("utf-8"))
    provenance = {
        **base_provenance,
        "stage": "reserved_external_voice_source_and_speaker_disjoint",
        "seed_voice_pool": args.seed,
        "base_reservation": {
            "path": str(args.base_reservation.resolve()),
            "sha256": sha256_file(args.base_reservation),
        },
        "reservation_sha256": sha256_file(reservation_path),
        "voice_manifest_sha256": sha256_file(voice_manifest_path),
        "voice_pool": {
            "real": "LibriSpeech test-clean",
            "real_license": "CC BY 4.0",
            "real_url": "https://www.openslr.org/12",
            "real_archive_path": str(args.librispeech_archive.resolve()),
            "real_archive_md5": args.librispeech_md5,
            "fake": "ASVspoof 2021 DF evaluation nocodec",
            "fake_parquets": [
                {"path": str(path.resolve()), "sha256": sha256_file(path)}
                for path in args.fake_parquet
            ],
            "selected_real_sources": len(real),
            "selected_real_speakers": len({row["speaker"] for row in real}),
            "max_selected_real_sources_per_speaker": max(
                Counter(row["speaker"] for row in real).values()
            ),
            "selected_fake_sources": len(fake),
            "selected_fake_speakers": len({row["speaker"] for row in fake}),
            "max_selected_fake_sources_per_speaker": max(
                Counter(row["speaker"] for row in fake).values()
            ),
            "selected_fake_generators": len({row["generator"] for row in fake}),
            "codec_filter": "nocodec",
        },
        "authenticity_detector_inference": False,
        "authenticity_score_computed": False,
    }
    (args.output_dir / "provenance.json").write_text(
        json.dumps(provenance, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(provenance["voice_pool"], indent=2))


if __name__ == "__main__":
    main()
