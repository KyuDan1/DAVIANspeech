#!/usr/bin/env python3
"""Materialize exact source and archive requirements for a MixFake reservation."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from urllib.parse import quote

import pandas as pd


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reservation", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--archive-index-json", type=Path)
    parser.add_argument("--repository-json", type=Path)
    parser.add_argument("--repository", default="Tnxts/MixFake")
    parser.add_argument("--archive-prefix", default="MixFake")
    args = parser.parse_args()

    required_outputs = [
        args.output_dir / "source_requirements.csv",
        args.output_dir / "archive_member_targets.txt",
    ]
    if args.archive_index_json is not None:
        required_outputs.extend([
            args.output_dir / "archive_volumes.csv",
            args.output_dir / "archive_requirements_provenance.json",
        ])
    existing = [path for path in required_outputs if path.exists()]
    if existing:
        raise FileExistsError(f"refusing to overwrite: {existing}")

    frame = pd.read_csv(args.reservation, dtype=str).fillna("")
    required = {
        "BASE_ID", "VOICE_SOURCE_ID", "VOICE_ARCHIVE_ID",
        "VOICE_ARCHIVE_MEMBER", "MUSIC_SOURCE_ID", "MUSIC_GROUP_ID",
        "MUSIC_ARCHIVE_ID", "MUSIC_ARCHIVE_MEMBER",
    }
    if missing := required.difference(frame):
        raise ValueError(f"reservation lacks columns: {sorted(missing)}")

    voice = frame[[
        "VOICE_SOURCE_ID", "VOICE_ARCHIVE_ID", "VOICE_ARCHIVE_MEMBER",
    ]].drop_duplicates()
    voice.columns = ["SOURCE_ID", "ARCHIVE_ID", "ARCHIVE_MEMBER"]
    voice.insert(0, "CANONICAL_GROUP", "")
    voice.insert(0, "KIND", "voice")
    music = frame[[
        "MUSIC_SOURCE_ID", "MUSIC_GROUP_ID", "MUSIC_ARCHIVE_ID",
        "MUSIC_ARCHIVE_MEMBER",
    ]].drop_duplicates()
    music.columns = [
        "SOURCE_ID", "CANONICAL_GROUP", "ARCHIVE_ID", "ARCHIVE_MEMBER",
    ]
    music.insert(0, "KIND", "music")
    sources = pd.concat([voice, music], ignore_index=True)
    sources["ARCHIVE_TARGET"] = [
        f"{args.archive_prefix}/{member.lstrip('/')}"
        for member in sources.ARCHIVE_MEMBER
    ]
    if len(sources) != 2 * len(frame):
        raise ValueError(
            "reservation source reuse detected: expected two unique sources "
            f"per base, got {len(sources)} sources for {len(frame)} bases"
        )
    if sources.ARCHIVE_TARGET.duplicated().any():
        raise ValueError("duplicate archive target")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    sources.sort_values(["KIND", "SOURCE_ID"]).to_csv(
        args.output_dir / "source_requirements.csv", index=False
    )
    targets = sorted(sources.ARCHIVE_TARGET)
    (args.output_dir / "archive_member_targets.txt").write_text(
        "\n".join(targets) + "\n", encoding="utf-8"
    )

    if args.archive_index_json is None:
        return
    index = json.loads(args.archive_index_json.read_text("utf-8"))
    revision = "main"
    repository_metadata: dict[str, object] = {}
    if args.repository_json is not None:
        repository_metadata = json.loads(args.repository_json.read_text("utf-8"))
        revision = str(repository_metadata.get("sha") or revision)
    volume_rows = []
    for row in index:
        path = str(row.get("path", ""))
        if not re.fullmatch(r"MixFake\.7z\.\d{3}", path):
            continue
        lfs = row.get("lfs") or {}
        volume_rows.append({
            "VOLUME": path,
            "BYTES": int(row["size"]),
            "EXPECTED_SHA256": str(lfs.get("oid", "")),
            "URL": (
                "https://huggingface.co/datasets/"
                f"{args.repository}/resolve/{revision}/{quote(path)}"
            ),
        })
    volumes = pd.DataFrame(volume_rows).sort_values("VOLUME")
    if list(volumes.VOLUME) != [f"MixFake.7z.{i:03d}" for i in range(1, 68)]:
        raise ValueError("expected exactly the 67 contiguous MixFake volumes")
    if (volumes.EXPECTED_SHA256 == "").any():
        raise ValueError("archive index lacks an LFS SHA-256")
    volumes.to_csv(args.output_dir / "archive_volumes.csv", index=False)
    provenance = {
        "repository": args.repository,
        "revision": revision,
        "repository_last_modified": repository_metadata.get("lastModified"),
        "archive_index_path": str(args.archive_index_json.resolve()),
        "archive_index_sha256": sha256_file(args.archive_index_json),
        "repository_json_path": (
            None if args.repository_json is None
            else str(args.repository_json.resolve())
        ),
        "repository_json_sha256": (
            None if args.repository_json is None
            else sha256_file(args.repository_json)
        ),
        "reservation_path": str(args.reservation.resolve()),
        "reservation_sha256": sha256_file(args.reservation),
        "source_requirement_rows": int(len(sources)),
        "archive_member_target_rows": int(len(targets)),
        "archive_volume_rows": int(len(volumes)),
        "archive_total_bytes": int(volumes.BYTES.sum()),
    }
    (args.output_dir / "archive_requirements_provenance.json").write_text(
        json.dumps(provenance, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
