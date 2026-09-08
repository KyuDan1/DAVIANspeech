#!/usr/bin/env python3
"""Build v83 plus frozen joint File/Voice and exact-codec Music soft-MoE."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import zipfile


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "best_eat_music_v83.zip"
OUTPUT = ROOT / "mixfake_joint_music_moe_v101.zip"
JOINT_CHECKPOINT = (
    ROOT / "reports/mixfake_multistream_v97/seed34_joint_robust/"
    "multistream_prompt.pt"
)
MUSIC_CHECKPOINT = (
    ROOT / "reports/mixfake_multistream_v96/"
    "seed35_music_exactcodec_paperwarm/multistream_prompt.pt"
)
NEW_ENTRIES = {
    "model/src/multistream_prompt_spectra.py": ROOT / "src/multistream_prompt_spectra.py",
    "model/src/mixfake_multistream_file_inference_v99.py": (
        ROOT / "src/mixfake_multistream_file_inference_v99.py"
    ),
    "model/mixfake-joint-v101/multistream_prompt.pt": JOINT_CHECKPOINT,
    "model/mixfake-music-v101/multistream_prompt.pt": MUSIC_CHECKPOINT,
}


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def patched_script(source, music_weight=.28, batch_size=12):
    import_marker = (
        "from three_stream_anchor_residual_inference import "
        "apply_three_stream_anchor_residual  # noqa: E402\n"
    )
    new_import = (
        "from mixfake_multistream_file_inference_v99 import "
        "apply_mixfake_joint_music_moe  # noqa: E402\n"
    )
    if source.count(import_marker) != 1 or new_import in source:
        raise ValueError("unexpected v83 import marker")
    source = source.replace(import_marker, import_marker + new_import)
    call_marker = (
        "    apply_music_only(args.test_dir, args.output, "
        "BASE_DIR / \"model\" / \"music82\", device=args.device)\n"
    )
    new_call = (
        "    # Frozen source-disjoint soft-MoE. Presence outputs are untouched.\n"
        "    apply_mixfake_joint_music_moe(\n"
        "        args.test_dir, args.output,\n"
        "        BASE_DIR / \"model\" / \"spectra-aasist\",\n"
        "        BASE_DIR / \"model\" / \"mixfake-joint-v101\" / \"multistream_prompt.pt\",\n"
        "        BASE_DIR / \"model\" / \"mixfake-music-v101\" / \"multistream_prompt.pt\",\n"
        "        device=args.device, file_weight=0.40, voice_weight=0.12,\n"
        f"        music_weight={music_weight:.12g}, batch_size={batch_size}, joint_views=3, music_views=1,\n"
        "        file_or_weight=0.15,\n"
        "    )\n"
    )
    if source.count(call_marker) != 1 or "mixfake-joint-v101" in source:
        raise ValueError("unexpected v83 call marker")
    return source.replace(call_marker, call_marker + new_call)


def inventory(handle):
    names = handle.namelist()
    return {
        "members": len(names),
        "expanded_bytes": sum(info.file_size for info in handle.infolist()),
        "compressed_bytes": sum(info.compress_size for info in handle.infolist()),
        "duplicates": sorted({name for name in names if names.count(name) > 1}),
        "unsafe_paths": [
            name for name in names
            if name.startswith("/") or ".." in Path(name).parts
        ],
        "top_level": sorted({Path(name).parts[0] for name in names if Path(name).parts}),
        "maximum_member_bytes": max(info.file_size for info in handle.infolist()),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--music-checkpoint", type=Path, default=MUSIC_CHECKPOINT)
    parser.add_argument("--music-weight", type=float, default=.28)
    parser.add_argument("--batch-size", type=int, default=12)
    args = parser.parse_args()
    if not 0 <= args.music_weight <= 1:
        parser.error("music-weight must lie in [0, 1]")
    if args.batch_size <= 0:
        parser.error("batch-size must be positive")
    output = args.output.resolve()
    music_checkpoint = args.music_checkpoint.resolve()
    new_entries = dict(NEW_ENTRIES)
    new_entries["model/mixfake-music-v101/multistream_prompt.pt"] = music_checkpoint
    if output.exists():
        raise FileExistsError(output)
    for path in (BASE, JOINT_CHECKPOINT, music_checkpoint, *new_entries.values()):
        if not Path(path).is_file():
            raise FileNotFoundError(path)
    base_report = json.loads(BASE.with_suffix(".report.json").read_text())
    if sha256_file(BASE) != base_report["sha256"]:
        raise ValueError("v83 archive hash mismatch")
    with zipfile.ZipFile(BASE) as source:
        source_script = source.read("script.py").decode("utf-8")
        source_names = set(source.namelist())
        source_meta = {
            info.filename: (info.CRC, info.file_size, info.compress_size)
            for info in source.infolist()
        }
    script = patched_script(
        source_script, music_weight=args.music_weight, batch_size=args.batch_size
    )
    with tempfile.TemporaryDirectory(prefix="v101-build-", dir=ROOT / "reports") as raw:
        directory = Path(raw)
        (directory / "script.py").write_text(script, encoding="utf-8")
        for name, source in new_entries.items():
            target = directory / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
        archive = directory / "candidate.zip"
        shutil.copyfile(BASE, archive)
        subprocess.run(
            ["zip", "-q", str(archive), "script.py", *new_entries],
            cwd=directory, check=True,
        )
        with zipfile.ZipFile(archive) as candidate:
            current = inventory(candidate)
            if current["duplicates"] or current["unsafe_paths"]:
                raise ValueError(current)
            if set(current["top_level"]) != {"model", "script.py", "requirements.txt"}:
                raise ValueError(current["top_level"])
            if set(candidate.namelist()) != source_names | set(new_entries):
                raise ValueError("unexpected archive member set")
            for name, metadata in source_meta.items():
                if name == "script.py":
                    continue
                info = candidate.getinfo(name)
                if (info.CRC, info.file_size, info.compress_size) != metadata:
                    raise ValueError(f"base member changed: {name}")
            if candidate.read("script.py") != script.encode():
                raise ValueError("script payload mismatch")
            for name, source in new_entries.items():
                if hashlib.sha256(candidate.read(name)).hexdigest() != sha256_file(source):
                    raise ValueError(f"new member payload mismatch: {name}")
            if candidate.testzip() is not None:
                raise ValueError("CRC verification failed")
        if archive.stat().st_size >= 10_000_000_000:
            raise ValueError("compressed archive exceeds 10 GB")
        if current["expanded_bytes"] >= 32_000_000_000:
            raise ValueError("expanded archive exceeds 32 GB")
        digest = sha256_file(archive)
        os.replace(archive, output)
    report = {
        "status": "complete_archive_validation",
        "archive": str(output), "sha256": digest,
        "bytes": output.stat().st_size, "inventory": current,
        "base_archive": str(BASE), "base_sha256": base_report["sha256"],
        "joint_checkpoint_sha256": sha256_file(JOINT_CHECKPOINT),
        "music_checkpoint_sha256": sha256_file(music_checkpoint),
        "changed_outputs": [
            "FILE_FAKE_PROB", "VOICE_FAKE_PROB", "MUSIC_FAKE_PROB"
        ],
        "weights": {"file": .40, "voice": .12, "music": args.music_weight},
        "file_or_weight": .15,
        "views": {"joint": 3, "music": 1},
        "batch_size": args.batch_size,
        "all_crc_verified": True, "official_submitted": False,
        "full_pipeline_smoke_verified": False,
    }
    output.with_suffix(".report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
