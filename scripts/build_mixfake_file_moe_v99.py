#!/usr/bin/env python3
"""Build the v83 anchor plus the frozen three-view File soft-MoE."""
from __future__ import annotations

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
OUTPUT = ROOT / "mixfake_file_moe_v99.zip"
CHECKPOINT = (
    ROOT / "reports/mixfake_multistream_v94/seed31_file_robust/"
    "multistream_prompt.pt"
)
NEW_ENTRIES = {
    "model/src/multistream_prompt_spectra.py": ROOT / "src/multistream_prompt_spectra.py",
    "model/src/mixfake_multistream_file_inference_v99.py": (
        ROOT / "src/mixfake_multistream_file_inference_v99.py"
    ),
    "model/mixfake-file-v99/multistream_prompt.pt": CHECKPOINT,
}


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def patched_script(text):
    import_marker = (
        "from three_stream_anchor_residual_inference import "
        "apply_three_stream_anchor_residual  # noqa: E402\n"
    )
    new_import = (
        "from mixfake_multistream_file_inference_v99 import "
        "apply_mixfake_file_moe  # noqa: E402\n"
    )
    if text.count(import_marker) != 1 or new_import in text:
        raise ValueError("unexpected v83 import marker")
    text = text.replace(import_marker, import_marker + new_import)
    call_marker = (
        "    apply_music_only(args.test_dir, args.output, "
        "BASE_DIR / \"model\" / \"music82\", device=args.device)\n"
    )
    call = (
        "    # Source/layout-robust fixed soft-MoE selected on development only.\n"
        "    apply_mixfake_file_moe(\n"
        "        args.test_dir, args.output,\n"
        "        BASE_DIR / \"model\" / \"spectra-aasist\",\n"
        "        BASE_DIR / \"model\" / \"mixfake-file-v99\" / \"multistream_prompt.pt\",\n"
        "        device=args.device, weight=0.40, batch_size=12, views=3,\n"
        "    )\n"
    )
    if text.count(call_marker) != 1 or "mixfake-file-v99" in text:
        raise ValueError("unexpected v83 call marker")
    return text.replace(call_marker, call_marker + call)


def inventory(handle):
    names = handle.namelist()
    unsafe = [
        name for name in names
        if name.startswith("/") or ".." in Path(name).parts
    ]
    return {
        "members": len(names),
        "expanded_bytes": sum(info.file_size for info in handle.infolist()),
        "compressed_bytes": sum(info.compress_size for info in handle.infolist()),
        "duplicates": sorted({name for name in names if names.count(name) > 1}),
        "unsafe_paths": unsafe,
        "top_level": sorted({Path(name).parts[0] for name in names if Path(name).parts}),
        "maximum_member_bytes": max(info.file_size for info in handle.infolist()),
    }


def main():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    if not CHECKPOINT.is_file():
        raise FileNotFoundError(CHECKPOINT)
    base_report = json.loads(BASE.with_suffix(".report.json").read_text())
    if sha256_file(BASE) != base_report["sha256"]:
        raise ValueError("v83 archive hash mismatch")
    if shutil.which("zip") is None:
        raise RuntimeError("Info-ZIP is required")
    with zipfile.ZipFile(BASE) as source:
        source_script = source.read("script.py").decode("utf-8")
        source_names = set(source.namelist())
        source_meta = {
            info.filename: (info.CRC, info.file_size, info.compress_size)
            for info in source.infolist()
        }
    script = patched_script(source_script)
    with tempfile.TemporaryDirectory(prefix="v99-build-", dir=ROOT / "reports") as raw:
        directory = Path(raw)
        (directory / "script.py").write_text(script, encoding="utf-8")
        for name, source in NEW_ENTRIES.items():
            target = directory / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
        archive = directory / "candidate.zip"
        shutil.copyfile(BASE, archive)
        subprocess.run(
            ["zip", "-q", str(archive), "script.py", *NEW_ENTRIES],
            cwd=directory, check=True,
        )
        with zipfile.ZipFile(archive) as candidate:
            current = inventory(candidate)
            if current["duplicates"] or current["unsafe_paths"]:
                raise ValueError(current)
            if set(current["top_level"]) != {"model", "script.py", "requirements.txt"}:
                raise ValueError(current["top_level"])
            expected_names = source_names | set(NEW_ENTRIES)
            if set(candidate.namelist()) != expected_names:
                raise ValueError("unexpected archive member set")
            for name, metadata in source_meta.items():
                if name == "script.py":
                    continue
                info = candidate.getinfo(name)
                if (info.CRC, info.file_size, info.compress_size) != metadata:
                    raise ValueError(f"base member changed: {name}")
            if candidate.read("script.py") != script.encode("utf-8"):
                raise ValueError("script payload mismatch")
            for name, source in NEW_ENTRIES.items():
                if hashlib.sha256(candidate.read(name)).hexdigest() != sha256_file(source):
                    raise ValueError(f"new member payload mismatch: {name}")
            if candidate.testzip() is not None:
                raise ValueError("CRC verification failed")
        if archive.stat().st_size >= 10_000_000_000:
            raise ValueError("compressed archive exceeds 10 GB")
        if current["expanded_bytes"] >= 32_000_000_000:
            raise ValueError("expanded archive exceeds 32 GB")
        digest = sha256_file(archive)
        os.replace(archive, OUTPUT)
    report = {
        "status": "complete_archive_validation",
        "archive": str(OUTPUT), "sha256": digest,
        "bytes": OUTPUT.stat().st_size, "inventory": current,
        "base_archive": str(BASE), "base_sha256": base_report["sha256"],
        "checkpoint_sha256": sha256_file(CHECKPOINT),
        "changed_outputs": ["FILE_FAKE_PROB"],
        "file_expert_weight": .40, "views": 3,
        "all_crc_verified": True, "official_submitted": False,
        "full_pipeline_smoke_verified": False,
    }
    OUTPUT.with_suffix(".report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
