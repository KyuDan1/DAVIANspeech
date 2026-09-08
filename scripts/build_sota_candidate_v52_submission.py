#!/usr/bin/env python3
"""Build the frozen v52 package: Music v51, then final File consistency."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

try:
    from .build_component_consistent_file_v51_submission import build_package
except ImportError:  # pragma: no cover - direct CLI execution
    from build_component_consistent_file_v51_submission import build_package


ROOT = Path(__file__).resolve().parents[1]
EXPECTED_TOP_LEVEL = {"model", "script.py", "requirements.txt"}
EXPECTED_BASE_ASSETS = {
    "script.py": "e3f0ad9be9f01876fc89872f897a6389472fbcf940a90c2967046d734225a4dd",
    "requirements.txt": "978befe6e5c8a064cddd2690e7e55b4b153f87eabf0dcf3b0604a567f4edd6ed",
    "model/src/music_only_residual.py": "d8bd7e51ed4ac68b7a3e23e5f964c19ba197cc5bf9051f7a0f0e6ccae3d211bf",
    "model/music-only-v51/head_00.pt": "32d846904629c7a99c253fdd7fbd7931176ea919c20652344a7422d5e4f6a3dd",
    "model/music-only-v51/head_01.pt": "b32c8fdcef7f4aa34d1d78833fc5a18b35e9664a49a626de4ae2111b78a15316",
    "model/music-only-v51/head_02.pt": "8cd61a05bfcf974ad396a828a0a06345ff87a959ae0f3b5a0db24a2103b3c0a8",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_top_level(package: Path) -> None:
    entries = {path.name for path in Path(package).iterdir()}
    if entries != EXPECTED_TOP_LEVEL:
        raise ValueError(
            f"package top-level must be {sorted(EXPECTED_TOP_LEVEL)}, got {sorted(entries)}"
        )
    if not (Path(package) / "model").is_dir():
        raise ValueError("package model entry must be a directory")


def validate_music_base(base: Path) -> None:
    base = Path(base)
    if not base.is_dir():
        raise FileNotFoundError(f"missing frozen Music v51 package: {base}")
    validate_top_level(base)
    for relative, expected in EXPECTED_BASE_ASSETS.items():
        path = base / relative
        if not path.is_file() or sha256(path) != expected:
            raise ValueError(f"frozen Music v51 asset mismatch: {relative}")


def validate_execution_order(source: str) -> None:
    markers = (
        "    apply_component_query_music_residual(",
        "    apply_wpt_file_expert(",
        "    apply_spectra_voice_fusion(",
        "    apply_sparse_call_consensus(",
        "    apply_component_consistent_file_fusion(",
        "    for path in (",
    )
    positions = []
    for marker in markers:
        if source.count(marker) != 1:
            raise ValueError(f"v52 execution marker count differs: {marker.strip()}")
        positions.append(source.index(marker))
    if positions != sorted(positions):
        raise ValueError("v52 File consistency must run after every specialist")


def build_sota_candidate(base: Path, output: Path) -> None:
    validate_music_base(base)
    build_package(
        base, output,
        file_weight=0.20,
        voice_threshold=0.10,
        music_threshold=0.20,
    )
    validate_top_level(output)
    validate_execution_order((Path(output) / "script.py").read_text("utf-8"))
    runtime = Path(output) / "model/src/component_consistent_file_fusion.py"
    if sha256(runtime) != sha256(ROOT / "src/component_consistent_file_fusion.py"):
        raise ValueError("generated File consistency runtime differs from source")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base", type=Path, default=ROOT / "music_only_v51_frozen",
    )
    parser.add_argument(
        "--output", type=Path, default=ROOT / "sota_candidate_v52_frozen",
    )
    args = parser.parse_args()
    build_sota_candidate(args.base, args.output)
    print(f"Built {args.output} from frozen Music v51")


if __name__ == "__main__":
    main()
