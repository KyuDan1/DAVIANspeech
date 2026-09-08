#!/usr/bin/env python3
"""Build frozen v50 + Forensics Voice 7.5% + final File consistency."""

from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path
import shutil

try:
    from .build_sparse_voice_v49_submission import (
        MAX_EXPANDED_BYTES, validate_package_tree,
    )
except ImportError:  # pragma: no cover - direct CLI execution
    from build_sparse_voice_v49_submission import (
        MAX_EXPANDED_BYTES, validate_package_tree,
    )


ROOT = Path(__file__).resolve().parents[1]
EXPECTED_TOP_LEVEL = {"model", "requirements.txt", "script.py"}
BASE_ASSETS = {
    "script.py": "238cc39a493e9df9e5a9e088c9f97494641e03ddee923df3c04c8f7806897713",
    "requirements.txt": "978befe6e5c8a064cddd2690e7e55b4b153f87eabf0dcf3b0604a567f4edd6ed",
    "model/spectra-aasist/wpt_spectra_multitask.pt": "584c5abf964d7b2e772905503c084daac45a83c52755ddb12291b77447cbe37e",
    "model/sparse-call-consensus.npz": "bbe68298bb86cdb103a10b4caa3fb6b983cc2defa1617166dd30aa89a3a4f128",
}
EXTERNAL_ASSETS = {
    "checkpoint_epoch_4.safetensors": "39e37fa5a958c2b46ebb6a5937874c36c39003906dcdd5d0aba6154bc6b2dc21",
    "model.py": "d79bc120cc78f8244e104113e4de67aed59469f11706728d0c465fc85a480944",
    "config.json": "c572c68e057b56cc78e9ebb0f6d3be49f1ce487b41e68a1db909e3e129f82a80",
    "xlsr_config/config.json": "532c26cf268de70bb3d3eaf0ac0b7ba8cb2f09d01b2278d14e91f58d1f1764f6",
    "SOURCE.md": "42b11635b5b89b3644ad18825582af5b2266cc62108514332a180d00847eb90f",
    "README.md": "a495096d360de019b47ff049a8fbbe0232ff55b8d338298b70ef83d33f57e6d1",
}
IMPORT_MARKER = (
    "from sparse_call_consensus import apply_sparse_call_consensus  # noqa: E402\n"
)
IMPORT_INSERT = IMPORT_MARKER + (
    "from forensics_xlsr_wild_fusion import "
    "apply_forensics_voice_fusion  # noqa: E402\n"
    "from component_consistent_file_fusion import "
    "apply_component_consistent_file_fusion  # noqa: E402\n"
)
CLEANUP_MARKER = "    for path in (\n"
FINAL_STAGES = '''    # Frozen external diversity expert: original-mixture Voice only.
    apply_forensics_voice_fusion(
        args.test_dir, args.output,
        BASE_DIR / "model" / "forensics-xlsr-wild",
        device=args.device, voice_weight=0.075,
        windows=3, window_samples=80_000, file_batch_size=8,
    )
    # Must run last so File consumes the final Voice/Music probabilities.
    apply_component_consistent_file_fusion(
        args.output, file_weight=0.20,
        voice_presence_threshold=0.10,
        music_presence_threshold=0.20,
    )
''' + CLEANUP_MARKER


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def replace_once(source: str, marker: str, replacement: str, label: str) -> str:
    if source.count(marker) != 1:
        raise ValueError(f"v50 {label} marker changed; refusing unsafe patch")
    return source.replace(marker, replacement)


def patch_entrypoint(source: str) -> str:
    """Insert fixed Voice and File stages after all frozen v50 specialists."""
    if "apply_forensics_voice_fusion" in source:
        raise ValueError("Forensics stage already exists")
    if "apply_component_consistent_file_fusion" in source:
        raise ValueError("component-consistent File stage already exists")
    source = replace_once(source, IMPORT_MARKER, IMPORT_INSERT, "import")
    return replace_once(source, CLEANUP_MARKER, FINAL_STAGES, "cleanup")


def validate_execution_order(source: str) -> None:
    markers = (
        "    apply_sparse_call_consensus(",
        "    apply_forensics_voice_fusion(",
        "    apply_component_consistent_file_fusion(",
        "    for path in (",
    )
    positions = []
    for marker in markers:
        if source.count(marker) != 1:
            raise ValueError(f"v54 execution marker count differs: {marker.strip()}")
        positions.append(source.index(marker))
    if positions != sorted(positions):
        raise ValueError("v54 stages are not in frozen execution order")


def validate_hashes(root: Path, expected: dict[str, str], label: str) -> None:
    for relative, wanted in expected.items():
        path = Path(root) / relative
        if not path.is_file() or sha256(path) != wanted:
            raise ValueError(f"{label} asset mismatch: {relative}")


def replace_copy(source: Path, destination: Path) -> None:
    """Copy a mutable file after removing a possible shared hardlink inode."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.unlink(missing_ok=True)
    shutil.copy2(source, destination)


def replace_link(source: Path, destination: Path) -> None:
    """Publish a read-only frozen asset as a regular hardlinked file."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.unlink(missing_ok=True)
    os.link(source, destination)


def package_size(package: Path) -> int:
    return sum(path.stat().st_size for path in Path(package).rglob("*") if path.is_file())


def validate_base_tree(base: Path) -> None:
    """Accept only harmless bytecode beside the exact v50 submission tree."""
    entries = {path.name for path in Path(base).iterdir()}
    if entries - {"__pycache__"} != EXPECTED_TOP_LEVEL:
        raise ValueError(f"unexpected frozen v50 structure: {sorted(entries)}")
    if not (Path(base) / "model").is_dir():
        raise ValueError("frozen v50 model entry must be a directory")
    if any(path.is_symlink() for path in Path(base).rglob("*")):
        raise ValueError("frozen v50 contains a symlink")


def build_package(base: Path, external: Path, output: Path) -> None:
    base, external, output = map(lambda path: Path(path).resolve(), (base, external, output))
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    validate_base_tree(base)
    validate_hashes(base, BASE_ASSETS, "frozen v50")
    validate_hashes(external, EXTERNAL_ASSETS, "Forensics")

    shutil.copytree(
        base, output, copy_function=os.link,
        ignore=shutil.ignore_patterns("data", "open", "output", "__pycache__", "*.pyc"),
    )
    try:
        replace_copy(base / "script.py", output / "script.py")
        script = patch_entrypoint((output / "script.py").read_text("utf-8"))
        (output / "script.py").write_text(script, encoding="utf-8")
        replace_copy(
            ROOT / "src/forensics_xlsr_wild_fusion.py",
            output / "model/src/forensics_xlsr_wild_fusion.py",
        )
        replace_copy(
            ROOT / "src/component_consistent_file_fusion.py",
            output / "model/src/component_consistent_file_fusion.py",
        )
        target = output / "model/forensics-xlsr-wild"
        for relative in EXTERNAL_ASSETS:
            replace_link(external / relative, target / relative)

        validate_package_tree(output)
        validate_hashes(target, EXTERNAL_ASSETS, "packaged Forensics")
        validate_execution_order(script)
        compile(script, str(output / "script.py"), "exec")
        compile(
            (output / "model/src/forensics_xlsr_wild_fusion.py").read_text("utf-8"),
            "forensics_xlsr_wild_fusion.py", "exec",
        )
        if package_size(output) > MAX_EXPANDED_BYTES:
            raise ValueError("v54 expanded package exceeds 32 GB")
    except Exception:
        shutil.rmtree(output, ignore_errors=True)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base", type=Path, default=ROOT / "sparse_voice_wpt_file_v50_frozen",
    )
    parser.add_argument(
        "--external", type=Path,
        default=ROOT / "models/external/forensics_xlsr_wild",
    )
    parser.add_argument(
        "--output", type=Path, default=ROOT / "forensics_voice_file_v54_frozen",
    )
    args = parser.parse_args()
    build_package(args.base, args.external, args.output)
    print(f"Built {args.output} from frozen v50")


if __name__ == "__main__":
    main()
