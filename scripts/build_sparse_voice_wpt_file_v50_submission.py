#!/usr/bin/env python3
"""Add the frozen seed06 WPT File residual to a sparse-Voice v49 archive."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil
import tempfile

try:
    from .build_sparse_voice_v49_submission import (
        inspect_archive,
        make_zip,
        safe_extract,
        sha256_file,
        validate_package_tree,
    )
except ImportError:  # pragma: no cover - direct script execution
    from build_sparse_voice_v49_submission import (
        inspect_archive,
        make_zip,
        safe_extract,
        sha256_file,
        validate_package_tree,
    )


ROOT = Path(__file__).resolve().parents[1]
WPT_CHECKPOINT_SHA256 = (
    "584c5abf964d7b2e772905503c084daac45a83c52755ddb12291b77447cbe37e"
)
SPECTRA_MODEL_SHA256 = (
    "2e2727a7397f78d28b0a2a2b8ee031ff08143b9c431ea7f06fc29a808b0180db"
)
SPECTRA_CODE_SHA256 = (
    "0e8141a7b182463737f685e1593427e4cc25f0d21042523fb308706789d6d207"
)
IMPORT_MARKER = (
    "from component_query_mhfa_inference import "
    "apply_component_query_mhfa_fusion  # noqa: E402\n"
)
IMPORT_INSERT = IMPORT_MARKER + (
    "from wpt_file_expert import apply_wpt_file_expert  # noqa: E402\n"
)
QUERY_MARKER = '''    apply_component_query_mhfa_fusion(
        args.output, eat_patch_graph, spear_component_bins,
        BASE_DIR / "model" / "component-query" / "head.pt",
        device=args.device, file_weight=0.025,
        music_weight=0.05,
        file_or_weight=0.3,
    )
'''
WPT_TEMPLATE = QUERY_MARKER + '''    # Frozen seed06 original-audio WPT: File only.
    apply_wpt_file_expert(
        args.test_dir, args.output, BASE_DIR / "model" / "spectra-aasist",
        BASE_DIR / "model" / "spectra-aasist" / "wpt_spectra_multitask.pt",
        device=args.device, file_batch_size=6,
        file_views=5, file_temperature=2.0,
        wpt_weight={file_weight:.8g},
    )
'''


def replace_once(text: str, marker: str, replacement: str, label: str) -> str:
    if text.count(marker) != 1:
        raise ValueError(f"v49 {label} marker changed; refusing unsafe injection")
    return text.replace(marker, replacement)


def patch_entrypoint(source: str, *, file_weight: float = .40) -> str:
    """Inject the frozen 5-view/T=2 File-only residual."""
    if not 0 <= file_weight <= .50:
        raise ValueError("cross-development guard limits WPT File weight to 50%")
    if "from wpt_file_expert import apply_wpt_file_expert" in source:
        raise ValueError("v49 WPT File marker changed; refusing unsafe injection")
    source = replace_once(source, IMPORT_MARKER, IMPORT_INSERT, "import")
    return replace_once(
        source,
        QUERY_MARKER,
        WPT_TEMPLATE.format(file_weight=file_weight),
        "WPT File fusion",
    )


def validate_shared_spectra(package: Path) -> Path:
    model_dir = Path(package) / "model/spectra-aasist"
    required = {
        model_dir / "model.safetensors": SPECTRA_MODEL_SHA256,
        model_dir / "model.py": SPECTRA_CODE_SHA256,
    }
    for path, expected in required.items():
        if not path.is_file() or sha256_file(path) != expected:
            raise ValueError(f"v49 does not contain the frozen shared asset: {path}")
    if not (model_dir / "xlsr_config/config.json").is_file():
        raise ValueError("shared Spectra asset has no offline XLS-R config")
    return model_dir


def build_package(
    base_archive: Path,
    checkpoint: Path,
    output: Path,
    *,
    file_weight: float = .40,
    archive: bool = False,
) -> Path | None:
    base_archive = Path(base_archive).resolve()
    checkpoint = Path(checkpoint).resolve()
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    archive_path = output.with_suffix(".zip")
    if archive and archive_path.exists():
        raise FileExistsError(f"Refusing to overwrite {archive_path}")
    inspect_archive(base_archive)
    if sha256_file(checkpoint) != WPT_CHECKPOINT_SHA256:
        raise ValueError("WPT checkpoint is not frozen seed06")

    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".wpt-file-v50-", dir=output.parent))
    try:
        safe_extract(base_archive, staging)
        model_dir = validate_shared_spectra(staging)
        source_dir = staging / "model/src"
        for name in (
            "wpt_spectra.py", "wpt_spectra_inference.py", "wpt_file_expert.py",
        ):
            shutil.copy2(ROOT / "src" / name, source_dir / name)
        shutil.copy2(checkpoint, model_dir / "wpt_spectra_multitask.pt")
        entrypoint = staging / "script.py"
        entrypoint.write_text(
            patch_entrypoint(
                entrypoint.read_text(encoding="utf-8"),
                file_weight=file_weight,
            ),
            encoding="utf-8",
        )
        validate_package_tree(staging)
        os.replace(staging, output)
        staging = None
    finally:
        if staging is not None:
            shutil.rmtree(staging, ignore_errors=True)

    if not archive:
        return None
    temporary = archive_path.with_name(f".{archive_path.name}.tmp")
    try:
        make_zip(output, temporary)
        inspect_archive(temporary)
        os.replace(temporary, archive_path)
    finally:
        temporary.unlink(missing_ok=True)
    return archive_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-zip", type=Path, required=True)
    parser.add_argument(
        "--checkpoint", type=Path,
        default=ROOT / "reports/wpt_spectra_v1/seed_20260906/wpt_spectra_multitask.pt",
    )
    parser.add_argument("--file-weight", type=float, default=.40)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--archive", action="store_true")
    args = parser.parse_args()
    built = build_package(
        args.base_zip, args.checkpoint, args.output,
        file_weight=args.file_weight, archive=args.archive,
    )
    print(
        f"Built {args.output}"
        + ("" if built is None else f" and {built}")
    )


if __name__ == "__main__":
    main()
