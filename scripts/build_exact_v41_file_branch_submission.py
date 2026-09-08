#!/usr/bin/env python3
"""Add an exact v41 File branch to v47 without repeating the base pipeline.

The entrypoint snapshots the common post-invariant CSV, then evaluates the
small segmental/unified heads and the original-audio WPT expert on that branch.
EAT and SPEAR statistics are exported during v47's existing encoder passes.
Only the final branch File logit is blended back into v47; all other v47
outputs remain untouched.
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import shutil
import zipfile


ROOT = Path(__file__).resolve().parents[1]
SPECTRA_MODEL_SHA256 = (
    "2e2727a7397f78d28b0a2a2b8ee031ff08143b9c431ea7f06fc29a808b0180db"
)
SPECTRA_CODE_SHA256 = (
    "0e8141a7b182463737f685e1593427e4cc25f0d21042523fb308706789d6d207"
)

PYTHON_IMPORT_MARKER = "from pathlib import Path\n"
PYTHON_IMPORT_INSERT = PYTHON_IMPORT_MARKER + "from shutil import copy2\n"
ENTRY_IMPORT_MARKER = (
    "from component_query_mhfa_inference import "
    "apply_component_query_mhfa_fusion  # noqa: E402\n"
)
ENTRY_IMPORT_INSERT = ENTRY_IMPORT_MARKER + (
    "from segmental_eat_music_inference import "
    "apply_segmental_eat_music_fusion  # noqa: E402\n"
    "from unified_dual_ssl_inference import "
    "apply_unified_music_fusion  # noqa: E402\n"
    "from wpt_spectra_inference import "
    "apply_wpt_fixed_moe_fusion  # noqa: E402\n"
    "from wpt_file_expert import "
    "apply_file_submission_blend  # noqa: E402\n"
)
VARIABLE_MARKER = (
    '    spear_component_bins = BASE_DIR / "output" / ".spear_component_bins.npz"\n'
)
VARIABLE_INSERT = VARIABLE_MARKER + '''    v41_branch = BASE_DIR / "output" / ".v41_file_branch.csv"
    eat_hierarchical = BASE_DIR / "output" / ".v41_eat_hierarchical.npz"
    eat_segmental = BASE_DIR / "output" / ".v41_eat_segmental.npz"
    spear_unified = BASE_DIR / "output" / ".v41_spear_unified.npz"
    telephone_ids = BASE_DIR / "output" / ".v41_telephone_ids.npz"
    unified_expert = BASE_DIR / "output" / ".v41_unified_expert.npz"
'''
EAT_MARKER = '''        patch_graph_checkpoint_path=(
            BASE_DIR / "model" / "eat-patch-graph" / "head.pt"
        ),
    )
'''
EAT_INSERT = '''        patch_graph_checkpoint_path=(
            BASE_DIR / "model" / "eat-patch-graph" / "head.pt"
        ),
        hierarchical_statistics_output_path=eat_hierarchical,
        hierarchical_checkpoint_path=(
            BASE_DIR / "model" / "hierarchical-eat-music" / "head_00.pt"
        ),
        segmental_statistics_output_path=eat_segmental,
        segmental_checkpoint_path=(
            BASE_DIR / "model" / "segmental-eat-music" / "head_00.pt"
        ),
        telephone_ids_output_path=telephone_ids,
    )
'''
SPEAR_MARKER = '''        temporal_bin_checkpoint_path=(
            BASE_DIR / "model" / "component-query" / "head.pt"
        ),
    )
'''
SPEAR_INSERT = '''        temporal_bin_checkpoint_path=(
            BASE_DIR / "model" / "component-query" / "head.pt"
        ),
        additional_temporal_bin_requests=[(
            spear_unified,
            BASE_DIR / "model" / "unified-dual-ssl" / "head_00.pt",
        )],
    )
'''
DUAL_MARKER = '''    apply_dual_domain_fusion(
        args.output, eat_stats, spear_stats,
        [BASE_DIR / "model" / "channel-invariant" / "dual_domain_head.pt"],
        device=args.device, file_weight=0.05,
        voice_weight=0.05, music_weight=0.05,
    )
'''
DUAL_INSERT = DUAL_MARKER + '''    # v41 and v47 are identical through this point.
    copy2(args.output, v41_branch)
'''
QUERY_MARKER = '''    apply_component_query_mhfa_fusion(
        args.output, eat_patch_graph, spear_component_bins,
        BASE_DIR / "model" / "component-query" / "head.pt",
        device=args.device, file_weight=0.025,
        music_weight=0.05,
        file_or_weight=0.3,
    )
'''
BRANCH_TEMPLATE = QUERY_MARKER + '''    apply_segmental_eat_music_fusion(
        v41_branch, eat_hierarchical,
        sorted((BASE_DIR / "model/hierarchical-eat-music").glob("head_*.pt")),
        eat_segmental,
        sorted((BASE_DIR / "model/segmental-eat-music").glob("head_*.pt")),
        device=args.device, telephone_ids_path=telephone_ids,
        expert_weight=0.25,
        music_weight=0.20, file_weight=0.10,
        phone_music_weight=0.30, phone_file_weight=0.30,
        file_music_presence_threshold=0.50,
    )
    apply_unified_music_fusion(
        v41_branch, eat_hierarchical, spear_unified,
        sorted((BASE_DIR / "model" / "unified-dual-ssl").glob("head_*.pt")),
        device=args.device, music_weight=0.20,
        expert_output_path=unified_expert,
    )
    apply_wpt_fixed_moe_fusion(
        args.test_dir, v41_branch, BASE_DIR / "model" / "{spectra_subdir}",
        BASE_DIR / "model" / "{spectra_subdir}" / "wpt_spectra_multitask.pt",
        unified_expert, device=args.device, file_batch_size=6,
        file_views=5, file_temperature=2.0,
        voice_outer_weight=0.10, file_outer_weight=0.75,
    )
    apply_file_submission_blend(
        args.output, v41_branch, branch_weight={branch_weight:.8g},
    )
    for path in (
        v41_branch, eat_hierarchical, eat_segmental, spear_unified,
        telephone_ids, unified_expert,
    ):
        path.unlink(missing_ok=True)
'''


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def replace_once(text: str, marker: str, replacement: str, label: str) -> str:
    if text.count(marker) != 1:
        raise ValueError(f"v47 {label} marker changed; refusing unsafe injection")
    return text.replace(marker, replacement)


def patch_entrypoint(
    source: str, *, spectra_subdir: str, branch_weight: float,
) -> str:
    if not spectra_subdir or "/" in spectra_subdir or "\\" in spectra_subdir:
        raise ValueError("spectra_subdir must be one directory name")
    if not 0 <= branch_weight <= 1:
        raise ValueError("branch weight must lie in [0, 1]")
    replacements = (
        (PYTHON_IMPORT_MARKER, PYTHON_IMPORT_INSERT, "Python import"),
        (ENTRY_IMPORT_MARKER, ENTRY_IMPORT_INSERT, "expert imports"),
        (VARIABLE_MARKER, VARIABLE_INSERT, "branch variables"),
        (EAT_MARKER, EAT_INSERT, "EAT multi-export"),
        (SPEAR_MARKER, SPEAR_INSERT, "SPEAR multi-export"),
        (DUAL_MARKER, DUAL_INSERT, "branch snapshot"),
        (
            QUERY_MARKER,
            BRANCH_TEMPLATE.format(
                spectra_subdir=spectra_subdir,
                branch_weight=branch_weight,
            ),
            "v41 File branch",
        ),
    )
    for marker, replacement, label in replacements:
        source = replace_once(source, marker, replacement, label)
    return source


def copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def choose_spectra_target(output: Path, v41_package: Path) -> tuple[Path, str]:
    """Reuse sparse-v24 Spectra weights when the base already contains them."""
    reusable = output / "model/spectra-aasist"
    if reusable.is_dir():
        if (
            sha256_file(reusable / "model.safetensors") != SPECTRA_MODEL_SHA256
            or sha256_file(reusable / "model.py") != SPECTRA_CODE_SHA256
        ):
            raise ValueError("existing spectra-aasist asset is not frozen v24/v41")
        return reusable, "spectra-aasist"
    target = output / "model/wpt-file-expert"
    source = v41_package / "model/wpt-spectra"
    for name in ("model.py", "model.safetensors", "README.md", "SOURCE.md"):
        path = source / name
        if path.is_file():
            copy(path, target / name)
    copy(source / "xlsr_config/config.json", target / "xlsr_config/config.json")
    return target, "wpt-file-expert"


def make_zip(source: Path, archive: Path) -> None:
    with zipfile.ZipFile(
        archive, "w", compression=zipfile.ZIP_DEFLATED,
        compresslevel=1, allowZip64=True,
    ) as handle:
        for path in sorted(source.rglob("*")):
            if path.is_dir() or "__pycache__" in path.parts or path.suffix == ".pyc":
                continue
            handle.write(path, path.relative_to(source).as_posix())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base-zip", type=Path, default=ROOT / "component_query_or_v47.zip"
    )
    parser.add_argument(
        "--v41-package", type=Path, default=ROOT / "wpt_file_lme_v41"
    )
    parser.add_argument("--branch-weight", type=float, default=.70)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--archive", action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output}")
    archive = args.output.with_suffix(".zip")
    if args.archive and archive.exists():
        raise FileExistsError(f"Refusing to overwrite {archive}")
    if not args.base_zip.is_file() or not args.v41_package.is_dir():
        parser.error("base ZIP and frozen v41 package are required")

    shutil.unpack_archive(str(args.base_zip), str(args.output), format="zip")
    source_dir = args.output / "model/src"
    for name in (
        "anchor_spear_stats_fusion.py", "spear_detector.py",
        "spear_temporal_bins.py", "wpt_file_expert.py",
    ):
        copy(ROOT / "src" / name, source_dir / name)
    for name in (
        "hierarchical_eat_music.py", "hierarchical_eat_music_inference.py",
        "segmental_eat_music.py", "segmental_eat_music_inference.py",
        "unified_dual_ssl_head.py", "unified_dual_ssl_inference.py",
        "wpt_spectra.py", "wpt_spectra_inference.py",
    ):
        copy(args.v41_package / "model/src" / name, source_dir / name)
    for directory in (
        "hierarchical-eat-music", "segmental-eat-music", "unified-dual-ssl",
    ):
        for path in (args.v41_package / "model" / directory).glob("*"):
            if path.is_file():
                copy(path, args.output / "model" / directory / path.name)
    spectra_target, spectra_subdir = choose_spectra_target(
        args.output, args.v41_package
    )
    copy(
        args.v41_package / "model/wpt-spectra/wpt_spectra_multitask.pt",
        spectra_target / "wpt_spectra_multitask.pt",
    )

    entrypoint = args.output / "script.py"
    entrypoint.write_text(
        patch_entrypoint(
            entrypoint.read_text(encoding="utf-8"),
            spectra_subdir=spectra_subdir,
            branch_weight=args.branch_weight,
        ),
        encoding="utf-8",
    )
    if args.archive:
        make_zip(args.output, archive)
        print(f"Built {args.output} and {archive}")
    else:
        print(f"Built {args.output}")


if __name__ == "__main__":
    main()
