#!/usr/bin/env python3
"""Build exact v47 plus the frozen v24 Voice-only sparse-call expert.

The patch deliberately leaves v47 File, Music, and both Presence columns
unchanged.  The already-computed vocal XLS-R windows are retained while the
vocal stem is also queued for Spectra scoring inside the existing separation
loop.  No second XLS-R pass is introduced.
"""

from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path, PurePosixPath
import shutil
import stat
import tempfile
import zipfile


ROOT = Path(__file__).resolve().parents[1]
BASE_ARCHIVE_SHA256 = (
    "0ddb1d9d3d21def4f2e3b761e074f52145db45caebfe5c2198976a0af2ce25ed"
)
MAX_EXPANDED_BYTES = 32_000_000_000
MAX_ARCHIVE_BYTES = 10_000_000_000
REQUIRED_TOP_LEVEL = {"model", "requirements.txt", "script.py"}

# These are copied from the already-built v24 package, not from a mutable
# checkout module or training report.  Hash pinning makes "frozen v24" an
# executable property of the builder.
FROZEN_V24_ASSETS = {
    "model/src/spectra_aasist_detector.py": (
        "c3de01da2bcd314523c836e8198f2042ea34093c1dcc8f92cdf60036638808fe"
    ),
    "model/src/sparse_call_consensus.py": (
        "24f348682754b6f48b17806d0edae48d6b5f5a37ec71d5275f46a94408701be1"
    ),
    "model/spectra-aasist/model.py": (
        "0e8141a7b182463737f685e1593427e4cc25f0d21042523fb308706789d6d207"
    ),
    "model/spectra-aasist/model.safetensors": (
        "2e2727a7397f78d28b0a2a2b8ee031ff08143b9c431ea7f06fc29a808b0180db"
    ),
    "model/spectra-aasist/xlsr_config/config.json": (
        "ba742f733410ca021711e10e0d5902f09af4845469d32bbb05063ae9ac6df32b"
    ),
    "model/sparse-call-consensus.npz": (
        "bbe68298bb86cdb103a10b4caa3fb6b983cc2defa1617166dd30aa89a3a4f128"
    ),
}

PIPELINE_IMPORT_MARKER = (
    "from xlsr_antideepfake import XlsrAntiDeepfake  # noqa: E402\n"
)
PIPELINE_IMPORT_INSERT = PIPELINE_IMPORT_MARKER + (
    "from spectra_aasist_detector import SpectraStemScorer  # noqa: E402\n"
)
FAKE_SIGNATURE_MARKER = (
    "def fake_probability(detector, audio, device, window, batch_size,\n"
    "                     pooling=\"max\", temperature=5.0):"
)
FAKE_SIGNATURE_INSERT = (
    "def fake_probability(detector, audio, device, window, batch_size,\n"
    "                     pooling=\"max\", temperature=5.0, profile=None,\n"
    "                     profile_id=None):"
)
FAKE_EMPTY_MARKER = """    if audio.size == 0:
        return 0.0
    rms = float(np.sqrt(np.mean(np.square(audio, dtype=np.float64))))
    if rms < SILENCE_RMS:
        # Nothing to judge: an empty stem must not create fake evidence.
        return 0.0

    windows = np.stack([
        extract_segment(audio, start, window)
        for start in segment_starts(audio.size, window)
    ])
"""
FAKE_EMPTY_INSERT = """    if audio.size == 0:
        audio = np.empty(0, dtype=np.float32)
    rms = float(np.sqrt(np.mean(np.square(audio, dtype=np.float64)))) if (
        audio.size
    ) else 0.0
    if rms < SILENCE_RMS:
        # Retain one aligned placeholder. Spectra's valid flag suppresses it.
        if profile is not None:
            profile["ids"].append(str(profile_id))
            profile["starts"].append(0)
            profile["scores"].append(0.0)
            profile["offsets"].append(len(profile["scores"]))
            profile["durations"].append(audio.size / AUDIO_SR)
        return 0.0

    starts = segment_starts(audio.size, window)
    windows = np.stack([
        extract_segment(audio, start, window) for start in starts
    ])
"""
FAKE_RETURN_MARKER = (
    "    return pool_window_scores(np.asarray(scores), pooling, temperature)\n"
)
FAKE_RETURN_INSERT = """    values = np.asarray(scores)
    if profile is not None:
        profile["ids"].append(str(profile_id))
        profile["starts"].extend(starts)
        profile["scores"].extend(values.tolist())
        profile["offsets"].append(len(profile["scores"]))
        profile["durations"].append(audio.size / AUDIO_SR)
    return pool_window_scores(values, pooling, temperature)
"""
SCORER_MARKER = (
    "    artifact_detector = ArtifactNetMusicDetector(args.artifactnet_dir)\n\n"
)
SCORER_INSERT = SCORER_MARKER + """    spectra_scorer = SpectraStemScorer(
        args.spectra_model_dir, device=args.device,
        windows=args.spectra_windows,
        file_batch_size=args.spectra_file_batch_size,
        collect_sliding=True,
    )
    xlsr_profile = {
        "ids": [], "offsets": [0], "starts": [], "scores": [],
        "durations": [],
    }

"""
VOICE_CALL_MARKER = """        voice_audio, music_audio = separator.separate(path)
        voice_fake = fake_probability(
            detector, voice_audio, device, args.window, args.batch_size,
            args.pooling, args.temperature
        )
"""
VOICE_CALL_INSERT = """        voice_audio, music_audio = separator.separate(path)
        spectra_scorer.add(path.stem, voice_audio)
        voice_fake = fake_probability(
            detector, voice_audio, device, args.window, args.batch_size,
            args.pooling, args.temperature, profile=xlsr_profile,
            profile_id=path.stem,
        )
"""
WRITE_MARKER = (
    '    print(f"[3/3] Writing {args.output}", flush=True)\n'
    "    args.output.parent.mkdir(parents=True, exist_ok=True)\n"
)
WRITE_INSERT = """    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.xlsr_consensus_statistics_output,
        ids=np.asarray(xlsr_profile["ids"]),
        offsets=np.asarray(xlsr_profile["offsets"], dtype=np.int64),
        starts=np.asarray(xlsr_profile["starts"], dtype=np.int64),
        scores=np.asarray(xlsr_profile["scores"], dtype=np.float32),
        durations=np.asarray(xlsr_profile["durations"], dtype=np.float32),
        window=np.asarray(args.window, dtype=np.int64),
    )
    spectra_scorer.save(
        args.spectra_statistics_output,
        args.spectra_consensus_statistics_output,
    )
    print(f"[3/3] Writing {args.output}", flush=True)
"""

ENTRY_IMPORT_MARKER = (
    "from component_query_mhfa_inference import "
    "apply_component_query_mhfa_fusion  # noqa: E402\n"
)
ENTRY_IMPORT_INSERT = ENTRY_IMPORT_MARKER + (
    "from spectra_aasist_detector import apply_spectra_voice_fusion  # noqa: E402\n"
    "from sparse_call_consensus import apply_sparse_call_consensus  # noqa: E402\n"
)
ENTRY_VARIABLE_MARKER = (
    '    spear_component_bins = BASE_DIR / "output" / ".spear_component_bins.npz"\n'
)
ENTRY_VARIABLE_INSERT = ENTRY_VARIABLE_MARKER + """    spectra_stats = BASE_DIR / "output" / ".spectra_voice_v49.npz"
    xlsr_consensus = BASE_DIR / "output" / ".xlsr_voice_v49.npz"
    spectra_consensus = BASE_DIR / "output" / ".spectra_sliding_v49.npz"
    args.spectra_model_dir = BASE_DIR / "model" / "spectra-aasist"
    args.spectra_statistics_output = spectra_stats
    args.xlsr_consensus_statistics_output = xlsr_consensus
    args.spectra_consensus_statistics_output = spectra_consensus
    args.spectra_windows = 3
    args.spectra_file_batch_size = 4
"""
QUERY_MARKER = """    apply_component_query_mhfa_fusion(
        args.output, eat_patch_graph, spear_component_bins,
        BASE_DIR / "model" / "component-query" / "head.pt",
        device=args.device, file_weight=0.025,
        music_weight=0.05,
        file_or_weight=0.3,
    )
"""
VOICE_FUSION_INSERT = QUERY_MARKER + """    # Frozen v24 experts: Voice only. All File update weights stay zero.
    apply_spectra_voice_fusion(
        args.output, spectra_stats, voice_weight=0.10, file_weight=0.0,
        voice_presence_gate=0.50,
    )
    apply_sparse_call_consensus(
        args.output, xlsr_consensus, spectra_consensus,
        BASE_DIR / "model" / "sparse-call-consensus.npz",
        voice_weight=0.60,
        file_voice_only_weight=0.0,
    )
"""
ENTRY_CLEANUP_MARKER = (
    "    for path in (eat_stats, spear_stats, eat_patch_graph, "
    "spear_component_bins):\n"
)
ENTRY_CLEANUP_INSERT = (
    "    for path in (\n"
    "        eat_stats, spear_stats, eat_patch_graph, spear_component_bins,\n"
    "        spectra_stats, xlsr_consensus, spectra_consensus,\n"
    "    ):\n"
)


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


def patch_pipeline(source: str) -> str:
    """Patch exact v47 to export existing vocal XLS-R and Spectra windows."""
    for marker, replacement, label in (
        (PIPELINE_IMPORT_MARKER, PIPELINE_IMPORT_INSERT, "pipeline import"),
        (FAKE_SIGNATURE_MARKER, FAKE_SIGNATURE_INSERT, "fake signature"),
        (FAKE_EMPTY_MARKER, FAKE_EMPTY_INSERT, "empty-stem profile"),
        (FAKE_RETURN_MARKER, FAKE_RETURN_INSERT, "window profile"),
        (SCORER_MARKER, SCORER_INSERT, "Spectra scorer"),
        (VOICE_CALL_MARKER, VOICE_CALL_INSERT, "vocal scoring"),
        (WRITE_MARKER, WRITE_INSERT, "statistic export"),
    ):
        source = replace_once(source, marker, replacement, label)
    return source


def patch_entrypoint(source: str) -> str:
    """Append two frozen Voice fusions to exact v47 and clean their caches."""
    for marker, replacement, label in (
        (ENTRY_IMPORT_MARKER, ENTRY_IMPORT_INSERT, "entrypoint imports"),
        (ENTRY_VARIABLE_MARKER, ENTRY_VARIABLE_INSERT, "cache variables"),
        (QUERY_MARKER, VOICE_FUSION_INSERT, "Voice-only fusion"),
        (ENTRY_CLEANUP_MARKER, ENTRY_CLEANUP_INSERT, "cache cleanup"),
    ):
        source = replace_once(source, marker, replacement, label)
    return source


def inspect_archive(
    archive: Path, expected_sha256: str | None = None,
) -> list[zipfile.ZipInfo]:
    """Reject wrapper, duplicate, traversal, link, and oversized ZIP inputs."""
    archive = Path(archive)
    if archive.stat().st_size > MAX_ARCHIVE_BYTES:
        raise ValueError(f"ZIP exceeds 10 GB: {archive.stat().st_size} bytes")
    if expected_sha256 is not None and sha256_file(archive) != expected_sha256:
        raise ValueError(f"archive hash mismatch: {archive}")
    try:
        handle = zipfile.ZipFile(archive)
    except (OSError, zipfile.BadZipFile) as error:
        raise ValueError(f"invalid ZIP archive: {archive}") from error
    with handle:
        infos = handle.infolist()
        if not infos:
            raise ValueError("empty ZIP archive")
        seen: set[str] = set()
        top_level: set[str] = set()
        total = 0
        for info in infos:
            name = info.filename
            pure = PurePosixPath(name)
            canonical = name.rstrip("/")
            if (
                not name or "\\" in name or name.startswith("/")
                or any(part in {"", ".", ".."} for part in pure.parts)
                or pure.as_posix() != canonical
            ):
                raise ValueError(f"unsafe ZIP member: {name!r}")
            normalized = pure.as_posix().rstrip("/")
            if normalized in seen:
                raise ValueError(f"duplicate ZIP member: {normalized}")
            seen.add(normalized)
            top_level.add(pure.parts[0])
            mode = (info.external_attr >> 16) & 0o170000
            if mode == stat.S_IFLNK:
                raise ValueError(f"symlink ZIP member: {name!r}")
            if info.flag_bits & 0x1:
                raise ValueError(f"encrypted ZIP member: {name!r}")
            if not info.is_dir():
                total += info.file_size
        if top_level != REQUIRED_TOP_LEVEL:
            raise ValueError(
                f"unexpected top-level ZIP structure: {sorted(top_level)}"
            )
        for required in ("script.py", "requirements.txt"):
            if required not in seen:
                raise ValueError(f"missing ZIP member: {required}")
        if not any(name.startswith("model/") for name in seen):
            raise ValueError("model directory is empty")
        if total > MAX_EXPANDED_BYTES:
            raise ValueError(f"expanded ZIP exceeds 32 GB: {total} bytes")
        return infos


def safe_extract(archive: Path, destination: Path) -> None:
    """Extract only after all archive members pass :func:`inspect_archive`."""
    inspect_archive(archive)
    with zipfile.ZipFile(archive) as handle:
        handle.extractall(destination)


def validate_frozen_assets(v24_package: Path) -> None:
    for relative, expected in FROZEN_V24_ASSETS.items():
        source = Path(v24_package) / relative
        if not source.is_file():
            raise FileNotFoundError(f"missing frozen v24 asset: {source}")
        actual = sha256_file(source)
        if actual != expected:
            raise ValueError(
                f"frozen v24 asset hash mismatch: {relative}: {actual}"
            )


def validate_package_tree(package: Path) -> None:
    entries = {path.name for path in Path(package).iterdir()}
    if entries != REQUIRED_TOP_LEVEL:
        raise ValueError(f"unexpected package structure: {sorted(entries)}")
    if not (Path(package) / "model").is_dir():
        raise ValueError("model must be a directory")
    for path in Path(package).rglob("*"):
        if path.is_symlink():
            raise ValueError(f"package contains symlink: {path}")


def make_zip(source: Path, archive: Path) -> None:
    """Create a wrapper-free ZIP64 archive without transient bytecode."""
    validate_package_tree(source)
    with zipfile.ZipFile(
        archive, "w", compression=zipfile.ZIP_DEFLATED,
        compresslevel=1, allowZip64=True,
    ) as handle:
        for path in sorted(Path(source).rglob("*")):
            if path.is_dir() or "__pycache__" in path.parts or path.suffix == ".pyc":
                continue
            handle.write(path, path.relative_to(source).as_posix())


def build_package(
    base_archive: Path,
    v24_package: Path,
    output: Path,
    *,
    archive: bool = False,
) -> Path | None:
    """Build through a private staging directory and publish only on success."""
    base_archive = Path(base_archive).resolve()
    v24_package = Path(v24_package).resolve()
    output = Path(output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    archive_path = output.with_suffix(".zip")
    if archive and archive_path.exists():
        raise FileExistsError(f"Refusing to overwrite {archive_path}")

    inspect_archive(base_archive, BASE_ARCHIVE_SHA256)
    validate_frozen_assets(v24_package)
    staging = Path(tempfile.mkdtemp(prefix=".sparse_voice_v49-", dir=output.parent))
    try:
        safe_extract(base_archive, staging)
        pipeline = staging / "model/src/pipeline.py"
        entrypoint = staging / "script.py"
        pipeline.write_text(
            patch_pipeline(pipeline.read_text(encoding="utf-8")),
            encoding="utf-8",
        )
        entrypoint.write_text(
            patch_entrypoint(entrypoint.read_text(encoding="utf-8")),
            encoding="utf-8",
        )
        for relative in FROZEN_V24_ASSETS:
            source = v24_package / relative
            destination = staging / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
        validate_package_tree(staging)
        os.replace(staging, output)
        staging = None
    finally:
        if staging is not None:
            shutil.rmtree(staging, ignore_errors=True)

    if archive:
        temporary_archive = archive_path.with_name(f".{archive_path.name}.tmp")
        try:
            make_zip(output, temporary_archive)
            inspect_archive(temporary_archive)
            os.replace(temporary_archive, archive_path)
        finally:
            temporary_archive.unlink(missing_ok=True)
        return archive_path
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base-zip", type=Path, default=ROOT / "component_query_or_v47.zip"
    )
    parser.add_argument(
        "--v24-package", type=Path,
        default=ROOT / "sparse_call_consistency_v24",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--archive", action="store_true")
    args = parser.parse_args()
    built_archive = build_package(
        args.base_zip, args.v24_package, args.output, archive=args.archive,
    )
    if built_archive is None:
        print(f"Built {args.output}")
    else:
        print(f"Built {args.output} and {built_archive}")


if __name__ == "__main__":
    main()
