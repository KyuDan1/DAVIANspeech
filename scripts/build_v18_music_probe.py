#!/usr/bin/env python3
"""Derive a one-column Music-EER probe from the exact submitted v18 ZIP.

The builder refuses every base archive except the byte-exact official v18
archive.  It streams members into a new ZIP, changes only ``script.py``, and
checks that every other member retains its size and CRC.  The injected final
step assigns the same 0.5 Music fake score independently to every CSV row.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import inspect
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import stat
import tempfile
import zipfile


ROOT = Path(__file__).resolve().parents[1]
BASE_ARCHIVE_SHA256 = (
    "b3f7082a268acf08bbc1bfc36a25ced1be6232d81c7cf79be7ab9f890ae2d845"
)
BASE_ARCHIVE_BYTES = 7_907_816_613
BASE_MEMBER_COUNT = 107
BASE_EXPANDED_BYTES = 7_907_795_069
BASE_SCRIPT_SHA256 = (
    "954f30bcaa621aca1055db0c0ccba119dc5ae9b38cf55874f59807e355826de6"
)
BASE_REQUIREMENTS_SHA256 = (
    "978befe6e5c8a064cddd2690e7e55b4b153f87eabf0dcf3b0604a567f4edd6ed"
)
MAX_ARCHIVE_BYTES = 10_000_000_000
MAX_EXPANDED_BYTES = 32_000_000_000
MAX_ARCHIVE_NAME_CHARS = 30
ALLOWED_TOP_LEVEL = {"model", "script.py", "requirements.txt"}

IMPORT_MARKER = "import os\nimport sys\n"
IMPORT_REPLACEMENT = "import csv\nimport os\nimport sys\n"
FUNCTION_MARKER = "\n\ndef main():\n"
FINAL_MARKER = "    for path in (eat_stats, spear_stats):\n"
FINAL_REPLACEMENT = (
    "    # Diagnostic only: run after every model/fusion and change one column.\n"
    "    _apply_music_constant_probe(args.output)\n"
    + FINAL_MARKER
)


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _apply_music_constant_probe(output_path: Path) -> None:
    """Set only MUSIC_FAKE_PROB to 0.5, without using another sample."""
    output_path = Path(output_path)
    temporary = output_path.with_name(f".{output_path.name}.music-probe.tmp")
    temporary.unlink(missing_ok=True)
    try:
        with output_path.open("r", encoding="utf-8", newline="") as source:
            reader = csv.reader(source)
            header = next(reader)
            if header.count("MUSIC_FAKE_PROB") != 1:
                raise ValueError("submission must contain one MUSIC_FAKE_PROB column")
            music_index = header.index("MUSIC_FAKE_PROB")
            with temporary.open("w", encoding="utf-8", newline="") as target:
                writer = csv.writer(target, lineterminator="\n")
                writer.writerow(header)
                rows = 0
                for line_number, row in enumerate(reader, start=2):
                    if len(row) != len(header):
                        raise ValueError(
                            f"submission row {line_number} has {len(row)} columns; "
                            f"expected {len(header)}"
                        )
                    row[music_index] = "0.5"
                    writer.writerow(row)
                    rows += 1
                if rows == 0:
                    raise ValueError("submission contains no prediction rows")
        os.replace(temporary, output_path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def replace_once(text: str, marker: str, replacement: str, label: str) -> str:
    if text.count(marker) != 1:
        raise ValueError(f"official v18 {label} marker changed; refusing injection")
    return text.replace(marker, replacement)


def patch_entrypoint(source: bytes) -> bytes:
    """Inject one final CSV-column override into the exact v18 entrypoint."""
    if sha256_bytes(source) != BASE_SCRIPT_SHA256:
        raise ValueError("script.py is not the exact official v18 entrypoint")
    text = source.decode("utf-8")
    helper = "\n\n" + inspect.getsource(_apply_music_constant_probe).rstrip() + "\n"
    text = replace_once(text, IMPORT_MARKER, IMPORT_REPLACEMENT, "import")
    text = replace_once(text, FUNCTION_MARKER, helper + FUNCTION_MARKER, "main")
    text = replace_once(text, FINAL_MARKER, FINAL_REPLACEMENT, "final cleanup")
    compile(text, "script.py", "exec")
    if text.index("_apply_music_constant_probe(args.output)") < text.index(
        "apply_dual_domain_fusion("
    ):
        raise AssertionError("probe override must run after final model fusion")
    return text.encode("utf-8")


def _is_unsafe_name(name: str) -> bool:
    path = PurePosixPath(name)
    return (
        not name
        or name.startswith("/")
        or "\\" in name
        or any(part in {"", ".", ".."} for part in path.parts)
    )


def inspect_inventory(handle: zipfile.ZipFile) -> dict[str, object]:
    infos = handle.infolist()
    names = [info.filename for info in infos]
    duplicates = sorted({name for name in names if names.count(name) > 1})
    unsafe = sorted(name for name in names if _is_unsafe_name(name))
    symlinks = sorted(
        info.filename
        for info in infos
        if stat.S_ISLNK((info.external_attr >> 16) & 0xFFFF)
    )
    top_level = sorted({PurePosixPath(name).parts[0] for name in names})
    return {
        "member_count": len(infos),
        "expanded_bytes": sum(info.file_size for info in infos),
        "compressed_member_bytes": sum(info.compress_size for info in infos),
        "top_level": top_level,
        "duplicates": duplicates,
        "unsafe_paths": unsafe,
        "symlinks": symlinks,
    }


def verify_base_archive(path: Path) -> dict[str, object]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"missing official v18 archive: {path}")
    if path.stat().st_size != BASE_ARCHIVE_BYTES:
        raise ValueError("official v18 archive byte size mismatch")
    if sha256_file(path) != BASE_ARCHIVE_SHA256:
        raise ValueError("official v18 archive SHA-256 mismatch")
    with zipfile.ZipFile(path) as handle:
        inventory = inspect_inventory(handle)
        if inventory["member_count"] != BASE_MEMBER_COUNT:
            raise ValueError("official v18 member count mismatch")
        if inventory["expanded_bytes"] != BASE_EXPANDED_BYTES:
            raise ValueError("official v18 expanded size mismatch")
        if set(inventory["top_level"]) != ALLOWED_TOP_LEVEL:
            raise ValueError(f"unexpected top-level inventory: {inventory['top_level']}")
        if any(inventory[key] for key in ("duplicates", "unsafe_paths", "symlinks")):
            raise ValueError(f"unsafe official v18 ZIP inventory: {inventory}")
        if sha256_bytes(handle.read("script.py")) != BASE_SCRIPT_SHA256:
            raise ValueError("official v18 script hash mismatch")
        if sha256_bytes(handle.read("requirements.txt")) != BASE_REQUIREMENTS_SHA256:
            raise ValueError("official v18 requirements hash mismatch")
    return inventory


def _clone_info(info: zipfile.ZipInfo) -> zipfile.ZipInfo:
    clone = zipfile.ZipInfo(info.filename, date_time=info.date_time)
    clone.compress_type = info.compress_type
    clone.comment = info.comment
    clone.internal_attr = info.internal_attr
    clone.external_attr = info.external_attr
    clone.create_system = info.create_system
    return clone


def stream_probe_archive(base_zip: Path, output_zip: Path) -> None:
    """Copy the frozen archive while substituting only the patched entrypoint."""
    with zipfile.ZipFile(base_zip, "r") as source, zipfile.ZipFile(
        output_zip, "w", allowZip64=True
    ) as target:
        for info in source.infolist():
            clone = _clone_info(info)
            if info.filename == "script.py":
                target.writestr(clone, patch_entrypoint(source.read(info)))
                continue
            with source.open(info, "r") as input_handle, target.open(
                clone,
                "w",
                force_zip64=info.file_size >= zipfile.ZIP64_LIMIT,
            ) as output_handle:
                shutil.copyfileobj(input_handle, output_handle, 8 * 1024 * 1024)


def validate_probe_archive(
    base_zip: Path, probe_zip: Path, *, verify_crc: bool = True
) -> dict[str, object]:
    size = Path(probe_zip).stat().st_size
    if size >= MAX_ARCHIVE_BYTES:
        raise ValueError(f"probe ZIP is not below 10 GB: {size}")
    with zipfile.ZipFile(base_zip) as base, zipfile.ZipFile(probe_zip) as probe:
        base_infos = {info.filename: info for info in base.infolist()}
        probe_infos = {info.filename: info for info in probe.infolist()}
        inventory = inspect_inventory(probe)
        if set(inventory["top_level"]) != ALLOWED_TOP_LEVEL:
            raise ValueError(f"unexpected probe top level: {inventory['top_level']}")
        if inventory["expanded_bytes"] >= MAX_EXPANDED_BYTES:
            raise ValueError("probe ZIP is not below the 32 GB expanded limit")
        if any(inventory[key] for key in ("duplicates", "unsafe_paths", "symlinks")):
            raise ValueError(f"unsafe probe ZIP inventory: {inventory}")
        if set(base_infos) != set(probe_infos):
            raise ValueError("probe member names differ from official v18")
        for name, base_info in base_infos.items():
            if name == "script.py":
                continue
            probe_info = probe_infos[name]
            if (probe_info.file_size, probe_info.CRC) != (
                base_info.file_size,
                base_info.CRC,
            ):
                raise ValueError(f"non-entrypoint member changed: {name}")
        patched = probe.read("script.py")
        compile(patched, "script.py", "exec")
        if b"_apply_music_constant_probe(args.output)" not in patched:
            raise ValueError("probe call missing from packaged entrypoint")
        if verify_crc:
            bad_member = probe.testzip()
            if bad_member is not None:
                raise ValueError(f"CRC failure in {bad_member}")
    inventory.update(
        {
            "archive_bytes": size,
            "sha256": sha256_file(probe_zip),
            "base_sha256": BASE_ARCHIVE_SHA256,
            "script_sha256": sha256_bytes(patched),
            "crc_verified": bool(verify_crc),
        }
    )
    return inventory


def build_probe_archive(base_zip: Path, output_zip: Path) -> dict[str, object]:
    base_zip = Path(base_zip).resolve()
    output_zip = Path(output_zip).resolve()
    if output_zip.suffix.lower() != ".zip":
        raise ValueError("output must use the .zip suffix")
    if len(output_zip.name) > MAX_ARCHIVE_NAME_CHARS:
        raise ValueError("archive filename must be at most 30 characters")
    if output_zip.exists():
        raise FileExistsError(f"Refusing to overwrite {output_zip}")
    if base_zip == output_zip:
        raise ValueError("base and output archives must differ")
    base_inventory = verify_base_archive(base_zip)

    output_zip.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output_zip.name}.", suffix=".tmp", dir=output_zip.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    temporary.unlink()
    try:
        stream_probe_archive(base_zip, temporary)
        probe_inventory = validate_probe_archive(base_zip, temporary)
        os.replace(temporary, output_zip)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return {"base": base_inventory, "probe": probe_inventory}


def verify_prediction_pair(anchor_csv: Path, probe_csv: Path) -> dict[str, object]:
    """Require identical non-Music field strings and a constant Music field."""
    with Path(anchor_csv).open("r", encoding="utf-8", newline="") as handle:
        anchor_rows = list(csv.DictReader(handle))
    with Path(probe_csv).open("r", encoding="utf-8", newline="") as handle:
        probe_rows = list(csv.DictReader(handle))
    if not anchor_rows or len(anchor_rows) != len(probe_rows):
        raise ValueError("anchor/probe output row counts differ or are empty")
    fields = (
        "ID",
        "FILE_FAKE_PROB",
        "VOICE_FAKE_PROB",
        "VOICE_PRESENT_PROB",
        "MUSIC_PRESENT_PROB",
    )
    for row_number, (anchor, probe) in enumerate(
        zip(anchor_rows, probe_rows), start=2
    ):
        for field in fields:
            if anchor.get(field) != probe.get(field):
                raise ValueError(
                    f"non-Music field {field} differs on CSV row {row_number}"
                )
        if probe.get("MUSIC_FAKE_PROB") != "0.5":
            raise ValueError(f"Music probe is not 0.5 on CSV row {row_number}")
    return {
        "rows": len(anchor_rows),
        "bit_exact_fields": list(fields),
        "music_constant": "0.5",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base-zip", type=Path, default=ROOT / "channel_invariant_moe_v18.zip"
    )
    parser.add_argument(
        "--output", type=Path, default=ROOT / "v18_music_probe_v1.zip"
    )
    args = parser.parse_args()
    report = build_probe_archive(args.base_zip, args.output)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
