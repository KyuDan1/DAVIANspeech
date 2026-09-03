#!/usr/bin/env python3
"""Resume a large HTTP download using independent byte ranges."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import os
from pathlib import Path
import re

import requests


def remote_size(url: str) -> int:
    response = requests.head(url, allow_redirects=True, timeout=60)
    response.raise_for_status()
    value = response.headers.get("content-length")
    if value is None:
        raise RuntimeError("server did not provide content-length")
    return int(value)


def download_part(url: str, path: Path, start: int, end: int) -> tuple[int, int]:
    expected = end - start + 1
    present = path.stat().st_size if path.exists() else 0
    if present > expected:
        raise RuntimeError(f"oversized partial file: {path}")
    if present == expected:
        return path.name, present
    range_start = start + present
    with requests.get(
        url,
        headers={"Range": f"bytes={range_start}-{end}"},
        stream=True,
        timeout=(60, 180),
    ) as response:
        response.raise_for_status()
        content_range = response.headers.get("content-range", "")
        match = re.fullmatch(r"bytes (\d+)-(\d+)/(\d+)", content_range)
        if not match or (int(match[1]), int(match[2])) != (range_start, end):
            raise RuntimeError(f"unexpected Content-Range: {content_range!r}")
        with path.open("ab") as handle:
            for chunk in response.iter_content(8 * 1024 * 1024):
                if chunk:
                    handle.write(chunk)
    size = path.stat().st_size
    if size != expected:
        raise RuntimeError(f"short partial file {path}: {size} != {expected}")
    return path.name, size


def checksum(path: Path, algorithm: str) -> str:
    digest = hashlib.new(algorithm)
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("url")
    parser.add_argument("output", type=Path)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--md5")
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")

    size = remote_size(args.url)
    part_dir = args.output.with_name(args.output.name + ".parts")
    part_dir.mkdir(parents=True, exist_ok=True)
    chunk_size = (size + args.workers - 1) // args.workers
    ranges = [
        (index, index * chunk_size, min(size, (index + 1) * chunk_size) - 1)
        for index in range(args.workers)
        if index * chunk_size < size
    ]

    # A prior single-stream download is a valid prefix of the first range.
    first = part_dir / "part_000"
    if args.output.exists() and not first.exists():
        prefix_size = args.output.stat().st_size
        if prefix_size <= ranges[0][2] - ranges[0][1] + 1:
            os.replace(args.output, first)
        elif prefix_size != size:
            raise RuntimeError(f"cannot reuse existing file of size {prefix_size}")
        else:
            digest = checksum(args.output, "md5")
            if args.md5 and digest != args.md5:
                raise RuntimeError(f"MD5 mismatch: {digest} != {args.md5}")
            print(f"Already complete: {args.output} md5={digest}", flush=True)
            return

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [
            executor.submit(
                download_part, args.url, part_dir / f"part_{index:03d}", start, end
            )
            for index, start, end in ranges
        ]
        for future in as_completed(futures):
            name, downloaded = future.result()
            print(f"Completed {name}: {downloaded} bytes", flush=True)

    temporary = args.output.with_suffix(args.output.suffix + ".assembling")
    with temporary.open("wb") as destination:
        for index, _, _ in ranges:
            with (part_dir / f"part_{index:03d}").open("rb") as source:
                for chunk in iter(lambda: source.read(16 * 1024 * 1024), b""):
                    destination.write(chunk)
    if temporary.stat().st_size != size:
        raise RuntimeError("assembled size mismatch")
    digest = checksum(temporary, "md5")
    if args.md5 and digest != args.md5:
        raise RuntimeError(f"MD5 mismatch: {digest} != {args.md5}")
    os.replace(temporary, args.output)
    for index, _, _ in ranges:
        (part_dir / f"part_{index:03d}").unlink()
    part_dir.rmdir()
    print(f"Downloaded {args.output} ({size} bytes), md5={digest}", flush=True)


if __name__ == "__main__":
    main()
