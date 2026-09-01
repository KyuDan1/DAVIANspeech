"""Submit one validated code archive without exposing the DACON token."""

from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path

from dacon_submit_api import dacon_submit_api


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("--competition", required=True)
    parser.add_argument("--team", required=True)
    parser.add_argument("--memo", default="")
    parser.add_argument("--expected-sha256", required=True)
    args = parser.parse_args()

    token = os.environ.get("DACON_API_TOKEN")
    if not token:
        raise RuntimeError("DACON_API_TOKEN is not set")
    if not args.archive.is_file():
        raise FileNotFoundError(args.archive)
    actual = sha256(args.archive)
    if actual != args.expected_sha256:
        raise ValueError(f"archive SHA-256 mismatch: {actual}")

    result = dacon_submit_api.post_code_submission_file(
        str(args.archive), token, args.competition, args.team, args.memo
    )
    print(result, flush=True)
    if not result.get("isSubmitted"):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
