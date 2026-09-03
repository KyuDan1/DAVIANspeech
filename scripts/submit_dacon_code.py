"""Submit one validated code archive without exposing the DACON token."""

from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path
import time

from dacon_submit_api import dacon_submit_api
import requests


VALIDATE_URL = "https://app.dacon.io/api/v1/code-submission/validate"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def quota(token: str, competition: str, team: str) -> dict:
    response = requests.post(VALIDATE_URL, timeout=30, data={
        "cptId": competition, "teamName": team, "apiToken": token,
    })
    response.raise_for_status()
    return response.json()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("--competition", required=True)
    parser.add_argument("--team", required=True)
    parser.add_argument("--memo", default="")
    parser.add_argument("--expected-sha256", required=True)
    parser.add_argument("--wait", action="store_true")
    parser.add_argument("--poll-seconds", type=int, default=30)
    parser.add_argument("--max-wait-hours", type=float, default=2.0)
    args = parser.parse_args()

    token = os.environ.get("DACON_API_TOKEN")
    if not token:
        raise RuntimeError("DACON_API_TOKEN is not set")
    if not args.archive.is_file():
        raise FileNotFoundError(args.archive)
    if len(args.archive.name) > 30:
        raise ValueError("DACON code-submission filename must be at most 30 characters")
    deadline = time.monotonic() + args.max_wait_hours * 3600
    while True:
        status = quota(token, args.competition, args.team)
        available = int(status.get("quota", 0))
        print(f"quota={available}", flush=True)
        if available > 0:
            limit = int(status.get("upload_filesize_limit", 0))
            if limit and args.archive.stat().st_size > limit:
                raise ValueError(f"archive exceeds upload limit {limit}")
            break
        if not args.wait:
            raise RuntimeError("submission quota is exhausted")
        if time.monotonic() >= deadline:
            raise TimeoutError("submission quota did not reopen before deadline")
        time.sleep(args.poll_seconds)
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
