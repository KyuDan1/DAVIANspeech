"""Submit code-submission zips to DACON once the daily quota reopens.

The package is 4.4 GB, so this uploads from the machine that built it rather
than round-tripping through a laptop. It submits at most one zip per quota unit,
re-checking the quota between uploads, and stops on the first failure instead of
burning the remaining slots on a package that has already proven broken.

    python scripts/submit_dacon.py --config .config --wait \
        --probe path/to/run.zip:"memo shown on the submission page"

The DACON CLI takes the token as an argv flag and echoes it to stdout; this
reads it from a file instead so it stays out of the process list and the logs.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

import requests

VALIDATE_URL = "https://app.dacon.io/api/v1/code-submission/validate"


def read_config(path: Path) -> dict:
    config = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        match = re.match(r"\s*([A-Za-z_][A-Za-z0-9_]*)\s*[:=]\s*(.*)", line)
        if match:
            config[match.group(1)] = match.group(2).strip().strip('"').strip("'")
    required = ["DACON_SUBMISSION_API_KEY", "DACON_COMPETITION_ID",
                "DACON_COMPETION_TEAM_NAME"]
    missing = [key for key in required if key not in config]
    if missing:
        raise SystemExit(f"config missing {missing}")
    return config


def check_quota(config: dict) -> dict:
    response = requests.post(VALIDATE_URL, timeout=30, data={
        "cptId": config["DACON_COMPETITION_ID"],
        "teamName": config["DACON_COMPETION_TEAM_NAME"],
        "apiToken": config["DACON_SUBMISSION_API_KEY"],
    })
    if response.status_code != 200:
        return {"quota": 0, "error": f"HTTP {response.status_code} {response.text[:200]}"}
    return response.json()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--probe", action="append", required=True, metavar="ZIP:MEMO",
                        help="Repeatable. Submitted in the order given.")
    parser.add_argument("--wait", action="store_true",
                        help="Poll until quota opens instead of exiting.")
    parser.add_argument("--poll-seconds", type=int, default=120)
    parser.add_argument("--max-wait-hours", type=float, default=14.0)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    config = read_config(args.config)
    jobs = []
    for entry in args.probe:
        zip_path, _, memo = entry.partition(":")
        path = Path(zip_path)
        if not path.is_file():
            raise SystemExit(f"missing {path}")
        jobs.append((path, memo or path.stem))

    from dacon_submit_api import dacon_submit_api

    deadline = time.time() + args.max_wait_hours * 3600
    results = []
    for index, (path, memo) in enumerate(jobs, 1):
        while True:
            status = check_quota(config)
            quota = status.get("quota", 0)
            print(f"[{time.strftime('%H:%M:%S')}] quota={quota} "
                  f"({index}/{len(jobs)}: {path.name})", flush=True)
            if quota and quota > 0:
                break
            if not args.wait:
                raise SystemExit("quota exhausted; rerun with --wait")
            if time.time() > deadline:
                raise SystemExit("gave up waiting for quota")
            time.sleep(args.poll_seconds)

        limit = status.get("upload_filesize_limit")
        size = path.stat().st_size
        if limit and size > limit:
            raise SystemExit(f"{path.name} is {size} B, over the {limit} B limit")

        if args.dry_run:
            print(f"  DRY RUN: would submit {path.name} ({size/2**30:.2f} GiB) memo={memo!r}")
            results.append({"file": path.name, "dry_run": True})
            continue

        print(f"  submitting {path.name} ({size/2**30:.2f} GiB) memo={memo!r}", flush=True)
        result = dacon_submit_api.post_code_submission_file(
            str(path), config["DACON_SUBMISSION_API_KEY"],
            config["DACON_COMPETITION_ID"], config["DACON_COMPETION_TEAM_NAME"], memo,
        )
        print(f"  -> {result}", flush=True)
        results.append({"file": path.name, "memo": memo, "result": result})
        if not (isinstance(result, dict) and result.get("isSubmitted")):
            # Do not spend the next slot until a human has looked at this.
            print("  FAILED; stopping before the remaining probes.", flush=True)
            break

    print(json.dumps(results, ensure_ascii=False, indent=1))
    return 0 if all(r.get("dry_run") or r["result"].get("isSubmitted") for r in results) else 1


if __name__ == "__main__":
    sys.exit(main())
