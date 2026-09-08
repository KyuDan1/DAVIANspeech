#!/usr/bin/env python3
"""Prepare four immutable runners for the current-best File anchor on v70."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BANK = ROOT / "data/eval/long_component_stress_v70"
PACKAGE = ROOT / "reports/music_only_submission_v82/full_package_smoke"


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)

    import pandas as pd
    truth_path = BANK / "truth.csv"
    truth = pd.read_csv(truth_path, dtype={"ID": str})
    validation_path = BANK / "validation.json"
    validation = json.loads(validation_path.read_text())
    if (not validation["passed"] or validation["selection_allowed"]
            or validation["truth_sha256"] != sha(truth_path)):
        raise ValueError("invalid stress bank")
    script = PACKAGE / "script_anchor.py"
    smoke = json.loads((PACKAGE / "report.json").read_text())

    args.output.mkdir(parents=True)
    runners = []
    columns = ["ID", "FILE_FAKE_PROB", "VOICE_FAKE_PROB", "MUSIC_FAKE_PROB",
               "VOICE_PRESENT_PROB", "MUSIC_PRESENT_PROB"]
    for shard in range(4):
        runner = args.output / f"runner_{shard}"
        (runner / "data/test").mkdir(parents=True)
        (runner / "output").mkdir()
        os.symlink(PACKAGE.joinpath("model").resolve(), runner / "model")
        selected = truth.iloc[shard::4]
        for identity in selected.ID:
            source = BANK / "audio" / f"{identity}.flac"
            os.symlink(source.resolve(), runner / "data/test" / source.name)
        with (runner / "data/sample_submission.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns)
            writer.writeheader()
            for identity in selected.ID:
                writer.writerow(dict(ID=identity, **{name: 0.5 for name in columns[1:]}))
        runners.append(dict(shard=shard, root=str(runner.resolve()),
                            sample_sha256=sha(runner / "data/sample_submission.csv")))
    frozen = dict(
        schema="current_anchor_component_v90",
        rows=len(truth), shards=4, runners=runners,
        truth=str(truth_path.resolve()), truth_sha256=sha(truth_path),
        validation_sha256=sha(validation_path),
        script=str(script.resolve()), script_sha256=sha(script),
        package_smoke=smoke,
        anchor_identity="v50+v57 script_anchor packaged alongside v82/v83 non-Music-exact smoke",
        scope="PREVIOUSLY EXPOSED synthetic v70; diagnostic only",
        selection_allowed=False, automatic_submission_allowed=False,
    )
    (args.output / "frozen.json").write_text(json.dumps(frozen, indent=2) + "\n")
    print(json.dumps(dict(status="prepared", rows=len(truth), runners=4)), flush=True)


if __name__ == "__main__":
    main()
