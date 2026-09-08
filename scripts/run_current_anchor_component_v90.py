#!/usr/bin/env python3
"""Run and verify one frozen current-anchor v70 shard."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--shard", type=int, choices=range(4), required=True)
    args = parser.parse_args()
    frozen_path = args.run / "frozen.json"
    frozen = json.loads(frozen_path.read_text())
    record = frozen["runners"][args.shard]
    runner = Path(record["root"])
    if (sha(frozen["truth"]) != frozen["truth_sha256"]
            or sha(frozen["script"]) != frozen["script_sha256"]
            or sha(runner / "data/sample_submission.csv") != record["sample_sha256"]):
        raise ValueError("frozen input changed")
    output = runner / "output/submission.csv"
    if output.exists():
        raise FileExistsError(output)
    spec = importlib.util.spec_from_file_location(f"anchor_v90_{args.shard}", frozen["script"])
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.BASE_DIR = runner
    # The current-best File value is final immediately after the WPT expert.
    # Disable only the cache/export and post-WPT branches that update Music or
    # Voice with File weight fixed to zero.  This keeps the File graph exact
    # while avoiding a redundant original-audio XLS-R pass in this diagnostic.
    original_run = module.run

    def run_without_music_cache(runtime_args):
        runtime_args.xlsr_window_embeddings_output = None
        return original_run(runtime_args)

    module.run = run_without_music_cache
    module.apply_spectra_voice_fusion = lambda *unused_args, **unused_kwargs: None
    module.apply_sparse_call_consensus = lambda *unused_args, **unused_kwargs: None
    module.apply_three_stream_anchor_residual = lambda *unused_args, **unused_kwargs: None
    # Each runner owns a physically ID-matched 1/4 view because the legacy
    # post-processors do not implement pipeline.py's internal shard flags.
    sys.argv = [frozen["script"]]
    started = time.monotonic()
    module.main()
    elapsed = time.monotonic() - started
    if sha(frozen["truth"]) != frozen["truth_sha256"] or sha(frozen["script"]) != frozen["script_sha256"]:
        raise ValueError("frozen input changed during inference")
    import pandas as pd
    prediction = pd.read_csv(output, dtype={"ID": str})
    expected = (frozen["rows"] + 3 - args.shard) // 4
    if len(prediction) != expected or prediction.ID.duplicated().any():
        raise ValueError("invalid shard output")
    report = dict(status="complete", shard=args.shard, rows=len(prediction), seconds=elapsed,
                  predictions_sha256=sha(output), frozen_sha256=sha(frozen_path))
    (runner / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
