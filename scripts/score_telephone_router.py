#!/usr/bin/env python3
"""Score a registered audio bank with the portable telephone router."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pipeline import find_audio_files, load_audio  # noqa: E402
from telephone_router import TelephoneRouter  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bank", type=Path, required=True)
    parser.add_argument(
        "--router", type=Path,
        default=ROOT / "model_heads/telephone-router-narrowband-v1.npz",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    truth = pd.read_csv(args.bank / "truth.csv", dtype={"ID": str})
    files = {path.stem: path for path in find_audio_files(args.bank / "audio")}
    if set(truth.ID) != set(files):
        raise ValueError("Truth and audio IDs differ")
    router = TelephoneRouter(args.router)
    rows = []
    for sample_id in tqdm(truth.ID, desc="telephone router"):
        probability = router.probability(load_audio(files[sample_id]))
        rows.append({
            "ID": sample_id,
            "PHONE_PROB": probability,
            "IS_PHONE": int(probability >= router.threshold),
        })
    result = pd.DataFrame(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)
    print(f"Routed {result.IS_PHONE.sum()}/{len(result)} files", flush=True)


if __name__ == "__main__":
    main()
