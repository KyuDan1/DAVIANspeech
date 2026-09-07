#!/usr/bin/env python3
"""Score several repository banks with the deployed file-local phone router."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
from tqdm import tqdm


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from pipeline import load_audio  # noqa: E402
from telephone_router import TelephoneRouter  # noqa: E402
from train_wpt_spectra_multitask import load_frame  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--router", type=Path,
        default=ROOT / "model_heads/telephone-router-narrowband-v1.npz",
    )
    args = parser.parse_args()
    router = TelephoneRouter(args.router)
    rows = []
    for name in args.datasets:
        frame = load_frame(name, "eval")
        for row in tqdm(
            frame.itertuples(index=False), total=len(frame), desc=f"phone {name}"
        ):
            probability = router.probability(load_audio(Path(row.PATH)))
            rows.append({
                "DATASET": name, "ID": str(row.ID),
                "PHONE_PROB": probability,
                "IS_PHONE": int(probability >= router.threshold),
            })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.output, index=False)
    print(f"Saved {len(rows)} file-local routes to {args.output}")


if __name__ == "__main__":
    main()
