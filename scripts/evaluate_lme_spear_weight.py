"""Evaluate conservative SPEAR weights on the exact LME factorial predictions."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from evaluate_diagnostic import score_frame  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    truth = pd.read_csv(
        ROOT / "data/eval/factorial_eval_1200_v2/truth.csv", dtype={"ID": str}
    )
    anchor = pd.concat(
        pd.read_csv(path, dtype={"ID": str})
        for path in sorted(
            (ROOT / "reports/factorial_v2_lme_exact").glob("anchor_shard_*.csv")
        )
    )
    spear = pd.read_csv(
        ROOT / "reports/factorial_v2_current/spear_probe_scores.csv",
        dtype={"ID": str},
    )
    frame = truth.merge(anchor, on="ID", validate="one_to_one").merge(
        spear, on="ID", validate="one_to_one"
    )

    records = []
    groups = [("ALL", "ALL", frame)]
    for column in ("CHANNEL", "MIX_MODE", "COMPONENT_CASE"):
        groups.extend((column, str(value), group) for value, group in frame.groupby(column))
    for group_name, value, group in groups:
        for weight in (0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30):
            candidate = group.copy()
            candidate["FILE_FAKE_PROB"] = (
                (1 - weight) * group.FILE_FAKE_PROB
                + weight * group.SPEAR_JOINT_FILE_PROB
            )
            candidate["MUSIC_FAKE_PROB"] = (
                (1 - weight) * group.MUSIC_FAKE_PROB
                + weight * group.SPEAR_BINARY_MUSIC_PROB
            )
            metrics = score_frame(candidate)
            records.append(
                {
                    "GROUP": group_name,
                    "VALUE": value,
                    "WEIGHT": weight,
                    **metrics,
                }
            )
    result = pd.DataFrame(records)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)
    print(
        result.query("GROUP == 'ALL'")[
            ["WEIGHT", "FILE_EER", "VOICE_EER", "MUSIC_EER", "ADS"]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
