"""Evaluate anchor/SPEAR weights on the paired telephone factorial set."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit

import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from evaluate_diagnostic import official_eer  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    truth = pd.read_csv(
        ROOT / "data/eval/phone_factorial_1200_v1/truth.csv", dtype={"ID": str}
    )
    anchor = pd.concat([
        pd.read_csv(path, dtype={"ID": str}) for path in sorted(
            (ROOT / "reports/phone_factorial_1200_v1").glob("anchor_shard_*.csv")
        )
    ])
    shards = [np.load(path) for path in sorted(
        (ROOT / "output/spear_phone_factorial_1200_v1").glob("*.npz")
    )]
    ids = np.concatenate([item["ids"].astype(str) for item in shards])
    vectors = np.concatenate([item["embeddings"] for item in shards])
    music_head = np.load(ROOT / "model_heads/spear-mixed-music_fake-head.npz")
    joint = np.load(ROOT / "model_heads/spear-cross-component-joint-v1.npz")
    music = expit(vectors @ music_head["weight"] + music_head["bias"])
    hidden = vectors.reshape(-1, 13, 1280)
    normalized = (hidden - joint["mean"]) / joint["std"]
    logits = normalized[:, 2] @ joint["joint_weight"][2] + joint["joint_bias"][2]
    logits -= logits.max(axis=1, keepdims=True)
    posterior = np.exp(logits)
    posterior /= posterior.sum(axis=1, keepdims=True)
    expert = pd.DataFrame({
        "ID": ids, "SPEAR_FILE": 1 - posterior[:, 0], "SPEAR_MUSIC": music,
    })
    frame = truth.merge(anchor, on="ID", validate="one_to_one").merge(
        expert, on="ID", validate="one_to_one"
    )

    records = []
    groups = [("ALL", "ALL", frame)]
    groups += [("CHANNEL", name, group) for name, group in frame.groupby("CHANNEL")]
    groups += [("AUDIO_TYPE", name, group)
               for name, group in frame.groupby("AUDIO_TYPE")]
    for group_name, value, group in groups:
        for weight in (0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30):
            file_score = ((1 - weight) * group.FILE_FAKE_PROB
                          + weight * group.SPEAR_FILE)
            music_score = ((1 - weight) * group.MUSIC_FAKE_PROB
                           + weight * group.SPEAR_MUSIC)
            file_eer = official_eer(group.FILE_FAKE, file_score)
            voice_part = group.dropna(subset=["VOICE_FAKE"])
            music_part = group.dropna(subset=["MUSIC_FAKE"])
            voice_eer = (official_eer(voice_part.VOICE_FAKE,
                                      voice_part.VOICE_FAKE_PROB)
                         if voice_part.VOICE_FAKE.nunique() == 2 else np.nan)
            music_eer = (official_eer(music_part.MUSIC_FAKE,
                                      music_score.loc[music_part.index])
                         if music_part.MUSIC_FAKE.nunique() == 2 else np.nan)
            ads = (.5 * (1 - file_eer)
                   + (.2 * (1 - voice_eer) if np.isfinite(voice_eer) else 0)
                   + (.3 * (1 - music_eer) if np.isfinite(music_eer) else 0))
            records.append({
                "GROUP": group_name, "VALUE": value, "WEIGHT": weight,
                "N": len(group), "FILE_EER": file_eer,
                "VOICE_EER": voice_eer, "MUSIC_EER": music_eer, "ADS": ads,
            })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records).to_csv(args.output, index=False)
    print(pd.DataFrame(records).query(
        "(GROUP == 'ALL') or (WEIGHT in [0.0, 0.1, 0.2])"
    ).to_string(index=False))


if __name__ == "__main__":
    main()
