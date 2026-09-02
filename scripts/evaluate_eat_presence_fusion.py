"""Evaluate a train-free PANNs + EAT presence ensemble and gate propagation."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from safetensors import safe_open
from scipy.special import expit
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from evaluate_diagnostic import score_frame  # noqa: E402

VOICE_WEIGHT = 0.30
MUSIC_WEIGHT = 0.90


def read_shards(directory: Path, pattern: str) -> pd.DataFrame:
    return pd.concat(
        pd.read_csv(path, dtype={"ID": str})
        for path in sorted(directory.glob(pattern))
    )


def eat_presence(directory: Path) -> pd.DataFrame:
    shards = [np.load(path) for path in sorted(directory.glob("*.npz"))]
    ids = np.concatenate([shard["ids"].astype(str) for shard in shards])
    embeddings = np.concatenate([shard["embeddings"] for shard in shards])

    labels = pd.read_csv(
        ROOT / "models/panns/class_labels_indices.csv"
    ).display_name.tolist()
    index = {label: offset for offset, label in enumerate(labels)}
    groups = json.loads(
        (ROOT / "models/panns/component_labels.json").read_text("utf-8")
    )
    voice_indices = [index[label] for label in groups["voice"]]
    music_indices = [index[label] for label in groups["music"]]
    with safe_open(
        ROOT / "models/eat-base-as2m/model.safetensors",
        framework="np",
    ) as checkpoint:
        norm_weight = checkpoint.get_tensor("model.fc_norm.weight")
        norm_bias = checkpoint.get_tensor("model.fc_norm.bias")
        head_weight = checkpoint.get_tensor("model.head.weight")
        head_bias = checkpoint.get_tensor("model.head.bias")
    normalized = (embeddings - embeddings.mean(1, keepdims=True)) / np.sqrt(
        embeddings.var(1, keepdims=True) + 1e-5
    )
    normalized = normalized * norm_weight + norm_bias
    probabilities = expit(normalized @ head_weight.T + head_bias)
    return pd.DataFrame(
        {
            "ID": ids,
            "EAT_VOICE_PRESENT_PROB": probabilities[:, voice_indices].max(1),
            "EAT_MUSIC_PRESENT_PROB": probabilities[:, music_indices].max(1),
        }
    )


def component_or(voice, music, voice_present, music_present, gate):
    active_voice = voice_present >= gate
    active_music = music_present >= gate
    result = np.maximum(
        np.where(active_voice, voice, -np.inf),
        np.where(active_music, music, -np.inf),
    )
    neither = ~np.isfinite(result)
    result[neither] = np.where(
        voice_present[neither] >= music_present[neither],
        voice[neither], music[neither],
    )
    return result


def main() -> None:
    output_dir = ROOT / "reports/eat_presence_v1"
    output_dir.mkdir(parents=True, exist_ok=True)
    banks = [
        (
            "factorial", ROOT / "data/eval/factorial_eval_1200_v2/truth.csv",
            read_shards(ROOT / "reports/factorial_v2_lme_exact", "anchor_shard_*.csv"),
            ROOT / "reports/eat_presence_v1/factorial_multiview",
            ROOT / "reports/factorial_v2_current/spear_probe_scores.csv",
        ),
        (
            "yue", ROOT / "data/eval/yue_cross_component_audit_v1/truth.csv",
            pd.read_csv(ROOT / "reports/yue_audit_anchor/submission.csv", dtype={"ID": str}),
            ROOT / "reports/eat_presence_v1/yue_multiview",
            ROOT / "reports/yue_audit_spear_scores.csv",
        ),
        (
            "competition_v2", ROOT / "data/eval/competition_v2/truth.csv",
            pd.read_csv(ROOT / "output/competition_v2_candidate.csv", dtype={"ID": str}),
            ROOT / "reports/eat_presence_v1/comp2_multiview",
            ROOT / "output/spear_competition_v2_probe_scores.csv",
        ),
        (
            "competition_v3", ROOT / "data/eval/competition_v3/truth.csv",
            pd.read_csv(ROOT / "output/competition_v3_candidate.csv", dtype={"ID": str}),
            ROOT / "reports/eat_presence_v1/comp3_multiview",
            ROOT / "output/spear_competition_v3_probe_scores.csv",
        ),
        (
            "phone_factorial", ROOT / "data/eval/phone_factorial_1200_v1/truth.csv",
            read_shards(ROOT / "reports/phone_factorial_1200_v1", "anchor_shard_*.csv"),
            ROOT / "reports/eat_presence_v1/phone_multiview", None,
        ),
    ]
    records = []
    for name, truth_path, anchor, eat_dir, spear_path in banks:
        frame = pd.read_csv(truth_path, dtype={"ID": str}).merge(
            anchor, on="ID", validate="one_to_one"
        ).merge(read_shards(eat_dir, "*.csv"), on="ID", validate="one_to_one")
        old_voice = frame.VOICE_PRESENT_PROB.to_numpy()
        old_music = frame.MUSIC_PRESENT_PROB.to_numpy()
        new_voice = (
            (1 - VOICE_WEIGHT) * old_voice
            + VOICE_WEIGHT * frame.EAT_VOICE_PRESENT_PROB.to_numpy()
        )
        new_music = (
            (1 - MUSIC_WEIGHT) * old_music
            + MUSIC_WEIGHT * frame.EAT_MUSIC_PRESENT_PROB.to_numpy()
        )
        old_voice_auc = roc_auc_score(frame.VOICE_PRESENT, old_voice)
        new_voice_auc = roc_auc_score(frame.VOICE_PRESENT, new_voice)
        old_music_auc = (
            roc_auc_score(frame.MUSIC_PRESENT, old_music)
            if frame.MUSIC_PRESENT.nunique() == 2 else np.nan
        )
        new_music_auc = (
            roc_auc_score(frame.MUSIC_PRESENT, new_music)
            if frame.MUSIC_PRESENT.nunique() == 2 else np.nan
        )
        records.append(
            {
                "DATASET": name, "VARIANT": "presence_only",
                "OLD_VOICE_AUC": old_voice_auc, "NEW_VOICE_AUC": new_voice_auc,
                "OLD_MUSIC_AUC": old_music_auc, "NEW_MUSIC_AUC": new_music_auc,
                "OLD_CPS": np.nanmean([old_voice_auc, old_music_auc]),
                "NEW_CPS": np.nanmean([new_voice_auc, new_music_auc]),
            }
        )
        if spear_path is None:
            continue
        spear = pd.read_csv(spear_path, dtype={"ID": str})
        frame = frame.merge(spear, on="ID", validate="one_to_one")
        fused_music = (
            0.9 * frame.MUSIC_FAKE_PROB + 0.1 * frame.SPEAR_BINARY_MUSIC_PROB
        )
        old_file = 0.9 * frame.FILE_FAKE_PROB + 0.1 * frame.SPEAR_JOINT_FILE_PROB
        for gate in (0.4, 0.5, 0.6, 0.7, 0.8):
            candidate = frame.copy()
            new_anchor_file = component_or(
                frame.VOICE_FAKE_PROB.to_numpy(),
                frame.MUSIC_FAKE_PROB.to_numpy(), new_voice, new_music, gate,
            )
            candidate["FILE_FAKE_PROB"] = (
                0.9 * new_anchor_file + 0.1 * frame.SPEAR_JOINT_FILE_PROB
            )
            candidate["MUSIC_FAKE_PROB"] = fused_music
            candidate["VOICE_PRESENT_PROB"] = new_voice
            candidate["MUSIC_PRESENT_PROB"] = new_music
            metrics = score_frame(candidate)
            records.append(
                {"DATASET": name, "VARIANT": "propagated_gate", "GATE": gate, **metrics}
            )
        candidate = frame.copy()
        candidate["FILE_FAKE_PROB"] = old_file
        candidate["MUSIC_FAKE_PROB"] = fused_music
        candidate["VOICE_PRESENT_PROB"] = new_voice
        candidate["MUSIC_PRESENT_PROB"] = new_music
        records.append(
            {"DATASET": name, "VARIANT": "decoupled_gate", **score_frame(candidate)}
        )

    result = pd.DataFrame(records)
    result.to_csv(output_dir / "metrics.csv", index=False)
    print(result.round(6).to_string(index=False))


if __name__ == "__main__":
    main()
