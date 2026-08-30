"""Select conservative file-local presence routing on development data only."""

from __future__ import annotations

import glob
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy.special import expit, logit

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from evaluate_diagnostic import official_eer  # noqa: E402

VOICE_HEAD = np.load(ROOT / "model_heads/xlsr-mixed-domain-balanced-v13.npz")
MUSIC_HEAD = np.load(ROOT / "model_heads/fourier-domain-balanced-v13.npz")


def embeddings(prefix, name):
    result = {}
    paths = glob.glob(str(ROOT / "output" / f"{prefix}_{name}" / "*.npz"))
    if not paths:
        paths = [ROOT / "output" / f"{prefix}_{name}.npz"]
    for path in paths:
        shard = np.load(path)
        result.update({i: v for i, v in zip(shard["ids"].astype(str), shard["embeddings"])})
    return result


def load(name, diagnostic_name, mixed_voice=True):
    truth = pd.read_csv(ROOT / "data/eval" / name / "truth.csv", dtype={"ID": str})
    diagnostic = pd.read_csv(ROOT / "output" / diagnostic_name, dtype={"ID": str}).set_index("ID").loc[truth.ID].reset_index()
    fourier = embeddings("fourier", name)
    matrix = np.stack([fourier[i] for i in truth.ID])
    whole = expit(matrix @ MUSIC_HEAD["weight"] + float(MUSIC_HEAD["bias"]))
    segment = pd.read_csv(ROOT / "output" / f"{name}_fourier_segments_robust.csv").set_index("ID").loc[truth.ID, "SEG_MAX"].to_numpy()
    music = expit(.3 * logit(np.clip(whole, 1e-7, 1 - 1e-7)) + .7 * logit(np.clip(segment, 1e-7, 1 - 1e-7)))
    released = diagnostic.raw_fake_xlsr.to_numpy()
    adapted = released
    if mixed_voice:
        xlsr = embeddings("xlsr_emb", name)
        matrix = np.stack([xlsr[i] for i in truth.ID])
        learned = expit(matrix @ VOICE_HEAD["weight"] + float(VOICE_HEAD["bias"]))
        weight = float(VOICE_HEAD["blend_new"])
        adapted = expit((1 - weight) * logit(np.clip(released, 1e-7, 1 - 1e-7)) + weight * logit(np.clip(learned, 1e-7, 1 - 1e-7)))
    return truth, diagnostic, released, adapted, music


def main():
    mixed = [
        load("external_mixed_v1", "external_mixed_v1_v10_diagnostics.csv"),
        load("source_disjoint_mixed_v1", "source_disjoint_mixed_v1_v10_diagnostics.csv"),
        load("source_disjoint_mixed_equal_v1", "source_disjoint_mixed_equal_v1_v11_diagnostics.csv"),
    ]
    music = load("source_disjoint_music_v1", "source_disjoint_music_v1_v10_diagnostics.csv", False)
    records = []
    for voice_min in (.4, .5, .6, .7):
        for music_max in (.05, .075, .1, .125, .15):
            for music_min in (.5, .6, .7, .8):
                for voice_max in (.1, .15, .2, .25, .3):
                    domains = []
                    for truth, diag, released, adapted, music_score in mixed:
                        voice_only = (diag.voice_present >= voice_min) & (diag.music_present <= music_max)
                        music_only = (diag.music_present >= music_min) & (diag.voice_present <= voice_max) & ~voice_only
                        voice_score = np.where(voice_only, released, adapted)
                        file_score = np.where(voice_only, voice_score, np.where(music_only, music_score, np.maximum(voice_score, music_score)))
                        frame = truth.copy(); frame["voice"] = voice_score; frame["music"] = music_score; frame["file"] = file_score
                        for condition in ("sequential", "simultaneous"):
                            group = frame[frame.CONDITION == condition]
                            fe = official_eer(group.FILE_FAKE, group.file)
                            ve = official_eer(group.VOICE_FAKE, group.voice)
                            me = official_eer(group.MUSIC_FAKE, group.music)
                            domains.append(.5 * (1 - fe) + .2 * (1 - ve) + .3 * (1 - me))
                    truth, diag, released, _, music_score = music
                    voice_only = (diag.voice_present >= voice_min) & (diag.music_present <= music_max)
                    music_only = (diag.music_present >= music_min) & (diag.voice_present <= voice_max) & ~voice_only
                    file_score = np.where(voice_only, released, np.where(music_only, music_score, np.maximum(released, music_score)))
                    selected = truth.SPLIT == "alignment"
                    domains.append(1 - official_eer(truth.loc[selected, "FILE_FAKE"], file_score[selected]))
                    records.append({"VOICE_MIN": voice_min, "MUSIC_MAX": music_max,
                                    "MUSIC_MIN": music_min, "VOICE_MAX": voice_max,
                                    "DEV_MIN": min(domains), "DEV_MEAN": np.mean(domains)})
    table = pd.DataFrame(records).sort_values(["DEV_MIN", "DEV_MEAN"], ascending=False)
    output = ROOT / "output/type-routing-v14-grid.csv"; table.to_csv(output, index=False)
    print(table.head(15).to_string(index=False))


if __name__ == "__main__":
    main()
