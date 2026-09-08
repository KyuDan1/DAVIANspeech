import csv
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import mixfake_multistream_file_inference_v99 as module  # noqa: E402


def test_file_moe_changes_only_file_column(tmp_path, monkeypatch):
    audio = tmp_path / "audio"
    audio.mkdir()
    for name in ("a.wav", "b.wav"):
        (audio / name).write_bytes(b"fixture")
    submission = tmp_path / "submission.csv"
    fields = [
        "ID", "FILE_FAKE_PROB", "VOICE_FAKE_PROB", "MUSIC_FAKE_PROB",
        "VOICE_PRESENT_PROB", "MUSIC_PRESENT_PROB",
    ]
    rows = [
        dict(zip(fields, ["a", ".2", ".3", ".4", ".5", ".6"])),
        dict(zip(fields, ["b", ".8", ".7", ".6", ".5", ".4"])),
    ]
    with submission.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)
    monkeypatch.setattr(
        module, "predict_file",
        lambda paths, *_args, **_kwargs: np.asarray([.6, .4], np.float32),
    )
    module.apply_mixfake_file_moe(
        audio, submission, tmp_path, tmp_path / "head.pt", weight=.4
    )
    with submission.open(newline="") as handle:
        changed = list(csv.DictReader(handle))
    for before, after in zip(rows, changed):
        for field in fields:
            if field not in ("ID", "FILE_FAKE_PROB"):
                assert after[field] == before[field]
    expected = module._sigmoid(
        .6 * module._logit(.2) + .4 * module._logit(np.float32(.6))
    )
    assert float(changed[0]["FILE_FAKE_PROB"]) == float(expected)


def test_file_moe_rejects_invalid_weight(tmp_path):
    try:
        module.apply_mixfake_file_moe(
            tmp_path, tmp_path / "missing.csv", tmp_path, tmp_path / "x", weight=1.1
        )
    except ValueError as error:
        assert "weight" in str(error)
    else:
        raise AssertionError("invalid weight was accepted")


def test_joint_music_moe_preserves_presence(tmp_path, monkeypatch):
    audio = tmp_path / "audio"
    audio.mkdir(); (audio / "x.wav").write_bytes(b"fixture")
    submission = tmp_path / "submission.csv"
    fields = [
        "ID", "FILE_FAKE_PROB", "VOICE_FAKE_PROB", "MUSIC_FAKE_PROB",
        "VOICE_PRESENT_PROB", "MUSIC_PRESENT_PROB",
    ]
    before = dict(zip(fields, ["x", ".2", ".3", ".4", ".55", ".65"]))
    with submission.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerow(before)
    calls = []
    def fake_predict(_paths, _model, _checkpoint, **kwargs):
        calls.append(kwargs["expected_task"])
        if kwargs["expected_task"] == "all":
            return np.asarray([[.8, .1, .7]], np.float32)
        return np.asarray([[.2, .9, .4]], np.float32)
    monkeypatch.setattr(module, "predict_tasks", fake_predict)
    module.apply_mixfake_joint_music_moe(
        audio, submission, tmp_path, tmp_path / "joint.pt", tmp_path / "music.pt"
    )
    with submission.open(newline="") as handle:
        after = next(csv.DictReader(handle))
    assert calls == ["all", "music"]
    assert after["VOICE_PRESENT_PROB"] == before["VOICE_PRESENT_PROB"]
    assert after["MUSIC_PRESENT_PROB"] == before["MUSIC_PRESENT_PROB"]
    assert all(after[name] != before[name] for name in (
        "FILE_FAKE_PROB", "VOICE_FAKE_PROB", "MUSIC_FAKE_PROB",
    ))
    voice = module._sigmoid(.88 * module._logit(.3) + .12 * module._logit(.8))
    music = module._sigmoid(.72 * module._logit(.4) + .28 * module._logit(.9))
    file_direct = module._sigmoid(
        .6 * module._logit(.2) + .4 * module._logit(.7)
    )
    component_or = 1 - (1 - voice) * (1 - music)
    expected_file = module._sigmoid(
        .85 * module._logit(file_direct) + .15 * module._logit(component_or)
    )
    assert np.isclose(float(after["FILE_FAKE_PROB"]), expected_file)
