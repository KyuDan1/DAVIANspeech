import csv

import numpy as np

import src.sofia_mert_detector as module


def test_existing_sofia_pass_emits_temporal_statistics_without_second_forward(
    tmp_path, monkeypatch,
):
    audio = tmp_path / "audio"
    audio.mkdir()
    (audio / "sample.flac").touch()
    submission = tmp_path / "submission.csv"
    columns = [
        "ID", "FILE_FAKE_PROB", "VOICE_FAKE_PROB", "MUSIC_FAKE_PROB",
        "VOICE_PRESENT_PROB", "MUSIC_PRESENT_PROB",
    ]
    with submission.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerow({
            "ID": "sample", "FILE_FAKE_PROB": .2,
            "VOICE_FAKE_PROB": .3, "MUSIC_FAKE_PROB": .4,
            "VOICE_PRESENT_PROB": .8, "MUSIC_PRESENT_PROB": .9,
        })

    calls = []

    class FakeDetector:
        def __init__(self, *args, **kwargs):
            pass

        def score_path(self, path):  # pragma: no cover - must not be called
            raise AssertionError("separate score forward was used")

        def score_and_statistics_path(self, path):
            calls.append(path)
            return .7, np.zeros((3, 13, 2, 768), dtype=np.float16)

    monkeypatch.setattr(module, "SofiaMertDetector", FakeDetector)
    statistics = tmp_path / "mert_stats.npz"
    module.apply_sofia_mert_fusion(
        audio, submission, tmp_path, tmp_path / "head.pt",
        device="cpu", statistics_output_path=statistics,
    )
    assert calls == [audio / "sample.flac"]
    with np.load(statistics, allow_pickle=False) as archive:
        assert archive["ids"].tolist() == ["sample"]
        assert archive["statistics"].shape == (1, 3, 13, 2, 768)
