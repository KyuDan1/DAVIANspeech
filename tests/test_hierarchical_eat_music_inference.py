import numpy as np
import pandas as pd

from src.hierarchical_eat_music_inference import (
    apply_hierarchical_eat_music_fusion,
)


def test_domain_conditional_fusion_changes_only_music_and_gated_file(
    tmp_path, monkeypatch,
):
    submission = tmp_path / "submission.csv"
    pd.DataFrame({
        "ID": ["clean", "phone"],
        "FILE_FAKE_PROB": [.2, .2],
        "VOICE_FAKE_PROB": [.4, .4],
        "MUSIC_FAKE_PROB": [.2, .2],
        "VOICE_PRESENT_PROB": [.8, .8],
        "MUSIC_PRESENT_PROB": [.4, .8],
    }).to_csv(submission, index=False)
    statistics = tmp_path / "statistics.npz"
    np.savez(
        statistics,
        ids=np.asarray(["clean", "phone"]),
        statistics=np.zeros((2, 1), np.float32),
        view_mask=np.ones((2, 1), bool),
    )
    phone = tmp_path / "phone.npz"
    np.savez(phone, ids=np.asarray(["phone"]))
    monkeypatch.setattr(
        "src.hierarchical_eat_music_inference.predict_hierarchical_music",
        lambda *args, **kwargs: np.asarray([.8, .8]),
    )
    apply_hierarchical_eat_music_fusion(
        submission, statistics, [tmp_path / "unused.pt"],
        device="cpu", telephone_ids_path=phone,
        music_weight=.1, file_weight=.1,
        phone_music_weight=.5, phone_file_weight=.5,
        file_music_presence_threshold=.5,
    )
    result = pd.read_csv(submission)
    assert result.VOICE_FAKE_PROB.tolist() == [.4, .4]
    assert result.VOICE_PRESENT_PROB.tolist() == [.8, .8]
    assert result.MUSIC_PRESENT_PROB.tolist() == [.4, .8]
    assert result.MUSIC_FAKE_PROB.iloc[1] > result.MUSIC_FAKE_PROB.iloc[0] > .2
    assert result.FILE_FAKE_PROB.iloc[0] == .2
    assert result.FILE_FAKE_PROB.iloc[1] > .2
