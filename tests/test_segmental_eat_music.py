import numpy as np
import pandas as pd
import torch

from scripts.extract_eat_hierarchical_stats import view_starts
from src.segmental_eat_music import SegmentalEatMusicHead
from src.segmental_eat_music_inference import apply_segmental_eat_music_fusion


def test_segment_view_starts_avoid_redundant_dense_crops():
    crop = 6 * 16_000
    assert view_starts(4 * 16_000, crop, 8, "segments") == [0]
    assert len(view_starts(10 * 16_000, crop, 8, "segments")) == 3
    starts = view_starts(60 * 16_000, crop, 8, "segments")
    assert len(starts) == 8
    assert starts[0] == 0 and starts[-1] == 54 * 16_000
    assert np.all(np.diff(starts) > 0)


def test_segmental_head_masks_padding_and_supports_both_paths():
    mean = torch.zeros(3, 5, 8)
    std = torch.ones_like(mean)
    values = torch.randn(2, 4, 3, 5, 8)
    mask = torch.tensor([[True, True, False, False], [True, True, True, True]])
    for mode in ("content", "structure", "both"):
        model = SegmentalEatMusicHead(
            mean, std, max_views=4, heads=2,
            dropout=0, branch_mode=mode,
        ).eval()
        first = model(values, mask)
        changed = values.clone()
        changed[0, 2:] = 10_000
        second = model(changed, mask)
        assert first.shape == (2,)
        assert torch.isfinite(first).all()
        assert torch.allclose(first[0], second[0], atol=1e-6)


def test_fixed_inner_moe_preserves_other_axes_and_outer_phone_route(
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
    endpoint = tmp_path / "endpoint.npz"
    segments = tmp_path / "segments.npz"
    for path in (endpoint, segments):
        np.savez(
            path, ids=np.asarray(["clean", "phone"]),
            statistics=np.zeros((2, 1), np.float32),
            view_mask=np.ones((2, 1), bool),
        )
    phone = tmp_path / "phone.npz"
    np.savez(phone, ids=np.asarray(["phone"]))
    monkeypatch.setattr(
        "src.segmental_eat_music_inference._shared_projection",
        lambda *args, **kwargs: np.eye(2, dtype=np.float32),
    )
    monkeypatch.setattr(
        "src.segmental_eat_music_inference.predict_hierarchical_music",
        lambda *args, **kwargs: np.asarray([.6, .6]),
    )
    monkeypatch.setattr(
        "src.segmental_eat_music_inference.predict_segmental_music",
        lambda *args, **kwargs: np.asarray([.9, .9]),
    )
    apply_segmental_eat_music_fusion(
        submission, endpoint, [tmp_path / "h.pt"],
        segments, [tmp_path / "s.pt"], device="cpu",
        telephone_ids_path=phone, expert_weight=.25,
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
