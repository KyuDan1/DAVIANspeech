import csv

import numpy as np
import pandas as pd
import pytest
import torch

from scripts.train_spear_voice_mil import channel_pairs, sample_weights
from src.spear_voice_mil_head import SpearVoiceMILHead, spear_voice_mil_loss
from src.spear_voice_mil_inference import apply_spear_voice_mil_residual


def make_model(dimension=4):
    return SpearVoiceMILHead(
        dimension, torch.zeros(dimension), torch.ones(dimension),
        hidden=8, dropout=0.0, temperature=2.0,
    )


def test_presence_router_downweights_a_non_speech_fake_bin():
    model = make_model()
    mask = torch.ones(1, 2, dtype=torch.bool)
    fake_logits = torch.tensor([[-3.0, 5.0]])
    speech_second = torch.stack((
        torch.tensor([[-5.0, 5.0]]), fake_logits,
    ), dim=-1).reshape(1, 2, 2)
    nonspeech_second = speech_second.clone()
    nonspeech_second[..., 0] *= -1
    routed_high, _ = model.aggregate(speech_second, mask)
    routed_low, _ = model.aggregate(nonspeech_second, mask)
    assert routed_high > routed_low


def test_voice_mil_loss_is_finite():
    model = make_model()
    features = torch.randn(3, 2, 4)
    mask = torch.ones(3, 2, dtype=torch.bool)
    probability, logits, _ = model(features, mask)
    loss, terms = spear_voice_mil_loss(
        probability, logits, mask,
        torch.tensor([[1, 1], [1, 0], [1, 1]], dtype=torch.float32),
        torch.tensor([[0, 0], [1, 0], [1, 1]], dtype=torch.float32),
        torch.tensor([0, 1, 1], dtype=torch.float32),
        torch.ones(3),
    )
    assert torch.isfinite(loss)
    assert set(terms) == {"bag", "local_presence", "local_fake"}


def test_balancing_and_channel_pairs_are_group_aware():
    frame = pd.DataFrame({
        "ID": ["a", "a_phone", "b", "c"],
        "DATASET": ["clean", "phone", "other", "other"],
        "VOICE_GENERATOR_GROUP": ["g0", "g0", "g1", "g2"],
        "VOICE_FAKE": [0, 0, 1, 1],
        "MIXTURE_ID": [np.nan, np.nan, "m", "m"],
        "PARENT_ID": [np.nan, "a", np.nan, np.nan],
    })
    weights = sample_weights(frame)
    assert np.isfinite(weights).all() and weights.mean() == pytest.approx(1.0)
    assert set(map(tuple, channel_pairs(frame))) == {(0, 1), (2, 3)}


def test_voice_residual_changes_only_voice_column(tmp_path):
    model = make_model()
    checkpoint = tmp_path / "head.pt"
    projection = np.eye(4, dtype=np.float32)
    torch.save({
        "model_type": "spear_voice_mil_v1",
        "model": model.state_dict(),
        "config": {
            "feature_dimension": 4, "hidden": 8, "dropout": 0.0,
            "temperature": 2.0, "minimum_presence_weight": 0.05,
        },
        "projection": projection, "layers": np.asarray([0]),
        "bins": np.asarray(2),
    }, checkpoint)
    statistics = tmp_path / "stats.npz"
    np.savez_compressed(
        statistics, ids=np.asarray(["x"]),
        features=np.zeros((1, 1, 2, 1, 1, 4), dtype=np.float16),
        mask=np.ones((1, 1, 2), dtype=bool), projection=projection,
        layers=np.asarray([0]), bins=np.asarray(2),
    )
    submission = tmp_path / "submission.csv"
    columns = [
        "ID", "FILE_FAKE_PROB", "VOICE_FAKE_PROB", "MUSIC_FAKE_PROB",
        "VOICE_PRESENT_PROB", "MUSIC_PRESENT_PROB",
    ]
    original = ["x", "0.2000000001", "0.3000000002", "0.4000000003",
                "0.5000000004", "0.6000000005"]
    with submission.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle); writer.writerow(columns); writer.writerow(original)
    apply_spear_voice_mil_residual(
        submission, statistics, [checkpoint], device="cpu", voice_weight=.1,
    )
    with submission.open(newline="", encoding="utf-8") as handle:
        final = next(csv.reader(handle)); row = next(csv.reader(handle))
    assert final == columns
    assert row[2] != original[2]
    assert row[:2] + row[3:] == original[:2] + original[3:]
