import numpy as np
import pytest
import torch

from src.dense_component_v71 import (DenseComponentHead, interval_targets,
    dense_component_loss, aggregate_files, paired_dense_logits)
from src.long_component_stress import render_case


def metadata():
    return dict(VOICE_INTERVALS=[[0., 10.24]], MUSIC_INTERVALS=[],
                VOICE_FAKE_INTERVALS=[[3., 5.]], MUSIC_FAKE_INTERVALS=[])


def test_unknown_metadata_never_invents_dense_real_labels():
    _, valid = interval_targets(None, [0], [163840])
    assert not valid.any()


def test_interval_or_and_absent_music_mask():
    labels, valid = interval_targets(metadata(), [0], [163840])
    assert labels[0, 23, :].tolist() == [1, 1, 0, 1, 0]
    assert labels[0, 10, :].tolist() == [0, 0, 0, 1, 0]
    assert not valid[..., 2].any()
    np.testing.assert_array_equal(labels[..., 0], labels[..., 1])


def test_window_offset_and_padding_are_respected():
    labels, valid = interval_targets(metadata(), [48000], [32000], margin=0.)
    assert labels[0, :12, 1].all()
    assert not valid[0, 12:].any()


def test_boundary_bins_not_supervised():
    data = metadata()
    data['VOICE_FAKE_INTERVALS'] = [[3.12, 5.04]]
    _, valid = interval_targets(data, [0], [163840])
    assert not valid[0, 19].any()
    assert not valid[0, 31].any()


def test_invalid_fake_outside_presence_rejected():
    data = metadata()
    data['VOICE_INTERVALS'] = [[0., 2.]]
    with pytest.raises(ValueError, match='presence'):
        interval_targets(data, [0], [163840])


def test_dense_loss_ignores_absent_and_unknown_and_has_gradients():
    labels, valid = interval_targets(metadata(), [0], [163840])
    logits = torch.zeros((1, 64, 5), requires_grad=True)
    loss = dense_component_loss(logits, torch.tensor(labels), torch.tensor(valid))
    loss.backward()
    assert torch.isfinite(loss)
    assert logits.grad[..., 1].abs().sum() > 0
    assert logits.grad[..., 2].abs().sum() == 0
    unknown = dense_component_loss(logits, torch.full_like(logits, float('nan')), torch.zeros_like(logits, dtype=torch.bool))
    assert unknown == 0


def test_positive_negative_balancing_invariant_to_repeated_negatives():
    logits = torch.tensor([[[1.] * 5, [2.] * 5]])
    labels = torch.tensor([[[0.] * 5, [1.] * 5]])
    first = dense_component_loss(logits, labels, torch.ones_like(labels, dtype=torch.bool))
    extended = torch.cat([logits[:, :1].repeat(1, 9, 1), logits[:, 1:]], 1)
    target = torch.cat([labels[:, :1].repeat(1, 9, 1), labels[:, 1:]], 1)
    second = dense_component_loss(extended, target, torch.ones_like(target, dtype=torch.bool))
    torch.testing.assert_close(first, second)


def test_temporal_head_keeps_grid_and_file_independence():
    torch.manual_seed(71)
    head = DenseComponentHead(dimension=4, width=3)
    tokens = torch.randn(2, 3, 512, 4)
    mask = torch.ones(2, 512, dtype=torch.bool)
    mask[0, 80:] = False
    batched, valid = head(tokens, mask)
    single, _ = head(tokens[:1], mask[:1])
    torch.testing.assert_close(batched[:1], single)
    assert batched.shape == (2, 64, 5)
    assert valid[0].sum() == 10
    assert torch.count_nonzero(batched[0, 10:]) == 0
    changed = tokens.clone()
    changed[0, :, 80:] = float('nan')
    alternate, _ = head(changed, mask)
    torch.testing.assert_close(batched, alternate)


def test_file_pooling_does_not_use_neighbor_scores():
    values = torch.randn(3, 64, 5)
    valid = torch.ones(3, 64, dtype=torch.bool)
    batch = aggregate_files(values, valid, [1, 2], 5.)
    one = aggregate_files(values[:1], valid[:1], [1], 5.)
    torch.testing.assert_close(batch[:1], one)
    with pytest.raises(ValueError):
        aggregate_files(values, valid, [1], 5.)


def test_one_encoder_forward_shared_by_both_heads():
    calls = []
    class Encoder:
        def __call__(self, windows, lengths):
            calls.append(len(windows))
            return {'adapted': torch.zeros(len(windows), 3, 512, 4)}, torch.ones(len(windows), 512, dtype=torch.bool)
    heads = {name: DenseComponentHead(dimension=4, width=3) for name in ['file_only', 'dense_supervised']}
    pooled, dense, valid = paired_dense_logits(Encoder(), heads, torch.zeros(5, 12), torch.ones(5), [2, 3], 2)
    assert calls == [2, 2, 1]  # not twice for the two models
    assert all(x.shape == (2, 5) for x in pooled.values())
    assert all(x.shape == (5, 64, 5) for x in dense.values())
    assert valid.shape == (5, 64)


@pytest.mark.parametrize('layout,task', [('sparse_voice', 1), ('sparse_music', 2)])
def test_partial_component_targets_only_inserted_region(layout, task):
    streams = {key: np.full(16000 * 8, .05, np.float32) for key in ['VR', 'VF', 'MR', 'MF']}
    _, meta = render_case(streams, 8, layout, 'FR' if task == 1 else 'RF', 0, 32000, 1.)
    labels, valid = interval_targets(meta, [0], [128000])
    positives = labels[0, :, task].astype(bool) & valid[0, :, task]
    assert 4 <= positives.sum() <= 7
    assert labels[0, :50, task + 2].all()  # presence remains throughout actual audio


@pytest.mark.parametrize('layout', ['sparse_voice', 'sparse_music', 'pure_voice', 'pure_music',
    'concurrent', 'sequential_voice_first', 'sequential_music_first'])
def test_training_renderer_deterministic_and_real_controls_change_group(monkeypatch, layout):
    import pandas as pd
    import src.dense_component_data_v71 as data
    source = np.sin(np.arange(64000, dtype=np.float32) * .07) * .1
    monkeypatch.setattr(data, 'cached_audio', lambda _: (source, 'test-hash'))
    catalog = pd.DataFrame([dict(ID=f'{component}{label}{index}', COMPONENT=component, LABEL=label,
        GROUP_ID=f'{component}{label}{index}', GENERATOR='test', PATH='unused')
        for component in ['VOICE', 'MUSIC'] for label in [0, 1] for index in range(2)])
    cfg = dict(seed=71, synthetic_durations=[4], synthetic_layouts=[layout], insertion_durations=[.5],
               synthetic_snr=[0], synthetic_channels=['clean'])
    dataset = data.DenseTrainingBags(pd.DataFrame([{'ID': 'test'}]), catalog, cfg, [1.], None, draws=1)
    first = dataset.scene(np.random.default_rng(71))
    repeat = dataset.scene(np.random.default_rng(71))
    np.testing.assert_array_equal(first[0], repeat[0])
    assert first[3] == repeat[3]
    assert len(first[0]) == 64000
    assert first[1][0] == max(first[1][1], first[1][2])
    if layout.startswith('sparse'):
        component = 'V' if layout == 'sparse_voice' else 'M'
        assert first[3]['sources'][component + 'R']['group'] != first[3]['sources']['REAL_INSERT']['group']
    labels, valid = interval_targets(first[2], [0], [64000])
    for task in range(5):
        if valid[..., task].any():
            assert labels[..., task][valid[..., task]].max() == first[1][task]


@pytest.mark.parametrize('channel', ['clean', 'g711_ulaw', 'g722_wb', 'opus_nb_8k', 'transcode_g711_opus'])
def test_training_channels_preserve_annotated_duration(channel):
    from src.full_coverage_wpt import resolve_ffmpeg
    from src.telephone_channel import apply_channel
    source = np.sin(np.arange(64000, dtype=np.float32) * .07) * .1
    changed = apply_channel(source, channel, resolve_ffmpeg(), key=71)
    assert len(changed) == len(source)
    assert np.isfinite(changed).all()
