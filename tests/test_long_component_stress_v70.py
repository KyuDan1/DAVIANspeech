import numpy as np
import pytest

from src.long_component_stress import SR, geometry_start, join_parts, make_stream, render_case


def streams(seconds=10):
    t = np.arange(seconds * SR) / SR
    return {key: (.05 * np.sin(2 * np.pi * freq * t)).astype('float32')
            for key, freq in [('VR', 211), ('VF', 367), ('MR', 587), ('MF', 733)]}


@pytest.mark.parametrize('layout', ['concurrent', 'sequential_voice_first', 'sequential_music_first', 'sparse_voice', 'sparse_music'])
@pytest.mark.parametrize('case', ['RR', 'RF', 'FR', 'FF'])
def test_component_labels_and_finite_exact_duration(layout, case):
    audio, meta = render_case(streams(), 10, layout, case, 0, 3 * SR)
    assert len(audio) == 10 * SR and np.isfinite(audio).all() and np.abs(audio).max() <= .98
    assert meta['VOICE_PRESENT'] == meta['MUSIC_PRESENT'] == 1
    assert meta['VOICE_FAKE'] == int(case[0] == 'F')
    assert meta['MUSIC_FAKE'] == int(case[1] == 'F')
    assert meta['FILE_FAKE'] == max(meta['VOICE_FAKE'], meta['MUSIC_FAKE'])
    assert bool(meta['VOICE_FAKE_INTERVALS']) == bool(meta['VOICE_FAKE'])
    assert bool(meta['MUSIC_FAKE_INTERVALS']) == bool(meta['MUSIC_FAKE'])
    if layout == 'sparse_voice' and case[0] == 'F':
        assert meta['VOICE_FAKE_INTERVALS'] == [[3., 5.]]
    if layout == 'sparse_music' and case[1] == 'F':
        assert meta['MUSIC_FAKE_INTERVALS'] == [[3., 5.]]


def test_sparse_counterfactual_changes_only_inserted_region():
    source = streams()
    for layout in ['sparse_voice', 'sparse_music']:
        case = 'FR' if layout == 'sparse_voice' else 'RF'
        real, _ = render_case(source, 10, layout, 'RR', 0, 3 * SR)
        fake, _ = render_case(source, 10, layout, case, 0, 3 * SR)
        np.testing.assert_array_equal(real[:3 * SR], fake[:3 * SR])
        np.testing.assert_array_equal(real[5 * SR:], fake[5 * SR:])
        assert not np.array_equal(real[3 * SR:5 * SR], fake[3 * SR:5 * SR])


def test_voice_and_music_order_are_distinct_and_label_stable():
    source = streams()
    first, a = render_case(source, 10, 'sequential_voice_first', 'FR', 0, SR)
    second, b = render_case(source, 10, 'sequential_music_first', 'FR', 0, SR)
    assert a['VOICE_INTERVALS'] == [[0., 5.]] and b['VOICE_INTERVALS'] == [[5., 10.]]
    assert a['FILE_FAKE'] == b['FILE_FAKE'] == 1 and not np.array_equal(first, second)


def test_looping_is_explicit_deterministic_and_not_empty():
    part = streams(1)['VR']
    a = make_stream([part], 30, 4)
    np.testing.assert_array_equal(a, make_stream([part], 30, 4))
    assert len(a) == 30 * SR and np.isfinite(a).all()
    with pytest.raises(ValueError):
        join_parts([], SR)


def test_position_is_paired_label_blind_and_in_range():
    for seconds in [10, 30, 60]:
        starts = [geometry_start(20260905, f'lcs70_{i}', seconds) for i in range(10)]
        assert starts == [geometry_start(20260905, f'lcs70_{i}', seconds) for i in range(10)]
        assert all(.25 * SR <= start <= (seconds - 2.25) * SR for start in starts)
        assert len(set(starts)) == 10


def test_unknown_layout_and_out_of_range_duration_rejected():
    with pytest.raises(ValueError):
        render_case(streams(), 10, 'sequential_invalid', 'RR', 0, SR)
    with pytest.raises(ValueError):
        render_case(streams(), 2, 'concurrent', 'RR', 0, SR)
