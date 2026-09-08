from types import SimpleNamespace
import itertools
import json
import numpy as np
import pandas as pd
import pytest

from scripts.reserve_long_voice_v61 import candidate_record, choose_roles, IDENTITIES
from scripts.build_long_voice_v61 import render_insertion, render_provenance, uniform_start, SR
from scripts.audit_sparse_interval_coverage_v61 import interval_coverage
from scripts.validate_long_voice_v61 import validate_recipe


def fake_row(label='replay_bonafide', index=0):
    return SimpleNamespace(utt_id=f'utterance_{index}', source=f'content_{index}',
        source_speaker_id=f'speaker_{index}', label=label,
        synthesis_details=dict(model='generator', reference=f'reference_{index}',
                               reference_speaker_id=f'ref_speaker_{index}'))


def test_real_replay_does_not_become_fake():
    record, reason = candidate_record(fake_row(), 'shard', SimpleNamespace(exact=set()))
    assert reason == 'eligible'
    assert record['FILE_FAKE'] == 0
    assert record['RECORDING'] == 'replay'


def test_reference_speaker_protection_is_enforced():
    record, reason = candidate_record(fake_row('fake'), 'shard', SimpleNamespace(exact={'ref_speaker_0'}))
    assert record is None and reason == 'protected'


def test_all_roles_are_identity_disjoint_and_deterministic():
    pools = {}
    for offset, label in enumerate(['fake', 'replay_fake', 'bonafide', 'replay_bonafide']):
        pools[label] = [candidate_record(fake_row(label, 100 * offset + i), 'shard',
                        SimpleNamespace(exact=set()))[0] for i in range(12)]
    first = choose_roles(pools, 2, 5)
    pd.testing.assert_frame_equal(first, choose_roles(pools, 2, 5))
    assert len(first) == 10
    seen = set()
    for row in first.to_dict('records'):
        identities = {row[column] for column in IDENTITIES} - {''}
        assert not seen & identities
        seen.update(identities)


@pytest.mark.parametrize('duration', [30, 45, 60])
@pytest.mark.parametrize('position', ['early', 'middle', 'late'])
def test_fixed_length_and_real_fake_share_identical_editor(duration, position):
    t = np.arange(3 * SR, dtype=np.float32)
    background, inserted = np.sin(t * .03) * .1, np.sin(t * .04) * .1
    result, bounds, crop = render_insertion(background, inserted, duration, position)
    control, control_bounds, _ = render_insertion(background, background, duration, position)
    assert len(result) == duration * SR and np.isfinite(result).all()
    assert bounds[1] - bounds[0] == 2.0
    assert control_bounds == bounds and 0 <= crop <= 1
    start, end = [int(item * SR) for item in bounds]
    np.testing.assert_array_equal(result[:start], control[:start])
    np.testing.assert_array_equal(result[end:], control[end:])


def test_silent_insert_rejected():
    with pytest.raises(ValueError, match='silent'):
        render_insertion(np.ones(3 * SR), np.zeros(3 * SR), 30, 'middle')


def make_recipe():
    pools = {}
    for offset, label in enumerate(['fake', 'replay_fake', 'bonafide', 'replay_bonafide']):
        pools[label] = [candidate_record(fake_row(label, 100 * offset + i), 'shard',
                        SimpleNamespace(exact=set()))[0] for i in range(6)]
    sources = choose_roles(pools, 1, 5)
    roles = sources.set_index('ROLE').to_dict('index')
    rows = []
    channels = ['clean', 'g711_ulaw', 'opus_nb_8k']
    for duration, position, recording, label, channel in itertools.product(
            [30, 45, 60], ['early', 'middle', 'late'], ['direct', 'replay'], [0, 1], channels):
        first = roles['background']
        second = roles[('replay_' if recording == 'replay' else '') + ('fake_insert' if label else 'real_insert')]
        start = {'early': 1., 'middle': (duration - 2.) / 2., 'late': duration - 3.}[position]
        rows.append(dict(ID=str(len(rows)), GROUP_ID='lv61_000', DURATION=duration, POSITION=position,
            RECORDING=recording, FILE_FAKE=label, CHANNEL=channel, INSERTION_START=start, INSERTION_END=start + 2,
            FIRST_SOURCE_ID=first['ID'], SECOND_SOURCE_ID=second['ID'], FIRST_GROUP=first['VOICE_SPEAKER'],
            SECOND_GROUP=second['VOICE_SPEAKER'], GENERATOR=second['VOICE_GENERATOR'],
            FAKE_RANGES=json.dumps([[start, start + 2]] if label else [])))
    return pd.DataFrame(rows), sources, dict(groups=1, channels=channels)


def test_full_recipe_validator_accepts_complete_factorial():
    assert validate_recipe(*make_recipe()) == 108


@pytest.mark.parametrize('column,value', [('FILE_FAKE', 1), ('INSERTION_END', 99), ('FAKE_RANGES', '[[1,3]]')])
def test_recipe_validator_rejects_corrupted_labels_or_intervals(column, value):
    frame, sources, plan = make_recipe()
    frame.loc[0, column] = value
    with pytest.raises(ValueError):
        validate_recipe(frame, sources, plan)


def test_render_provenance_overrides_reserved_stage_without_duplicate_keyword(tmp_path):
    (tmp_path / 'source_hashes.csv').write_text('ID,SHA256\na,hash\n')
    result = render_provenance({'stage': 'reserved', 'full_bank_complete': False}, tmp_path, 2160)
    assert result['stage'] == 'rendered_unscored_requires_integrity_validation'
    assert result['rendered_rows'] == 2160
    assert result['full_bank_complete'] is False


def test_original_three_positions_cannot_test_five_view_blind_spots():
    for duration in (30, 45, 60):
        for start in (1., (duration - 2.) / 2, duration - 3.):
            assert interval_coverage(duration, start, start + 2, 5) == 1


def test_uniform_positions_include_actual_five_view_blind_spots_without_scores():
    fractions = []
    for duration in (30, 45, 60):
        for group in range(20):
            start = uniform_start(20260905, f'lv61_{group:03d}', duration)
            assert .25 <= start <= duration - 2.25
            fractions.append(interval_coverage(duration, start, start + 2, 5))
    assert min(fractions) == 0 and max(fractions) == 1


def test_uniform_insertion_bounds_follow_fixed_seed_time():
    audio = np.ones(3 * SR, dtype=np.float32) * .1
    start = uniform_start(20260905, 'lv61_000', 60)
    result, bounds, _ = render_insertion(audio, audio, 60, 'uniform', start_seconds=start)
    assert [round(v * SR) for v in bounds] == [round(start * SR), round(start * SR) + 2 * SR]
    assert len(result) == 60 * SR
