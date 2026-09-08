"""NumPy-version-neutral diagnostic metadata, not prediction postprocessing."""
import pandas as pd


def _coalesce(frame, columns):
    output = pd.Series('', index=frame.index, dtype=object)
    for column in columns:
        if column in frame:
            values = frame[column].fillna('').astype(str).str.strip()
            selected = output.eq('') & values.ne('')
            output.loc[selected] = values.loc[selected]
    return output.replace('', 'unknown')


def _component_generator(frame, columns, present_column, fake_column):
    result = pd.Series('', index=frame.index, dtype=object)
    for column in columns:
        if column not in frame:
            continue
        values = frame[column].fillna('').astype(str).str.strip()
        for index in frame.index[values.ne('')]:
            current = [item for item in result.loc[index].split('+') if item]
            value = values.loc[index]
            if value not in current:
                current.append(value)
            result.loc[index] = '+'.join(sorted(current))
    present = pd.to_numeric(frame[present_column], errors='coerce').eq(1)
    fake = pd.to_numeric(frame[fake_column], errors='coerce').eq(1)
    result.loc[~present] = 'absent'
    result.loc[present & result.eq('') & ~fake] = 'real_unspecified'
    result.loc[present & result.eq('') & fake] = 'unknown_fake'
    return result


def add_diagnostic_axes(frame):
    result = frame.copy()
    vp = pd.to_numeric(result.VOICE_PRESENT, errors='coerce').eq(1)
    mp = pd.to_numeric(result.MUSIC_PRESENT, errors='coerce').eq(1)
    vf = pd.to_numeric(result.VOICE_FAKE, errors='coerce').fillna(0).eq(1)
    mf = pd.to_numeric(result.MUSIC_FAKE, errors='coerce').fillna(0).eq(1)
    mixed = vp & mp
    result['CELL_V57'] = 'not_mixed'
    # Unlike NumPy 2.x, NumPy 1.26 cannot add Unicode ndarrays with '+'.
    result.loc[mixed, 'CELL_V57'] = [('F' if v else 'R') + ('F' if m else 'R')
        for v, m in zip(vf[mixed], mf[mixed])]
    result['LAYOUT_V57'] = _coalesce(result, ('MIX_MODE', 'LAYOUT', 'CONDITION', 'MUSIC_LAYOUT', 'CONVERSATION_MODE'))
    result['CHANNEL_V57'] = _coalesce(result, ('CHANNEL', 'STRESS_VARIANT', 'CODEC'))
    result['VOICE_GENERATOR_V57'] = _component_generator(result,
        ('VOICE_GENERATOR', 'FIRST_GENERATOR', 'SECOND_GENERATOR', 'ATTACK'), 'VOICE_PRESENT', 'VOICE_FAKE')
    result['MUSIC_GENERATOR_V57'] = _component_generator(result,
        ('MUSIC_GENERATOR',), 'MUSIC_PRESENT', 'MUSIC_FAKE')
    return result
