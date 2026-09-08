"""Label-blind component stitching and paired long/partial-mixture stress."""
import hashlib

import numpy as np

SR = 16000


def normalize(audio):
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim != 1 or len(audio) < SR // 2 or not np.isfinite(audio).all():
        raise ValueError('finite mono source with at least .5 seconds required')
    rms = float(np.sqrt(np.mean(audio.astype(np.float64) ** 2)))
    if rms < 1e-6:
        raise ValueError('silent source')
    return audio * min(.08 / rms, .9 / np.abs(audio).max())


def join_parts(parts, length, fade=320):
    """Repeat the same fixed-piece schedule for either authenticity class."""
    if not parts or length <= 0 or fade < 1 or any(len(p) <= fade * 2 for p in parts):
        raise ValueError('nonempty long-enough parts and positive length/fade required')
    values = parts[0].copy()
    cursor = 1
    ramp = np.linspace(0, 1, fade, dtype=np.float32)
    while len(values) < length:
        next_part = parts[cursor % len(parts)]
        blend = values[-fade:] * (1 - ramp) + next_part[:fade] * ramp
        values = np.concatenate([values[:-fade], blend, next_part[fade:]])
        cursor += 1
    return values[:length].astype(np.float32)


def make_stream(audios, seconds, piece_seconds, fade=320):
    pieces = [join_parts([normalize(audio)], round(piece_seconds * SR), fade) for audio in audios]
    return join_parts(pieces, round(seconds * SR), fade)


def replace_span(background, insertion, start, fade=320):
    if start < 0 or start + len(insertion) > len(background) or len(insertion) < 2 * fade:
        raise ValueError('invalid insertion geometry')
    result = background.copy()
    weight = np.ones(len(insertion), dtype=np.float32)
    weight[:fade] = np.linspace(0, 1, fade)
    weight[-fade:] = np.linspace(1, 0, fade)
    result[start:start + len(insertion)] = background[start:start + len(insertion)] * (1 - weight) + insertion * weight
    return result


def geometry_start(seed, group, seconds, insertion_seconds=2):
    digest = hashlib.sha256(f'{seed}|{group}|{seconds}|paired-placement'.encode()).hexdigest()
    fraction = int(digest[:16], 16) / 2**64
    return round((.25 + fraction * (seconds - insertion_seconds - .5)) * SR)


def render_case(streams, seconds, layout, case, snr_db, start, insertion_seconds=2, fade=320):
    """streams VR/VF/MR/MF already fixed; labels only choose source, not DSP.

Real controls receive an equally long real splice at the same time. Sparse
conditions retain a real background for the partly fake component.
"""
    if case not in {'RR', 'RF', 'FR', 'FF'} or not 4 <= seconds <= 60:
        raise ValueError('valid component case and competition duration required')
    length = round(seconds * SR)
    if any(len(streams[k]) < length for k in ['VR', 'VF', 'MR', 'MF']):
        raise ValueError('streams do not cover requested duration')
    voice_fake, music_fake = int(case[0] == 'F'), int(case[1] == 'F')
    voice = streams['V' + case[0]][:length].copy()
    music = streams['M' + case[1]][:length].copy()
    voice_intervals = [[0., float(seconds)]]
    music_intervals = [[0., float(seconds)]]
    fake_v = voice_intervals if voice_fake else []
    fake_m = music_intervals if music_fake else []
    if layout in {'sequential_voice_first', 'sequential_music_first'}:
        half = length // 2
        voice[:] = 0
        music[:] = 0
        offset_v, offset_m = (0, half) if layout == 'sequential_voice_first' else (half, 0)
        voice[offset_v:offset_v + half] = streams['V' + case[0]][:half]
        music[offset_m:offset_m + half] = streams['M' + case[1]][:half]
        # Equal fade-out/in around the boundary, without an authenticity cue.
        voice[offset_v:offset_v + fade] *= np.linspace(0, 1, fade)
        voice[offset_v + half - fade:offset_v + half] *= np.linspace(1, 0, fade)
        music[offset_m:offset_m + fade] *= np.linspace(0, 1, fade)
        music[offset_m + half - fade:offset_m + half] *= np.linspace(1, 0, fade)
        voice_intervals = [[offset_v / SR, (offset_v + half) / SR]]
        music_intervals = [[offset_m / SR, (offset_m + half) / SR]]
        fake_v = voice_intervals if voice_fake else []
        fake_m = music_intervals if music_fake else []
    elif layout in {'sparse_voice', 'sparse_music'}:
        component = 'V' if layout == 'sparse_voice' else 'M'
        label = case[0] if component == 'V' else case[1]
        segment = round(insertion_seconds * SR)
        # Same fixed crop offset for real and fake control insertions.
        inserted = streams[component + label][SR:SR + segment]
        changed = replace_span(streams[component + 'R'][:length], inserted, start, fade)
        interval = [[start / SR, (start + segment) / SR]] if label == 'F' else []
        if component == 'V':
            voice, fake_v = changed, interval
        else:
            music, fake_m = changed, interval
    elif layout != 'concurrent':
        raise ValueError('unknown layout')
    mixture = voice * float(10 ** (snr_db / 20)) + music
    mixture = mixture / max(float(np.abs(mixture).max()) / .98, 1.)
    metadata = dict(FILE_FAKE=max(voice_fake, music_fake), VOICE_FAKE=voice_fake, MUSIC_FAKE=music_fake,
                    VOICE_PRESENT=1, MUSIC_PRESENT=1, VOICE_INTERVALS=voice_intervals,
                    MUSIC_INTERVALS=music_intervals, VOICE_FAKE_INTERVALS=fake_v, MUSIC_FAKE_INTERVALS=fake_m)
    return mixture.astype(np.float32), metadata
