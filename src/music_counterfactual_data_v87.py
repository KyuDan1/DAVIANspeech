"""TRAIN-source counterfactual music pairs; no generated/separated waveforms.

The two views use identical music crop, gain, channel and geometry, but different
voice sources (real/fake). Only Music labels are supplied: source music may have
vocals, so foreground authenticity cannot define Voice authenticity.
"""
import numpy as np

from .dense_component_data_v71 import cached_audio
from .long_component_stress import SR, normalize
from .telephone_channel import apply_channel


def crop_or_tile(audio, samples, rng):
    """Random native crop, or label-blind tiling only for short TRAIN sources."""
    audio = normalize(audio)
    if len(audio) >= samples:
        start = int(rng.integers(len(audio) - samples + 1))
        return audio[start:start + samples].copy(), dict(start_sample=start, tiled=False)
    from .long_component_stress import join_parts
    return join_parts([audio], samples), dict(start_sample=0, tiled=True)


def paired_mixtures(music, real_voice, fake_voice, snr_db, layout):
    """Same music waveform and gain in both outputs, including peak scaling."""
    if not (music.ndim == real_voice.ndim == fake_voice.ndim == 1):
        raise ValueError('mono inputs required')
    if not (len(music) == len(real_voice) == len(fake_voice)) or len(music) < SR:
        raise ValueError('equal nonempty lengths required')
    if not all(np.isfinite(x).all() for x in [music, real_voice, fake_voice]):
        raise ValueError('finite inputs required')
    m = music.copy()
    voices = [real_voice.copy(), fake_voice.copy()]
    if layout == 'concurrent':
        pass
    elif layout in ['voice_first', 'music_first']:
        half = len(m) // 2
        # Same deterministic 20ms fades for both authenticity views.
        voice_mask = np.zeros(len(m), np.float32)
        start, end = (0, half) if layout == 'voice_first' else (half, len(m))
        voice_mask[start:end] = 1.
        fade = min(320, (end-start)//2)
        voice_mask[start:start+fade] = np.linspace(0, 1, fade)
        voice_mask[end-fade:end] = np.linspace(1, 0, fade)
        music_mask = np.zeros(len(m), np.float32)
        start, end = (half, len(m)) if layout == 'voice_first' else (0, half)
        music_mask[start:end] = 1.
        music_mask[start:start+fade] = np.linspace(0, 1, fade)
        music_mask[end-fade:end] = np.linspace(1, 0, fade)
        m *= music_mask
        voices = [v * voice_mask for v in voices]
    else:
        raise ValueError('unknown layout')
    gain = float(10 ** (snr_db / 20))
    views = np.stack([m + gain * v for v in voices])
    # One scale for the entire TRAIN pair: avoid changed music loudness caused by
    # independent per-view peak normalization. Not used in evaluation inference.
    scale = max(float(np.abs(views).max()) / .98, 1.)
    return (views / scale).astype(np.float32)


class MusicCounterfactualPairs:
    def __init__(self, catalog, seed, ffmpeg):
        self.catalog = catalog.reset_index(drop=True)
        self.seed, self.ffmpeg = seed, ffmpeg
        self.pools = {}
        for component, label in [('MUSIC', 0), ('MUSIC', 1), ('VOICE', 0), ('VOICE', 1)]:
            subset = self.catalog[self.catalog.COMPONENT.eq(component) & self.catalog.LABEL.eq(label)]
            if not len(subset):
                raise ValueError('all four TRAIN source pools required')
            self.pools[component, label] = [group.index.to_numpy() for _, group in subset.groupby('GENERATOR', dropna=False)]

    def draw(self, epoch, index):
        rng = np.random.default_rng(np.random.SeedSequence([self.seed, epoch, index]))
        label = int(rng.integers(2))
        seconds = int(rng.choice([4, 8]))
        layout = str(rng.choice(['concurrent', 'voice_first', 'music_first']))
        snr = float(rng.choice([-10, -5, 0, 5, 10]))
        channel = str(rng.choice(['clean', 'g711_ulaw', 'g722_wb', 'opus_nb_8k', 'transcode_g711_opus']))
        channel_key = int(rng.integers(2**31))
        audio, sources = [], []
        for component, source_label in [('MUSIC', label), ('VOICE', 0), ('VOICE', 1)]:
            families = self.pools[component, source_label]
            family = families[int(rng.integers(len(families)))]
            row = self.catalog.iloc[int(rng.choice(family))]
            original, digest = cached_audio(row.PATH)
            crop, geometry = crop_or_tile(original, seconds * SR, rng)
            audio.append(crop)
            sources.append(dict(id=row.ID, group=row.GROUP_ID, sha256=digest,
                                component=component, label=source_label, **geometry))
        views = paired_mixtures(*audio, snr_db=snr, layout=layout)
        views = np.stack([apply_channel(v, channel, self.ffmpeg, key=channel_key) for v in views])
        if views.shape != (2, seconds * SR) or not np.isfinite(views).all():
            raise ValueError('invalid transformed geometry')
        trace = dict(epoch=epoch, draw=index, kind='TRAIN_music_counterfactual',
                     music_label=label, sources=sources, seconds=seconds, layout=layout,
                     snr_db=snr, channel=channel, channel_key=channel_key,
                     scope='Music labels only; voice authenticity in music is not asserted')
        return views, label, trace
