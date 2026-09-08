"""TRAIN-only 2x2 music/foreground factorial scenes for Music ranking."""
import numpy as np

from .dense_component_data_v71 import cached_audio
from .long_component_stress import SR
from .music_counterfactual_data_v87 import crop_or_tile
from .music_bank_sampler_v88 import BankBalancedMusicPairs
from .telephone_channel import apply_channel


def factorial_mixtures(real_music, fake_music, real_voice, fake_voice, snr_db, layout):
    audios = [real_music, fake_music, real_voice, fake_voice]
    if any(x.ndim != 1 for x in audios) or len({len(x) for x in audios}) != 1 or len(audios[0]) < SR:
        raise ValueError('four equal mono inputs required')
    if not all(np.isfinite(x).all() for x in audios):
        raise ValueError('finite inputs required')
    music_mask = np.ones(len(real_music), np.float32)
    voice_mask = np.ones(len(real_music), np.float32)
    if layout in ['voice_first', 'music_first']:
        half = len(real_music) // 2
        music_mask[:] = 0
        voice_mask[:] = 0
        vstart, vend = (0, half) if layout == 'voice_first' else (half, len(real_music))
        mstart, mend = (half, len(real_music)) if layout == 'voice_first' else (0, half)
        voice_mask[vstart:vend] = 1
        music_mask[mstart:mend] = 1
        fade = min(320, half // 2)
        voice_mask[vstart:vstart+fade] = np.linspace(0, 1, fade)
        voice_mask[vend-fade:vend] = np.linspace(1, 0, fade)
        music_mask[mstart:mstart+fade] = np.linspace(0, 1, fade)
        music_mask[mend-fade:mend] = np.linspace(1, 0, fade)
    elif layout != 'concurrent':
        raise ValueError('unknown layout')
    musics = [real_music * music_mask, fake_music * music_mask]
    voices = [real_voice * voice_mask, fake_voice * voice_mask]
    gain = float(10 ** (snr_db / 20))
    # Ordering makes labels and paired ranking explicit: MR/VR, MR/VF, MF/VR, MF/VF.
    views = np.stack([m + gain*v for m in musics for v in voices])
    # One scale across the 2x2 intervention prevents per-label peak normalization cues.
    scale = max(float(np.abs(views).max()) / .98, 1.)
    return (views / scale).astype(np.float32)


class MusicFactorialQuadruplets(BankBalancedMusicPairs):
    def choose(self, component, label, rng, samples):
        families = self.pools[component, label]
        family = families[int(rng.integers(len(families)))]
        row = self.catalog.iloc[int(rng.choice(family))]
        original, digest = cached_audio(row.PATH)
        audio, geometry = crop_or_tile(original, samples, rng)
        return audio, dict(id=row.ID, group=row.GROUP_ID, bank=row.SOURCE_BANK,
                           generator=row.GENERATOR, sha256=digest, component=component,
                           label=label, **geometry)

    def draw_factorial(self, epoch, index):
        rng = np.random.default_rng(np.random.SeedSequence([self.seed, epoch, index, 89]))
        seconds = int(rng.choice([4, 8]))
        layout = str(rng.choice(['concurrent', 'voice_first', 'music_first']))
        snr = float(rng.choice([-10, -5, 0, 5, 10]))
        channel = str(rng.choice(['clean', 'g711_ulaw', 'g722_wb', 'opus_nb_8k', 'transcode_g711_opus']))
        key = int(rng.integers(2**31))
        items = [self.choose(component, label, rng, seconds*SR)
                 for component, label in [('MUSIC', 0), ('MUSIC', 1), ('VOICE', 0), ('VOICE', 1)]]
        views = factorial_mixtures(*(x[0] for x in items), snr_db=snr, layout=layout)
        views = np.stack([apply_channel(v, channel, self.ffmpeg, key=key) for v in views])
        if views.shape != (4, seconds*SR) or not np.isfinite(views).all():
            raise ValueError('invalid transformed factorial')
        trace = dict(kind='TRAIN_music_factorial', epoch=epoch, draw=index,
                     sources=[x[1] for x in items], seconds=seconds, layout=layout,
                     snr_db=snr, channel=channel, channel_key=key,
                     music_labels=[0, 0, 1, 1], voice_interventions=[0, 1, 0, 1],
                     scope='Music labels only; bank-balanced; no separator/generator operation')
        return views, np.asarray([0, 0, 1, 1], np.float32), trace
