"""Online TRAIN-only scenes, paired real/fake editing operations and traces."""
import hashlib
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from .common_encoder_probe import complete_windows
from .dense_component_v71 import interval_targets
from .evaluate_diagnostic import LABEL_COLUMNS
from .long_component_stress import SR, make_stream, render_case, replace_span
from .pipeline import load_audio
from .telephone_channel import apply_channel


def source_sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for piece in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(piece)
    return digest.hexdigest()


@lru_cache(maxsize=128)
def cached_audio(path):
    # Hash each actually consumed raw payload, not merely its truth table.
    before = source_sha256(path)
    audio = load_audio(Path(path))
    if source_sha256(path) != before:
        raise ValueError('source changed during audio read')
    return audio, before


def temporal_row_metadata(row):
    if str(row.DATASET) != 'temporal_mixed_train_v2':
        return None
    result = {}
    for component in ['VOICE', 'MUSIC']:
        spans = [[float(row[component + '_START']), float(row[component + '_END'])]]
        result[component + '_INTERVALS'] = spans if row[component + '_PRESENT'] == 1 else []
        result[component + '_FAKE_INTERVALS'] = spans if row[component + '_FAKE'] == 1 else []
    return result


class DenseTrainingBags(Dataset):
    def __init__(self, frame, catalog, config, probabilities, ffmpeg, draws, epoch=1):
        self.frame, self.catalog = frame.reset_index(drop=True), catalog.reset_index(drop=True)
        self.config, self.ffmpeg = config, ffmpeg
        self.probabilities = np.asarray(probabilities, np.float64)
        self.probabilities /= self.probabilities.sum()
        self.draws, self.epoch = draws, epoch
        self.pools = {}
        for component, prefix in [('VOICE', 'V'), ('MUSIC', 'M')]:
            for label, suffix in [(0, 'R'), (1, 'F')]:
                subset = self.catalog[self.catalog.COMPONENT.eq(component) & self.catalog.LABEL.eq(label)]
                self.pools[prefix + suffix] = [part.index.to_numpy() for _, part in subset.groupby('GENERATOR', dropna=False)]
                if not len(subset):
                    raise ValueError('all four authorized source pools required')

    def __len__(self):
        return self.draws

    def choose_source(self, pool, rng, exclude_group=None):
        families = self.pools[pool]
        if exclude_group is not None:
            families = [indices[self.catalog.loc[indices, 'GROUP_ID'].ne(exclude_group).to_numpy()]
                        for indices in families]
            families = [indices for indices in families if len(indices)]
        if not families:
            raise ValueError('independent real insertion group unavailable')
        family = families[int(rng.integers(len(families)))]
        row = self.catalog.iloc[int(rng.choice(family))]
        audio, checksum = cached_audio(row.PATH)
        return audio, dict(id=row.ID, group=row.GROUP_ID, sha256=checksum)

    def scene(self, rng):
        cfg = self.config
        duration = float(rng.choice(cfg['synthetic_durations']))
        layout = str(rng.choice(cfg['synthetic_layouts']))
        case = str(rng.choice(['RR', 'RF', 'FR', 'FF']))
        insertion = min(float(rng.choice(cfg['insertion_durations'])), duration / 2)
        start = int(rng.integers(round(.1 * SR), round((duration - insertion - .1) * SR) + 1))
        streams, sources = {}, {}
        # Generator-balanced raw selection; no protected evaluation payloads.
        for pool in self.pools:
            audio, record = self.choose_source(pool, rng)
            piece = float(rng.choice([2, 3, 5])) if pool[0] == 'V' else 8.
            streams[pool] = make_stream([audio], duration, piece)
            sources[pool] = record
        alternate_real = None
        if layout in {'sparse_voice', 'sparse_music'}:
            component = 'V' if layout == 'sparse_voice' else 'M'
            # Both authenticity classes draw this source, so later DSP/channel
            # draws do not become label-dependent. Real controls change source
            # group too; otherwise speaker/song changes could reveal fake labels.
            audio, record = self.choose_source(component + 'R', rng, sources[component + 'R']['group'])
            alternate_real = make_stream([audio], duration, 3. if component == 'V' else 8.)
            sources['REAL_INSERT'] = record
        if layout in {'pure_voice', 'pure_music'}:
            is_voice = layout == 'pure_voice'
            label = int(case[0 if is_voice else 1] == 'F')
            audio = streams[('V' if is_voice else 'M') + ('F' if label else 'R')].copy()
            spans = [[0., duration]]
            metadata = dict(FILE_FAKE=label, VOICE_FAKE=label if is_voice else 0,
                MUSIC_FAKE=label if not is_voice else 0, VOICE_PRESENT=int(is_voice), MUSIC_PRESENT=int(not is_voice),
                VOICE_INTERVALS=spans if is_voice else [], MUSIC_INTERVALS=spans if not is_voice else [],
                VOICE_FAKE_INTERVALS=spans if is_voice and label else [],
                MUSIC_FAKE_INTERVALS=spans if not is_voice and label else [])
            snr = None
        else:
            snr = float(rng.choice(cfg['synthetic_snr']))
            audio, metadata = render_case(streams, duration, layout, case, snr, start, insertion)
            if alternate_real is not None:
                label = case[0 if component == 'V' else 1]
                if label == 'R':
                    inserted = alternate_real[SR:SR + round(insertion * SR)]
                    changed = replace_span(streams[component + 'R'], inserted, start)
                    voice = changed if component == 'V' else streams['V' + case[0]]
                    music = changed if component == 'M' else streams['M' + case[1]]
                    audio = voice * float(10 ** (snr / 20)) + music
                    audio = (audio / max(float(np.abs(audio).max()) / .98, 1.)).astype(np.float32)
        channel = str(rng.choice(cfg['synthetic_channels']))
        audio = apply_channel(audio, channel, self.ffmpeg, key=int(rng.integers(2**31)))
        if len(audio) != round(duration * SR) or not np.isfinite(audio).all():
            raise ValueError('channel transform changed annotated audio geometry')
        target = np.asarray([metadata[key] for key in LABEL_COLUMNS], np.float32)
        trace = dict(kind='synthetic_train', sources=sources, layout=layout, component_case=case,
            seconds=duration, insertion_seconds=insertion, start_sample=start, snr_db=snr,
            channel=channel, intervals=metadata)
        return audio, target, metadata, trace

    def __getitem__(self, index):
        rng = np.random.default_rng(np.random.SeedSequence([self.config['seed'], self.epoch, index]))
        if rng.random() < self.config['synthetic_probability']:
            audio, target, metadata, trace = self.scene(rng)
        else:
            row_index = int(rng.choice(len(self.frame), p=self.probabilities))
            row = self.frame.iloc[row_index]
            audio = load_audio(Path(row.PATH))
            target = row[LABEL_COLUMNS].to_numpy(np.float32)
            metadata = temporal_row_metadata(row)
            trace = dict(kind='original_train', row=row_index, dataset=row.DATASET, id=row.ID)
        windows, lengths, starts = complete_windows(audio)
        dense, valid = interval_targets(metadata, starts, lengths, self.config['boundary_margin_seconds'])
        trace.update(epoch=self.epoch, draw=int(index))
        return windows, lengths, target, dense, valid, trace


def dense_collate(items):
    return (torch.from_numpy(np.concatenate([x[0] for x in items])),
            torch.from_numpy(np.concatenate([x[1] for x in items])), [len(x[0]) for x in items],
            torch.from_numpy(np.stack([x[2] for x in items])),
            torch.from_numpy(np.concatenate([x[3] for x in items])),
            torch.from_numpy(np.concatenate([x[4] for x in items])), [x[5] for x in items])
