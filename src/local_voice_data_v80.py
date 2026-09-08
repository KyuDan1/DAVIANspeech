"""Matched real/partly-fake Voice pairs from an audited strict TRAIN catalog."""
import numpy as np
from torch.utils.data import Dataset
from .dense_component_data_v71 import cached_audio
from .long_component_stress import make_stream, replace_span, SR
from .telephone_channel import apply_channel


def identity_collate(item):
    return item


class LocalVoicePairs(Dataset):
    def __init__(self, catalog, config, ffmpeg, epoch=1, pairs=None):
        self.catalog, self.config, self.ffmpeg, self.epoch = catalog, config, ffmpeg, epoch
        self.pairs = config['pairs_per_epoch'] if pairs is None else pairs

    def __len__(self):
        return self.pairs

    def source(self, component, label, rng, exclude=None):
        frame=self.catalog[self.catalog.COMPONENT.eq(component)&self.catalog.LABEL.eq(label)]
        if exclude is not None:
            frame=frame[frame.GROUP_ID.ne(exclude)]
        families=list(frame.groupby('GENERATOR',dropna=False))
        if not families:
            raise ValueError('authorized independent source unavailable')
        family=families[int(rng.integers(len(families)))][1]
        row=family.iloc[int(rng.integers(len(family)))]
        audio,digest=cached_audio(row.PATH)
        return audio,dict(id=row.ID,group=row.GROUP_ID,sha256=digest,generator=row.GENERATOR)

    def __getitem__(self,index):
        cfg=self.config
        rng=np.random.default_rng(np.random.SeedSequence([cfg['seed'],self.epoch,index]))
        duration=float(rng.choice(cfg['durations']))
        insertion=min(float(rng.choice(cfg['insertions'])),duration/2)
        start=int(rng.integers(round(.1*SR),round((duration-insertion-.1)*SR)+1))
        mixed=bool(rng.integers(2))
        piece=float(rng.choice([2,3,5]))
        snr=float(rng.choice(cfg['snr']))
        channel=str(rng.choice(cfg['channels']))
        key=int(rng.integers(2**31))
        bg,bg_trace=self.source('VOICE',0,rng)
        real,real_trace=self.source('VOICE',0,rng,bg_trace['group'])
        fake,fake_trace=self.source('VOICE',1,rng,bg_trace['group'])
        music,music_trace=self.source('MUSIC',0,rng)
        bg=make_stream([bg],duration,piece)
        music=make_stream([music],duration,8.)
        outputs=[]
        for label,source in enumerate([real,fake]):
            stream=make_stream([source],duration,piece)
            voice=replace_span(bg,stream[SR:SR+round(insertion*SR)],start)
            audio=voice*float(10**(snr/20))+music if mixed else voice
            audio=(audio/max(float(np.abs(audio).max())/.98,1.)).astype(np.float32)
            audio=apply_channel(audio,channel,self.ffmpeg,key=key)
            if len(audio)!=round(duration*SR) or not np.isfinite(audio).all():
                raise ValueError('codec changed labeled geometry')
            spans=[[start/SR,start/SR+insertion]] if label else []
            outputs.append((audio,label,spans))
        trace=dict(epoch=self.epoch,pair=int(index),duration=duration,insertion=insertion,
            start_sample=start,mixed=mixed,snr=snr if mixed else None,channel=channel,codec_key=key,
            sources=dict(background=bg_trace,real_insert=real_trace,fake_insert=fake_trace,
                         real_music=music_trace if mixed else None))
        return outputs,trace
