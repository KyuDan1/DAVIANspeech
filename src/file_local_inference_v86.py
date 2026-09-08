"""File-local inference of completed v86 matched global/local readouts."""
import json
from pathlib import Path
import numpy as np
import torch
from .common_encoder_probe import CommonTokenHead,FrozenEncoderTokens,complete_windows
from .local_voice_v80 import LocalVoiceHead
from .file_local_readout_v86 import file_statistics,lme


class FileLocalPredictor:
    def __init__(self,run,root,device='cuda'):
        self.run,self.root=Path(run),Path(root)
        report=json.loads((self.run/'report.json').read_text())
        frozen=json.loads((self.run/'frozen.json').read_text())
        if report['status']!='complete' or frozen['smoke']:
            raise ValueError('completed full training required')
        self.config=frozen['config']
        self.encoder=FrozenEncoderTokens('xlsr',self.root/self.config['encoder'],device=device)
        self.parent=CommonTokenHead(1920,width=96,pooling='mean').to(device).eval()
        parent_state=torch.load(self.root/self.config['parent_head'],map_location='cpu',weights_only=False)
        self.parent.load_state_dict(parent_state['state_dict'])
        self.parent.requires_grad_(False)
        self.heads={}
        for name in ['global','local']:
            state=torch.load(self.run/f'{name}.pt',map_location='cpu',weights_only=False)
            if state['smoke'] or state['config']!=self.config or state['variant']!=name or state['scope']!='File only':
                raise ValueError('checkpoint metadata mismatch')
            head=LocalVoiceHead(self.parent.output_weight[0],self.parent.output_bias[0]).to(device).eval()
            head.load_state_dict(state['state_dict'],strict=True)
            self.heads[name]=head

    @torch.no_grad()
    def __call__(self,audio):
        windows,lengths,_=complete_windows(audio)
        pieces={'global':[],'local':[]};original=[]
        for window,length in zip(windows,lengths):
            tokens,mask=self.encoder(torch.from_numpy(window[None]),torch.tensor([length]))
            glob,local,valid=file_statistics(self.parent,tokens,mask)
            pieces['global'].append(glob);pieces['local'].append(local[valid])
            original.append(self.parent(tokens,mask)[:,0])
        result={'parent':float(lme(torch.cat(original),2.).sigmoid())}
        for name,head in self.heads.items():
            result[name]=float(lme(head(torch.cat(pieces[name])),self.config['temperature']).sigmoid())
        if not all(np.isfinite(v) and 0<=v<=1 for v in result.values()):
            raise ValueError('invalid File probability')
        return result
