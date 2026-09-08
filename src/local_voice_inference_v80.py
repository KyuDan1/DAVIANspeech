"""File-independent native v80 inference, Voice scores only; no fitting."""
import json
from pathlib import Path
import numpy as np
import torch
from .common_encoder_probe import CommonTokenHead,FrozenEncoderTokens,complete_windows
from .local_voice_v80 import voice_projection,local_statistics,temporal_geometry,LocalVoiceHead,file_logit
from .lossless_weights_v77 import sha256


class LocalVoicePredictor:
    def __init__(self,run,root):
        self.run,self.root=Path(run),Path(root)
        report=json.loads((self.run/'report.json').read_text())
        frozen=json.loads((self.run/'frozen.json').read_text())
        if report['status']!='complete' or report['frozen_sha256']!=sha256(self.run/'frozen.json'):
            raise ValueError('completed unchanged full v80 training required')
        for path,digest in frozen['code_and_weights_sha256'].items():
            if sha256(path)!=digest:raise ValueError('frozen v80 artifact changed')
        self.config=frozen['config']
        self.encoder=FrozenEncoderTokens('xlsr',self.root/self.config['encoder'])
        self.parent=CommonTokenHead(1920,width=96,pooling='mean').cuda()
        self.parent.load_state_dict(torch.load(self.root/self.config['parent_head'],map_location='cpu',weights_only=False)['state_dict'])
        self.parent.eval().requires_grad_(False)
        self.heads={}
        for name in ['file_only','dense']:
            checkpoint=torch.load(self.run/f'{name}.pt',map_location='cpu',weights_only=False)
            if (checkpoint['smoke'] or checkpoint['variant']!=name or checkpoint['config']!=self.config
                    or checkpoint['parent_sha256']!=sha256(self.root/self.config['parent_head'])):
                raise ValueError('head does not match frozen full training')
            head=LocalVoiceHead(self.parent.output_weight[1],self.parent.output_bias[1]).cuda()
            head.load_state_dict(checkpoint['state_dict'],strict=True)
            self.heads[name]=head.eval().requires_grad_(False)

    @torch.no_grad()
    def predict(self,audio):
        windows,lengths,starts=complete_windows(audio)
        parts,masks,global_logits=[],[],[]
        for window,length in zip(windows,lengths):
            tokens,mask=self.encoder(torch.from_numpy(window[None]),torch.tensor([length]))
            values=voice_projection(self.parent,tokens,mask)
            part,valid=local_statistics(values,mask,self.config['kernel'],self.config['stride'])
            parts.append(part);masks.append(valid)
            global_logits.append(self.parent(tokens,mask)[:,1])
        _,ownership=temporal_geometry(starts,lengths,tokens=tokens.shape[2],stride=self.config['stride'])
        valid=torch.cat(masks)&torch.from_numpy(ownership).cuda()
        features=torch.cat(parts)
        global_logits=torch.cat(global_logits)
        global_score=(torch.logsumexp(2*global_logits,0)-np.log(len(global_logits)))/2
        output={'global_parent':float(global_score.sigmoid())}
        for name,head in self.heads.items():
            output[name]=float(file_logit(head(features),valid,self.config['temperature']).sigmoid())
        if not all(np.isfinite(p) and 0<=p<=1 for p in output.values()):
            raise ValueError('invalid Voice output')
        return output
