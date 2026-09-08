#!/usr/bin/env python3
"""One fixed equal-logit Voice complement audit; no weight/threshold search."""
import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def main():
    import numpy as np
    import pandas as pd
    from src.lossless_weights_v77 import sha256
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    paths={
        'xlsr':ROOT/'reports/common_encoder_probe/xlsr/mean/development_predictions.csv',
        'spear':ROOT/'reports/common_encoder_probe/spear_independent/mean/development_predictions.csv',
        'wavlm_control':ROOT/'reports/wavlm_voice_lora_v79/full/control_development.csv',
        'wavlm_lora':ROOT/'reports/wavlm_voice_lora_v79/full/adapted_development.csv'}
    hashes={str(path):sha256(path) for path in paths.values()}
    frames={}
    identity=['DATASET','ID','VOICE_FAKE','MUSIC_PRESENT','MUSIC_FAKE']
    for name,path in paths.items():
        frame=pd.read_csv(path,dtype={'ID':str,'DATASET':str},low_memory=False)
        frame=frame.loc[frame.VOICE_PRESENT.eq(1)].sort_values(['DATASET','ID']).reset_index(drop=True)
        if frame[['DATASET','ID']].duplicated().any():raise ValueError('duplicate identity')
        if not frame.VOICE_FAKE_PROB.between(0,1).all():raise ValueError('invalid probabilities')
        if frames:pd.testing.assert_frame_equal(frame[identity],frames['xlsr'][identity],check_dtype=False)
        frames[name]=frame
    args.output.mkdir(parents=True)
    first=np.clip(frames['xlsr'].VOICE_FAKE_PROB.to_numpy(),1e-6,1-1e-6)
    outputs=[]
    for name in ['spear','wavlm_control','wavlm_lora']:
        second=np.clip(frames[name].VOICE_FAKE_PROB.to_numpy(),1e-6,1-1e-6)
        logit=.5*(np.log(first)-np.log1p(-first))+.5*(np.log(second)-np.log1p(-second))
        result=frames['xlsr'].drop(columns=[c for c in frames['xlsr'] if c.endswith('_PROB')]).copy()
        result['VOICE_FAKE_PROB']=1/(1+np.exp(-logit))
        path=args.output/f'equal_xlsr_{name}.csv'
        result.to_csv(path,index=False)
        outputs.append(dict(path=str(path),sha256=sha256(path)))
    if any(sha256(p)!=d for p,d in hashes.items()):raise ValueError('audit input changed')
    report=dict(status='complete_fixed_mixtures',input_sha256=hashes,outputs=outputs,
        script_sha256=sha256(__file__),weight=.5,epsilon=1e-6,
        scope='Voice-only development diagnostic, NOT a submission; no fitting or weight search')
    (args.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report))


if __name__=='__main__':main()
