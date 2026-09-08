#!/usr/bin/env python3
"""Matched-ID Voice EER and fixed-threshold errors, without selecting weights."""
import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def main():
    import numpy as np
    import pandas as pd
    from sklearn.metrics import roc_curve
    from src.lossless_weights_v77 import sha256
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',action='append',required=True,help='NAME=CSV')
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    records=[];hashes={};reference=None
    for specification in args.input:
        name,path=specification.split('=',1)
        hashes[path]=sha256(path)
        frame=pd.read_csv(path,dtype={'ID':str,'DATASET':str},low_memory=False)
        frame=frame.loc[frame.VOICE_PRESENT.eq(1)].sort_values(['DATASET','ID']).reset_index(drop=True)
        identity=frame[['DATASET','ID','VOICE_FAKE','MUSIC_PRESENT','MUSIC_FAKE']]
        if identity[['DATASET','ID']].duplicated().any():raise ValueError('duplicate IDs')
        if reference is None:reference=identity
        else:pd.testing.assert_frame_equal(identity,reference,check_dtype=False)
        if not frame.VOICE_FAKE_PROB.between(0,1).all():raise ValueError('invalid probabilities')
        frame['VOICE_CONTEXT']=np.where(frame.MUSIC_PRESENT.eq(0),'pure_voice',
            np.where(frame.MUSIC_FAKE.eq(1),'fake_music_RF_vs_FF','real_music_RR_vs_FR'))
        frame['COMPONENT_CASE']=[('F' if v else 'R')+('F' if m else 'R') if present else
            ('fake_voice_only' if v else 'real_voice_only') for v,m,present in
            zip(frame.VOICE_FAKE,frame.MUSIC_FAKE,frame.MUSIC_PRESENT)]
        slices=[('ALL','ALL',frame)]
        for axis in ['VOICE_CONTEXT','COMPONENT_CASE','DATASET','CHANNEL_V57','LAYOUT_V57']:
            if axis in frame:
                slices.extend((axis,str(group),part) for group,part in frame.groupby(axis,dropna=False))
        for axis,group,part in slices:
            eer=None
            if part.VOICE_FAKE.nunique()==2:
                fp,tp,_=roc_curve(part.VOICE_FAKE,part.VOICE_FAKE_PROB,pos_label=1,drop_intermediate=False)
                i=np.argmin(abs(fp-(1-tp)));eer=float((fp[i]+1-tp[i])/2)
            real=part[part.VOICE_FAKE.eq(0)];fake=part[part.VOICE_FAKE.eq(1)]
            records.append(dict(variant=name,axis=axis,group=group,rows=len(part),
                real_rows=len(real),fake_rows=len(fake),VOICE_EER=eer,
                FPR_at_fixed_half=float(real.VOICE_FAKE_PROB.ge(.5).mean()) if len(real) else None,
                FNR_at_fixed_half=float(fake.VOICE_FAKE_PROB.lt(.5).mean()) if len(fake) else None))
        if sha256(path)!=hashes[path]:raise ValueError('input changed during audit')
    args.output.mkdir(parents=True)
    pd.DataFrame(records).to_csv(args.output/'slices.csv',index=False)
    report=dict(status='complete',rows=len(reference),matched_ids_and_labels=True,
        source_sha256=hashes,script_sha256=sha256(__file__),
        overall=[r for r in records if r['axis']=='ALL'],
        contexts=[r for r in records if r['axis']=='VOICE_CONTEXT'],
        limitation='Development diagnostics, no statistical significance or official-score claim; RF vocal truth may be unverified.',
        thresholds_selected=False,model_or_submission_modified=False)
    (args.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
