#!/usr/bin/env python3
"""Fixed v80 Voice diagnostic on PREVIOUSLY EXPOSED long banks, never a new blind test."""
import argparse
import json
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
OLD=ROOT/'reports/long_voice_v61/v73_one_shot'
MODEL=ROOT/'reports/local_voice_v80/full'
EQUIVALENCE=ROOT/'reports/local_voice_v80/inference_equivalence'
NAMES=['global_parent','file_only','dense']


def validate(frozen):
    from src.lossless_weights_v77 import sha256
    for path,digest in frozen['artifacts_sha256'].items():
        if sha256(path)!=digest:raise ValueError(f'changed diagnostic artifact: {path}')


def prepare(output):
    import pandas as pd
    from src.lossless_weights_v77 import sha256
    if output.exists():raise FileExistsError(output)
    completed=json.loads((MODEL/'report.json').read_text())
    check=json.loads((EQUIVALENCE/'report.json').read_text())
    if completed['status']!='complete' or check['status']!='complete_equivalence':
        raise ValueError('completed full training and native correspondence required')
    check_frozen=json.loads((EQUIVALENCE/'frozen.json').read_text())
    if sha256(EQUIVALENCE/'frozen.json')!=check['frozen_sha256']:
        raise ValueError('correspondence record changed')
    validate(check_frozen)
    old=json.loads((OLD/'frozen.json').read_text())
    comparison=json.loads((OLD/'comparison/report.json').read_text())
    if comparison['status']!='complete' or comparison['frozen_sha256']!=sha256(OLD/'frozen.json'):
        raise ValueError('prior bank diagnostic must be complete, unchanged, and explicitly exposed')
    anchor_parts=[]
    for shard in range(old['shards']):
        path=OLD/'anchor'/f'shard_{shard}'/'output/submission.csv'
        if comparison['prediction_sha256'].get(str(path))!=sha256(path):
            raise ValueError('previous exact-anchor shard changed')
        anchor_parts.append(pd.read_csv(path,dtype={'ID':str})[['ID','VOICE_FAKE_PROB']])
    previous=pd.concat(anchor_parts).sort_values('ID').reset_index(drop=True)
    combined=pd.read_csv(OLD/'comparison/anchor_predictions.csv',dtype={'ID':str})[['ID','VOICE_FAKE_PROB']]
    combined=combined.sort_values('ID').reset_index(drop=True)
    pd.testing.assert_frame_equal(previous,combined,rtol=0,atol=2e-15)
    sources=[Path(__file__),ROOT/'src/local_voice_inference_v80.py',MODEL/'report.json',MODEL/'frozen.json',
        MODEL/'file_only.pt',MODEL/'dense.pt',EQUIVALENCE/'report.json',EQUIVALENCE/'frozen.json',
        OLD/'frozen.json',OLD/'comparison/report.json',OLD/'comparison/anchor_predictions.csv']
    sources.extend(Path(record['truth']) for record in old['banks'].values())
    hashes={str(p.resolve()):sha256(p) for p in sources}
    frozen=dict(artifacts_sha256=hashes,banks=old['banks'],inputs=old['inputs'],shards=4,
        models=NAMES,scope='PREVIOUSLY EXPOSED source-disjoint synthetic long Voice diagnostic; NOT fresh blind validation',
        no_weight_or_threshold_selection=True,automatic_submission_allowed=False)
    if len(frozen['inputs'])!=2880:raise ValueError('expected all fixed + uniform files')
    output.mkdir(parents=True)
    (output/'frozen.json').write_text(json.dumps(frozen,indent=2)+'\n')
    print(json.dumps(dict(status='prepared_exposed_diagnostic',rows=2880,shards=4)),flush=True)


def worker(output,shard):
    import csv
    import torch
    from src.lossless_weights_v77 import sha256
    from src.local_voice_inference_v80 import LocalVoicePredictor
    from src.pipeline import load_audio
    frozen=json.loads((output/'frozen.json').read_text());validate(frozen)
    if shard not in range(frozen['shards']):raise ValueError('invalid shard')
    stage=output/f'shard_{shard}'
    if stage.exists():raise FileExistsError(stage)
    torch.set_num_threads(2)
    predictor=LocalVoicePredictor(MODEL,ROOT)
    stage.mkdir()
    started=time.monotonic();rows=frozen['inputs'][shard::frozen['shards']]
    prediction=stage/'predictions.csv'
    with prediction.open('x') as stream:
        writer=csv.DictWriter(stream,fieldnames=['ID',*NAMES]);writer.writeheader()
        for index,row in enumerate(rows):
            if sha256(row['PATH'])!=row['SHA256']:raise ValueError('protected payload changed')
            values=predictor.predict(load_audio(Path(row['PATH'])))
            if sha256(row['PATH'])!=row['SHA256']:raise ValueError('protected payload changed during prediction')
            writer.writerow(dict(ID=row['ID'],**values));stream.flush()
            if index%100==0:
                print(json.dumps(dict(shard=shard,files=index,seconds=time.monotonic()-started)),flush=True)
    validate(frozen)
    report=dict(status='complete',rows=len(rows),seconds=time.monotonic()-started,
        peak_cuda_mib=torch.cuda.max_memory_allocated()/2**20,
        frozen_sha256=sha256(output/'frozen.json'),prediction_sha256=sha256(prediction))
    (stage/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(shard=shard,**report)),flush=True)


def score(output):
    import numpy as np
    import pandas as pd
    from sklearn.metrics import roc_curve
    from src.lossless_weights_v77 import sha256
    frozen=json.loads((output/'frozen.json').read_text());validate(frozen)
    if (output/'report.json').exists():raise FileExistsError('diagnostic already scored')
    parts=[];prediction_hashes={}
    for shard in range(frozen['shards']):
        stage=output/f'shard_{shard}'
        report=json.loads((stage/'report.json').read_text())
        path=stage/'predictions.csv'
        if report['status']!='complete' or report['frozen_sha256']!=sha256(output/'frozen.json') or report['prediction_sha256']!=sha256(path):
            raise ValueError('incomplete or changed shard')
        frame=pd.read_csv(path,dtype={'ID':str})
        if frame.ID.duplicated().any() or set(frame.ID)!={r['ID'] for r in frozen['inputs'][shard::frozen['shards']]}:
            raise ValueError('shard identity mismatch')
        parts.append(frame);prediction_hashes[str(path)]=sha256(path)
    predictions=pd.concat(parts,ignore_index=True)
    probabilities=predictions[NAMES].to_numpy()
    if not np.isfinite(probabilities).all() or (probabilities<0).any() or (probabilities>1).any():
        raise ValueError('invalid predictions')
    anchor=pd.read_csv(OLD/'comparison/anchor_predictions.csv',dtype={'ID':str})[['ID','VOICE_FAKE_PROB']]
    anchor=anchor.rename(columns={'VOICE_FAKE_PROB':'actual_anchor_voice'})
    records=[]
    for bank,record in frozen['banks'].items():
        truth=pd.read_csv(record['truth'],dtype={'ID':str,'GROUP_ID':str})
        frame=truth.merge(predictions,on='ID',validate='one_to_one').merge(anchor,on='ID',validate='one_to_one')
        if len(frame)!=record['rows'] or not frame.VOICE_PRESENT.eq(1).all():raise ValueError('bank population changed')
        for axis in ['ALL','DURATION','CHANNEL','POSITION']:
            groups=[('ALL',frame)] if axis=='ALL' else frame.groupby(axis,dropna=False)
            for group,part in groups:
                for name in [*NAMES,'actual_anchor_voice']:
                    if part.VOICE_FAKE.nunique()!=2:raise ValueError('both Voice classes required')
                    fp,tp,_=roc_curve(part.VOICE_FAKE,part[name],pos_label=1,drop_intermediate=False)
                    index=np.argmin(abs(fp-(1-tp)))
                    records.append(dict(bank=bank,axis=axis,group=str(group),model=name,rows=len(part),
                        VOICE_EER=float((fp[index]+1-tp[index])/2)))
    table=pd.DataFrame(records);table.to_csv(output/'slices.csv',index=False)
    report=dict(status='complete_exposed_diagnostic',overall=table[table.axis.eq('ALL')].to_dict('records'),
        prediction_sha256=prediction_hashes,frozen_sha256=sha256(output/'frozen.json'),
        scope=frozen['scope'],automatic_submission_allowed=False,
        limitation='Voice only, no File/Music/CPS/official Score claim; exposure prevents treating this as fresh confirmation')
    (output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=['prepare','worker','score'])
    parser.add_argument('--output',type=Path,default=ROOT/'reports/local_voice_v80/long_exposed_diagnostic')
    parser.add_argument('--shard',type=int)
    args=parser.parse_args()
    if args.mode=='prepare':prepare(args.output)
    elif args.mode=='worker':worker(args.output,args.shard)
    else:score(args.output)
