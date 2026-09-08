"""Evaluate the existing learned EAT File head on exposed long diagnostics."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
import time
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
OUT=ROOT/'reports/eat_direct_file_v85'


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()


def main():
    p=argparse.ArgumentParser()
    p.add_argument('mode',choices=['prepare','worker','score'])
    p.add_argument('--shard',type=int,choices=range(4))
    args=p.parse_args()
    if args.mode=='prepare':
        old=ROOT/'reports/long_voice_v61/v73_one_shot/frozen.json'
        protocol=json.loads(old.read_text())
        artifacts=[Path(__file__),old,ROOT/'reports/common_eat_adaptation/full/adapted/head.pt',
                   ROOT/'reports/common_eat_adaptation/full/completed.json',
                   ROOT/'reports/long_voice_v61/v73_one_shot/comparison/anchor_predictions.csv',
                   *[ROOT/'src'/n for n in ['common_eat_adaptation_inference.py','common_eat_adaptation.py','common_encoder_probe.py','eat_large_aasist_inference.py','eat_timm_compat.py']],
                   *[p for p in (ROOT/'models/eat-large-as2m-v56').iterdir() if p.suffix in ['.py','.json','.safetensors']]]
        artifacts.extend(Path(b['truth']) for b in protocol['banks'].values())
        OUT.mkdir(exist_ok=False)
        frozen=dict(inputs=protocol['inputs'],banks=protocol['banks'],shards=4,
                    artifacts_sha256={str(p):sha(p) for p in artifacts},
                    scope='PREVIOUSLY EXPOSED long diagnostics; no training or selection',head='existing learned File head, no component composition')
        (OUT/'frozen.json').write_text(json.dumps(frozen,indent=2))
        print(json.dumps({'status':'prepared','rows':len(frozen['inputs'])}),flush=True)
        return
    frozen=json.loads((OUT/'frozen.json').read_text())
    if args.mode=='worker':
        if args.shard is None:raise ValueError('--shard required')
        import numpy as np
        import torch
        from transformers import PreTrainedModel
        from src.common_eat_adaptation_inference import CommonEatAdaptationPredictor
        import librosa
        for path,digest in frozen['artifacts_sha256'].items():
            if sha(path)!=digest:raise ValueError('artifact changed')
        selected=frozen['inputs'][args.shard::4]
        destination=OUT/f'shard_{args.shard}'
        destination.mkdir(exist_ok=False)
        torch.set_num_threads(2)
        model=CommonEatAdaptationPredictor(ROOT/'reports/common_eat_adaptation/full/adapted/head.pt',ROOT)
        started=time.monotonic()
        with (destination/'predictions.csv').open('w',newline='') as handle:
            writer=csv.DictWriter(handle,fieldnames=['ID','BANK','FILE_FAKE_PROB'])
            writer.writeheader()
            for index,row in enumerate(selected):
                path=Path(row['PATH'])
                if sha(path)!=row['SHA256']:raise ValueError('audio changed')
                audio,_=librosa.load(path,sr=16000,mono=True,dtype=np.float32)
                value=float(model(audio)[0])
                if not 0<=value<=1:raise ValueError('invalid score')
                if sha(path)!=row['SHA256']:raise ValueError('audio changed during read')
                writer.writerow(dict(ID=row['ID'],BANK=row['BANK'],FILE_FAKE_PROB=value))
                if index%100==0: print(json.dumps(dict(shard=args.shard,files=index,seconds=time.monotonic()-started)),flush=True)
        result=dict(status='complete',rows=len(selected),seconds=time.monotonic()-started,
                    predictions_sha256=sha(destination/'predictions.csv'),frozen_sha256=sha(OUT/'frozen.json'))
        (destination/'report.json').write_text(json.dumps(result,indent=2))
        print(json.dumps(result),flush=True)
        return
    import pandas as pd
    from src.evaluate_diagnostic import official_eer
    parts=[]
    for shard in range(4):
        directory=OUT/f'shard_{shard}'
        r=json.loads((directory/'report.json').read_text())
        assert r['status']=='complete' and r['predictions_sha256']==sha(directory/'predictions.csv')
        assert r['frozen_sha256']==sha(OUT/'frozen.json')
        parts.append(pd.read_csv(directory/'predictions.csv'))
    scores=pd.concat(parts,ignore_index=True)
    assert len(scores)==len(frozen['inputs']) and not scores.ID.duplicated().any()
    assert set(scores.ID)=={r['ID'] for r in frozen['inputs']}
    anchor=pd.read_csv(ROOT/'reports/long_voice_v61/v73_one_shot/comparison/anchor_predictions.csv')
    results=[]
    for bank,spec in frozen['banks'].items():
        truth=pd.read_csv(spec['truth'])
        frame=truth.merge(scores[scores.BANK.eq(bank)],on='ID',validate='one_to_one').merge(anchor[['ID','FILE_FAKE_PROB']],on='ID',suffixes=('_eat','_anchor'),validate='one_to_one')
        assert len(frame)==spec['rows']
        for model in ['eat','anchor']:
            results.append(dict(bank=bank,model=model,rows=len(frame),FILE_EER=official_eer(frame.FILE_FAKE,frame['FILE_FAKE_PROB_'+model])))
    (OUT/'report.json').write_text(json.dumps(dict(status='complete_exposed_diagnostic',results=results,automatic_submission_allowed=False),indent=2))
    print(json.dumps(results),flush=True)


if __name__=='__main__':main()
