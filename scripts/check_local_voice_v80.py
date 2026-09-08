#!/usr/bin/env python3
"""Native v80 correspondence and reversed-order checks on development only."""
import argparse
import json
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def main():
    import numpy as np
    import pandas as pd
    import torch
    from src.local_voice_inference_v80 import LocalVoicePredictor
    from src.pipeline import load_audio
    from src.lossless_weights_v77 import sha256
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,default=ROOT/'reports/local_voice_v80/full')
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    frames={name:pd.read_csv(args.run/f'{name}_development.csv',dtype={'ID':str,'DATASET':str},low_memory=False)
            for name in ['global_parent','file_only','dense']}
    reference=frames['global_parent']
    for name,frame in frames.items():
        pd.testing.assert_frame_equal(frame[['DATASET','ID']],reference[['DATASET','ID']])
    indices=np.unique(np.rint(np.linspace(0,len(reference)-1,80)).astype(int))
    frame=reference.iloc[indices]
    sources=[Path(__file__),ROOT/'src/local_voice_inference_v80.py',args.run/'report.json',args.run/'frozen.json',
             args.run/'file_only.pt',args.run/'dense.pt',
             *[args.run/f'{name}_development.csv' for name in frames]]
    hashes={str(p.resolve()):sha256(p) for p in sources}
    torch.set_num_threads(2)
    predictor=LocalVoicePredictor(args.run,ROOT)
    args.output.mkdir(parents=True)
    frozen=dict(artifacts_sha256=hashes,development_indices=indices.tolist(),tolerance=2e-6,
        scope='80 development files; no accuracy or held-eval selection')
    (args.output/'frozen.json').write_text(json.dumps(frozen,indent=2)+'\n')
    observed={};maximum=0.;started=time.monotonic()
    for reverse in [False,True]:
        selected=frame.iloc[::-1] if reverse else frame
        for index,row in selected.iterrows():
            values=predictor.predict(load_audio(Path(row.PATH)))
            if reverse:
                if values!=observed[index]:raise ValueError('file order changes output')
            else:
                observed[index]=values
                for name,value in values.items():
                    difference=abs(value-float(frames[name].loc[index,'VOICE_FAKE_PROB']))
                    maximum=max(maximum,difference)
                    if difference>frozen['tolerance']:raise ValueError('training/native inference mismatch')
    if any(sha256(p)!=d for p,d in hashes.items()):raise ValueError('correspondence input changed')
    report=dict(status='complete_equivalence',files=len(frame),calls=2*len(frame),
        maximum_probability_difference=maximum,reverse_order_bit_exact=True,
        seconds=time.monotonic()-started,peak_cuda_mib=torch.cuda.max_memory_allocated()/2**20,
        frozen_sha256=sha256(args.output/'frozen.json'),automatic_submission_allowed=False)
    (args.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report),flush=True)


if __name__=='__main__':main()
