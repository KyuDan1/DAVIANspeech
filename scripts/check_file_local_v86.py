"""Check native File readouts against saved development and reversed ordering."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()


def main():
    import numpy as np
    import pandas as pd
    import torch
    from src.file_local_inference_v86 import FileLocalPredictor
    from src.pipeline import load_audio
    p=argparse.ArgumentParser()
    p.add_argument('--run',type=Path,default=ROOT/'reports/file_local_readout_v86/full')
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    frozen=json.loads((args.run/'frozen.json').read_text())
    for path,digest in frozen['artifacts_sha256'].items():
        if sha(path)!=digest:raise ValueError('training artifact changed')
    frames={name:pd.read_csv(args.run/f'{name}_development.csv',low_memory=False) for name in ['parent','global','local']}
    for f in frames.values():pd.testing.assert_frame_equal(f[['ID','DATASET']],frames['parent'][['ID','DATASET']])
    indices=np.unique(np.rint(np.linspace(0,len(frames['parent'])-1,80)).astype(int))
    sources=[Path(__file__),ROOT/'src/file_local_inference_v86.py',args.run/'report.json',args.run/'frozen.json',
             args.run/'global.pt',args.run/'local.pt',*[args.run/f'{n}_development.csv' for n in frames]]
    hashes={str(p):sha(p) for p in sources}
    args.output.mkdir(parents=True)
    (args.output/'frozen.json').write_text(json.dumps(dict(artifacts_sha256=hashes,indices=indices.tolist(),tolerance=2e-6),indent=2))
    torch.set_num_threads(2)
    model=FileLocalPredictor(args.run,ROOT)
    started=time.monotonic();observed={};maximum=0.
    for order in [indices,indices[::-1]]:
        for index in order:
            row=frames['parent'].iloc[index]
            values=model(load_audio(Path(row.PATH)))
            if index in observed:
                if values!=observed[index]:raise ValueError('file order changes output')
            else:
                observed[index]=values
                for name,value in values.items():
                    difference=abs(value-float(frames[name].iloc[index].FILE_FAKE_PROB))
                    maximum=max(maximum,difference)
                    if difference>2e-6:raise ValueError('native/training mismatch')
    if any(sha(p)!=digest for p,digest in hashes.items()):raise ValueError('verification artifact changed')
    report=dict(status='complete_native_equivalence',files=len(indices),calls=2*len(indices),
                maximum_probability_difference=maximum,reverse_order_bit_exact=True,
                seconds=time.monotonic()-started,scope='implementation correspondence, not accuracy or L4 benchmark')
    (args.output/'report.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report),flush=True)


if __name__=='__main__':main()
