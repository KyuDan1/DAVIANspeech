"""Matched File-only development analysis; never fit thresholds or weights."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    import numpy as np
    import pandas as pd
    from src.evaluate_diagnostic import official_eer
    p=argparse.ArgumentParser()
    p.add_argument('--run',type=Path,default=ROOT/'reports/file_local_readout_v86/full')
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    report_path=args.run/'report.json'
    assert json.loads(report_path.read_text())['status']=='complete'
    names=['parent','global','local']
    paths=[report_path,*[args.run/f'{name}_development.csv' for name in names]]
    frames={name:pd.read_csv(path,low_memory=False) for name,path in zip(names,paths[1:])}
    keys=['DATASET','ID'];labels=['FILE_FAKE','VOICE_FAKE','MUSIC_FAKE','VOICE_PRESENT','MUSIC_PRESENT']
    reference=frames['parent']
    for frame in frames.values():
        assert not frame.duplicated(keys).any()
        pd.testing.assert_frame_equal(frame[keys+labels],reference[keys+labels])
        assert np.isfinite(frame.FILE_FAKE_PROB).all() and frame.FILE_FAKE_PROB.between(0,1).all()
    hashes={str(path):sha(path) for path in paths}
    args.output.mkdir(parents=True)
    (args.output/'frozen.json').write_text(json.dumps(dict(inputs_sha256=hashes,code_sha256=sha(Path(__file__)),
        scope='Previously used development only; no official-score prediction; no selection/fitting'),indent=2))
    results=[]
    for name,frame in frames.items():
        groups=[('ALL','ALL',frame)]
        for axis in ['DATASET','AUDIO_TYPE','CHANNEL','CHANNEL_V57','MIX_MODE','LAYOUT_V57']:
            if axis in frame:
                groups.extend((axis,str(value),block) for value,block in frame.groupby(axis))
        mixed=frame[frame.VOICE_PRESENT.eq(1)&frame.MUSIC_PRESENT.eq(1)].dropna(subset=['VOICE_FAKE','MUSIC_FAKE'])
        rr=mixed.VOICE_FAKE.eq(0)&mixed.MUSIC_FAKE.eq(0)
        for case,v,m in [('FR',1,0),('RF',0,1),('FF',1,1)]:
            groups.append(('FILE_CONTRAST','RR_vs_'+case,mixed[rr|(mixed.VOICE_FAKE.eq(v)&mixed.MUSIC_FAKE.eq(m))]))
        for axis,group,block in groups:
            real=block[block.FILE_FAKE.eq(0)];fake=block[block.FILE_FAKE.eq(1)]
            results.append(dict(model=name,axis=axis,group=group,rows=len(block),real_n=len(real),fake_n=len(fake),
                FILE_EER=official_eer(block.FILE_FAKE,block.FILE_FAKE_PROB),
                FPR_at_05=float(real.FILE_FAKE_PROB.ge(.5).mean()),FNR_at_05=float(fake.FILE_FAKE_PROB.lt(.5).mean())))
    pd.DataFrame(results).to_csv(args.output/'metrics.csv',index=False)
    assert all(sha(Path(path))==digest for path,digest in hashes.items())
    output=dict(status='complete_File_development_slices',results=results,
                official_improvement_verified=False,automatic_submission_allowed=False,
                limitation='Paired source reuse and exposed development; a threshold-0.5 error is not EER.')
    (args.output/'report.json').write_text(json.dumps(output,indent=2))
    print(json.dumps([r for r in results if r['axis'] in ['ALL','AUDIO_TYPE','FILE_CONTRAST']]),flush=True)


if __name__=='__main__':main()
