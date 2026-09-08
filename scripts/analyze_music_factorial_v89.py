"""Matched parent/factorial Music slices; no threshold, weight or model fitting."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.check_file_local_v86 import sha


def main():
    import numpy as np
    import pandas as pd
    from src.evaluate_diagnostic import official_eer
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, default=ROOT/'reports/music_factorial_v89/full')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    assert json.loads((args.run/'report.json').read_text())['status'] == 'complete'
    paths = dict(parent=ROOT/'reports/common_eat_adaptation/full_file_local/development_predictions.csv',
                 factorial_bce=args.run/'factorial_bce_development.csv',
                 factorial_ranked=args.run/'factorial_ranked_development.csv')
    frames = {name: pd.read_csv(path, low_memory=False).sort_values(['DATASET','ID']).reset_index(drop=True)
              for name, path in paths.items()}
    keys = ['DATASET','ID']; labels = ['FILE_FAKE','VOICE_FAKE','MUSIC_FAKE','VOICE_PRESENT','MUSIC_PRESENT']
    for frame in frames.values():
        pd.testing.assert_frame_equal(frame[keys+labels], frames['parent'][keys+labels], check_dtype=False)
        assert np.isfinite(frame.MUSIC_FAKE_PROB).all() and frame.MUSIC_FAKE_PROB.between(0,1).all()
    sources = [Path(__file__), args.run/'report.json', *paths.values()]
    hashes = {str(p): sha(p) for p in sources}
    args.output.mkdir(parents=True)
    (args.output/'frozen.json').write_text(json.dumps(dict(artifacts_sha256=hashes,
        scope='exposed development, fixed-final comparison, no fitting'), indent=2))
    results=[]
    for name, frame in frames.items():
        music=frame[frame.MUSIC_PRESENT.eq(1)&frame.MUSIC_FAKE.isin([0,1])]
        groups=[('ALL','ALL',music)]
        for axis in ['DATASET','AUDIO_TYPE','CHANNEL','CHANNEL_V57','LAYOUT_V57','MIX_MODE']:
            if axis in music:
                groups.extend((axis,str(value),block) for value,block in music.groupby(axis))
        mixed=music[music.VOICE_PRESENT.eq(1)]
        groups.extend(('FOREGROUND_LABEL',label,mixed[mixed.VOICE_FAKE.eq(value)])
                      for label,value in [('VOICE_REAL',0),('VOICE_FAKE',1)])
        for axis,group,block in groups:
            real=block[block.MUSIC_FAKE.eq(0)]; fake=block[block.MUSIC_FAKE.eq(1)]
            results.append(dict(model=name,axis=axis,group=group,rows=len(block),real_n=len(real),fake_n=len(fake),
                MUSIC_EER=official_eer(block.MUSIC_FAKE,block.MUSIC_FAKE_PROB),
                FPR_at_05=float(real.MUSIC_FAKE_PROB.ge(.5).mean()),
                FNR_at_05=float(fake.MUSIC_FAKE_PROB.lt(.5).mean())))
    import pandas as pd
    pd.DataFrame(results).to_csv(args.output/'metrics.csv',index=False)
    assert all(sha(p)==digest for p,digest in hashes.items())
    report=dict(status='complete_development_slices',results=results,
                official_improvement_verified=False,automatic_submission_allowed=False)
    (args.output/'report.json').write_text(json.dumps(report,indent=2))
    print(json.dumps([r for r in results if r['axis'] in ['ALL','AUDIO_TYPE','FOREGROUND_LABEL']]),flush=True)


if __name__=='__main__':main()
