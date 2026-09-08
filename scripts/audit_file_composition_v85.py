"""Two fixed File reconstructions from CURRENT component scores; no fitting."""
import hashlib
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    import numpy as np
    import pandas as pd
    from src.evaluate_diagnostic import official_eer
    out=ROOT/'reports/file_composition_v85'
    out.mkdir(exist_ok=False)
    codec_path=ROOT/'reports/exact_anchor_v74/codec_recomputed/predictions.csv'
    codec_report=json.loads((codec_path.parent/'report.json').read_text())
    assert sha(codec_path)==codec_report['predictions_sha256']
    candidate_path=ROOT/'reports/component_composition_v73/equal_voice_pair_eat_music_predictions.csv'
    assert sha(candidate_path)=='d13b0d9818d6d583da6c8cf4cfdc136e6e27d5af81a4e01624e8666ff015b183'
    truth=pd.read_csv(candidate_path,low_memory=False)
    truth=truth[truth.DATASET.eq('codec_mixed_dev_v4')]
    codec=pd.read_csv(codec_path).merge(truth[['ID','FILE_FAKE','VOICE_FAKE','MUSIC_FAKE','CHANNEL','MIX_MODE']],on='ID',validate='one_to_one')
    assert len(codec)==600
    long_path=ROOT/'reports/long_voice_v61/v73_one_shot/comparison/anchor_predictions.csv'
    long_scores=pd.read_csv(long_path)
    paths=[codec_path,candidate_path,long_path]
    banks={'codec_mixed':codec}
    for name,bank in [('long_uniform','long_voice_uniform_v61'),('long_fixed','long_voice_sparse_v61')]:
        path=ROOT/'data/eval'/bank/'truth.csv'
        paths.append(path)
        truth=pd.read_csv(path)
        frame=truth.merge(long_scores,on='ID',validate='one_to_one')
        assert len(frame)==len(truth)
        banks[name]=frame
    frozen=dict(inputs_sha256={str(p):sha(p) for p in paths},code_sha256=sha(Path(__file__)),
                rules=['anchor','presence_weighted_noisy_or','presence_weighted_max'],
                selection_or_fitting=False,scope='Previously exposed diagnostics; no private-test or fresh blind claim')
    (out/'frozen.json').write_text(json.dumps(frozen,indent=2))
    results=[]
    for name,frame in banks.items():
        v=frame.VOICE_FAKE_PROB.to_numpy()*frame.VOICE_PRESENT_PROB.to_numpy()
        m=frame.MUSIC_FAKE_PROB.to_numpy()*frame.MUSIC_PRESENT_PROB.to_numpy()
        rules={'anchor':frame.FILE_FAKE_PROB.to_numpy(),
               'presence_weighted_noisy_or':1-(1-v)*(1-m),
               'presence_weighted_max':np.maximum(v,m)}
        frame=frame.copy()
        for rule,values in rules.items():
            frame[rule]=values
        groups=[('ALL','ALL',frame)]
        for axis in ['CHANNEL','MIX_MODE','DURATION']:
            if axis in frame:
                groups.extend((axis,str(value),block) for value,block in frame.groupby(axis))
        for axis,group,block in groups:
            for rule in rules:
                results.append(dict(bank=name,axis=axis,group=group,rule=rule,rows=len(block),
                                    FILE_EER=official_eer(block.FILE_FAKE,block[rule])))
    pd.DataFrame(results).to_csv(out/'metrics.csv',index=False)
    assert all(sha(Path(p))==digest for p,digest in frozen['inputs_sha256'].items())
    report=dict(status='complete_fixed_composition_diagnostic',results=results,
                probability_calibration_assumed_not_proven=True,automatic_submission_allowed=False,
                caveat='Noisy OR assumes component-event independence; max is an alternative heuristic, neither follows uniquely from File=OR labels.')
    (out/'report.json').write_text(json.dumps(report,indent=2))
    print(json.dumps([r for r in results if r['axis']=='ALL']),flush=True)


if __name__=='__main__':
    main()
