"""Matched RR-vs-FR/RF/FF File diagnostics, using actual anchor re-inference."""
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
    from src.evaluate_diagnostic import official_eer, PREDICTION_COLUMNS
    base_path=ROOT/'reports/exact_anchor_v74/codec_recomputed/predictions.csv'
    base_report=json.loads((base_path.parent/'report.json').read_text())
    candidate_path=ROOT/'reports/component_composition_v73/equal_voice_pair_eat_music_predictions.csv'
    assert base_report['status']=='complete' and sha(base_path)==base_report['predictions_sha256']
    assert sha(candidate_path)=='d13b0d9818d6d583da6c8cf4cfdc136e6e27d5af81a4e01624e8666ff015b183'
    base=pd.read_csv(base_path)
    candidate=pd.read_csv(candidate_path,low_memory=False)
    frame=candidate[candidate.DATASET.eq('codec_mixed_dev_v4')].copy()
    keys=['DATASET','ID']
    assert len(base)==len(frame)==600 and not base.duplicated(keys).any() and not frame.duplicated(keys).any()
    indexed=base.set_index(keys).loc[list(map(tuple,frame[keys].to_numpy()))]
    for column in PREDICTION_COLUMNS:
        frame['anchor_'+column]=indexed[column].to_numpy()
    assert (frame.VOICE_PRESENT.eq(1)&frame.MUSIC_PRESENT.eq(1)).all()
    frame['CASE']=np.char.add(np.where(frame.VOICE_FAKE.eq(1),'F','R'),np.where(frame.MUSIC_FAKE.eq(1),'F','R'))
    out=ROOT/'reports/file_failure_v85'
    out.mkdir(exist_ok=False)
    frozen=dict(inputs_sha256={str(p):sha(p) for p in [base_path,candidate_path]},
                code_sha256=sha(Path(__file__)),scope='Previously exposed codec-mixed development; not private-test composition inference')
    (out/'frozen.json').write_text(json.dumps(frozen,indent=2))
    results=[]
    groups=[('ALL','ALL',frame)]
    for axis in ['CHANNEL','MIX_MODE']:
        groups.extend((axis,str(value),block) for value,block in frame.groupby(axis))
    for axis,group,block in groups:
        for positive in ['FR','RF','FF']:
            pair=block[block.CASE.isin(['RR',positive])]
            for model,column in [('actual_anchor','anchor_FILE_FAKE_PROB'),('v73_composed','FILE_FAKE_PROB')]:
                real=pair[pair.CASE.eq('RR')]
                fake=pair[pair.CASE.eq(positive)]
                results.append(dict(axis=axis,group=group,comparison='RR_vs_'+positive,model=model,
                     real_n=len(real),fake_n=len(fake),eer=official_eer(pair.FILE_FAKE,pair[column]),
                     real_fpr_at_05=float(real[column].ge(.5).mean()),
                     fake_fnr_at_05=float(fake[column].lt(.5).mean()),
                     fake_score_median=float(fake[column].median())))
    pd.DataFrame(results).to_csv(out/'metrics.csv',index=False)
    report=dict(status='complete_matched_File_diagnostic',rows=600,results=results,
                label_order='Voice,Music: FR=fake Voice + real Music; RF=real Voice + fake Music',
                threshold_05_tuned=False,private_test_breakdown_known=False,
                caveat='Reused source components and codec copies are correlated; no independent-sample confidence claim')
    (out/'report.json').write_text(json.dumps(report,indent=2))
    print(json.dumps([r for r in results if r['axis']=='ALL']),flush=True)


if __name__=='__main__':
    main()
