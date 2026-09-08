"""Read-only v86 long File slices by duration/channel/position."""
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.check_file_local_v86 import sha


def main():
    import pandas as pd
    from src.evaluate_diagnostic import official_eer
    source=ROOT/'reports/file_local_readout_v86/long_exposed'
    output=ROOT/'reports/file_local_readout_v86/long_slices_v90'
    if output.exists():raise FileExistsError(output)
    report=json.loads((source/'report.json').read_text())
    assert report['status']=='complete_exposed_diagnostic'
    frozen=json.loads((source/'frozen.json').read_text())
    parts=[]; paths=[Path(__file__),source/'report.json',source/'frozen.json']
    for shard in range(4):
        path=source/f'shard_{shard}'/'predictions.csv'
        run=json.loads((path.parent/'report.json').read_text())
        assert run['status']=='complete' and run['predictions_sha256']==sha(path)
        parts.append(pd.read_csv(path));paths.extend([path,path.parent/'report.json'])
    predictions=pd.concat(parts,ignore_index=True)
    assert len(predictions)==2880 and not predictions.ID.duplicated().any()
    anchor_path=ROOT/'reports/long_voice_v61/v73_one_shot/comparison/anchor_predictions.csv'
    paths.append(anchor_path)
    anchor=pd.read_csv(anchor_path)[['ID','FILE_FAKE_PROB']].rename(columns={'FILE_FAKE_PROB':'anchor'})
    results=[]
    for bank,spec in frozen['banks'].items():
        truth_path=Path(spec['truth']);paths.append(truth_path)
        truth=pd.read_csv(truth_path)
        frame=truth.merge(predictions[predictions.BANK.eq(bank)],on='ID',validate='one_to_one').merge(anchor,on='ID',validate='one_to_one')
        assert len(frame)==spec['rows']
        groups=[('ALL','ALL',frame)]
        for axis in ['DURATION','CHANNEL','POSITION','RECORDING']:
            groups.extend((axis,str(value),part) for value,part in frame.groupby(axis))
        groups.extend(('DURATION_CHANNEL',f'{d}:{c}',part) for (d,c),part in frame.groupby(['DURATION','CHANNEL']))
        for axis,group,part in groups:
            for model in ['anchor','parent','global','local']:
                results.append(dict(bank=bank,axis=axis,group=group,model=model,rows=len(part),
                    real_n=int(part.FILE_FAKE.eq(0).sum()),fake_n=int(part.FILE_FAKE.eq(1).sum()),
                    FILE_EER=official_eer(part.FILE_FAKE,part[model])))
    hashes={str(p):sha(p) for p in paths}
    output.mkdir(parents=True)
    pd.DataFrame(results).to_csv(output/'metrics.csv',index=False)
    assert all(sha(p)==digest for p,digest in hashes.items())
    result=dict(status='complete_exposed_long_slices',results=results,
                automatic_submission_allowed=False,
                limitation='Previously exposed, source-reused synthetic voice-only diagnostics; no threshold/weight fitting.')
    (output/'report.json').write_text(json.dumps(result,indent=2))
    (output/'frozen.json').write_text(json.dumps(dict(artifacts_sha256=hashes),indent=2))
    print(json.dumps([r for r in results if r['axis']=='DURATION']),flush=True)


if __name__=='__main__':main()
