"""Read-only snapshot of actual v86 TRAIN draws, including semantic labels."""
import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--run',type=Path,default=Path('reports/file_local_readout_v86/full'))
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    with Path('reports/training_inventory_v78/train.csv').open(newline='') as handle:
        train=list(csv.DictReader(handle))
    counters={k:Counter() for k in ['kind','corpus','layout','channel','duration','actual_case','file_label']}
    snapshots=[];rows=0
    for path in sorted(args.run.glob('epoch_*_traces.jsonl')):
        payload=path.read_bytes()
        lines=payload.splitlines(keepends=True)
        if lines and not lines[-1].endswith(b'\n'):lines=lines[:-1]
        complete=b''.join(lines)
        snapshots.append(dict(path=str(path),complete_rows=len(lines),bytes=len(complete),sha256=hashlib.sha256(complete).hexdigest()))
        for line in lines:
            r=json.loads(line);rows+=1
            counters['kind'][r['kind']]+=1
            if r['kind']=='original_train':
                source=train[r['row']]
                assert source['ID']==r['id'] and source['DATASET']==r['dataset']
                counters['corpus'][source['DATASET']]+=1
                counters['file_label'][str(int(float(source['FILE_FAKE'])))]+=1
            elif r['kind']=='synthetic_train':
                metadata=r['intervals']
                voice=bool(metadata['VOICE_PRESENT']);music=bool(metadata['MUSIC_PRESENT'])
                vf=voice and bool(metadata['VOICE_FAKE']);mf=music and bool(metadata['MUSIC_FAKE'])
                assert bool(metadata['FILE_FAKE']) == (vf or mf), 'synthetic File label is inconsistent'
                case=('F' if vf else 'R') if voice else '-'
                case+=('F' if mf else 'R') if music else '-'
                counters['actual_case'][case]+=1
                counters['file_label'][str(int(vf or mf))]+=1
                for key,field in [('layout','layout'),('channel','channel'),('duration','seconds')]:
                    counters[key][str(r[field])]+=1
            else:raise ValueError('unknown trace kind')
    report=dict(status='complete_snapshot_audit',rows=rows,training_finished=(args.run/'report.json').is_file(),
                counts={k:dict(v) for k,v in counters.items()},snapshots=snapshots,
                original_rows_match_train_inventory=True,synthetic_File_OR_labels_valid=True,
                caveat='Snapshot of consumed TRAIN records only; pure-audio cases use actual presence/labels, not unused template case.')
    args.output.mkdir(parents=True)
    (args.output/'report.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report),flush=True)


if __name__=='__main__':main()
