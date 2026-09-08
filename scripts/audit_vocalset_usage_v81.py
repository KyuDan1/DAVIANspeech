#!/usr/bin/env python3
"""Read-only registry reference audit for existing third-party VocalSet files.

Absence of a textual reference is not decoded-audio duplicate proof and is
not authorization to silently register the original corpus for training.
"""
import argparse
import json
from pathlib import Path
import re
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def main():
    import pandas as pd
    import yaml
    from src.lossless_weights_v77 import sha256
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    config=ROOT/'configs/data_partitions.yaml'
    config_hash=sha256(config)
    roles=yaml.safe_load(config.read_text())
    subset=ROOT/'data/external/vocalset_presence_subset_v1/vocalset'
    files=sorted(subset.glob('*.wav'))
    if len(files)!=60:raise ValueError('existing subset changed; inspect before proceeding')
    terms=['vocalset','squillo',*[p.stem.casefold() for p in files]]
    pattern=re.compile('|'.join(re.escape(t) for t in terms),re.IGNORECASE)
    matches=[];sources={};scanned=[]
    for role,paths in roles.items():
        if not isinstance(paths,list):continue
        for relative in paths:
            if not isinstance(relative,str) or not relative.endswith('.csv'):continue
            path=ROOT/relative
            digest=sha256(path)
            frame=pd.read_csv(path,dtype=str,low_memory=False).fillna('')
            count=0
            for column in frame:
                selected=frame[column].str.contains(pattern,na=False)
                count+=int(selected.sum())
                for _,row in frame[selected].iterrows():
                    matches.append(dict(role=role,path=relative,column=column,id=row.get('ID',''),
                        matched_terms=sorted({m.group(0).casefold() for m in pattern.finditer(row[column])})))
            if sha256(path)!=digest:raise ValueError('registry manifest changed during scan')
            sources[str(path)]=digest
            scanned.append(dict(role=role,path=relative,rows=len(frame),matching_cells=count))
    # The exact rows used in recent training/development, including resolved PATH.
    for name in ['train','development']:
        path=ROOT/f'reports/training_inventory_v78/{name}.csv'
        digest=sha256(path);frame=pd.read_csv(path,dtype=str,low_memory=False).fillna('')
        count=0
        for column in frame:
            selected=frame[column].str.contains(pattern,na=False);count+=int(selected.sum())
            for _,row in frame[selected].iterrows():
                matches.append(dict(role='actual_inventory_'+name,path=str(path),column=column,id=row.get('ID',''),
                    matched_terms=sorted({m.group(0).casefold() for m in pattern.finditer(row[column])})))
        if sha256(path)!=digest:raise ValueError('inventory changed during scan')
        sources[str(path)]=digest;scanned.append(dict(role='actual_inventory_'+name,path=str(path),rows=len(frame),matching_cells=count))
    payloads=[dict(path=str(p),bytes=p.stat().st_size,sha256=sha256(p),claimed_singer=p.stem.split()[0]) for p in files]
    if sha256(config)!=config_hash:raise ValueError('partition roles changed during scan')
    args.output.mkdir(parents=True)
    (args.output/'matches.json').write_text(json.dumps(matches,indent=2)+'\n')
    (args.output/'subset_payloads.json').write_text(json.dumps(payloads,indent=2)+'\n')
    report=dict(status='complete_reference_audit_not_pcm_proof',matching_cells=len(matches),scanned=scanned,
        manifest_sha256=sources,partition_sha256=config_hash,subset_files=len(files),
        claimed_singers=len({p['claimed_singer'] for p in payloads}),
        script_sha256=sha256(__file__),registered_or_trained=False,
        limitations=['No decoded-PCM/content match against original archive or all renamed/cropped derivatives',
            'Third-party attribution and labels remain unverified',
            'No textual match does not prove all historical/unregistered experiments never used this subset'])
    (args.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(status=report['status'],matching_cells=len(matches),manifests=len(scanned),subset_files=len(files))),flush=True)


if __name__=='__main__':main()
