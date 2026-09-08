#!/usr/bin/env python3
"""Official CC-BY VocalSet archive, quarantined; never register or train data.

No extraction or model evaluation. Interrupted transfers keep a resumable
partial file. Completion requires official size/MD5 and every ZIP member CRC.
"""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
import time
import urllib.request
import zipfile

RECORD='https://zenodo.org/api/records/1442513'
NAME='VocalSet11.zip'
EXPECTED_SIZE=2077243579
EXPECTED_MD5='0c09396242f946e7111ad7d8fc649b81'


def archive_record(metadata):
    if metadata['id']!=1442513 or metadata['metadata']['license']['id']!='cc-by-4.0':
        raise ValueError('official record or license changed; manual review required')
    records=[f for f in metadata['files'] if f['key']==NAME]
    if len(records)!=1:raise ValueError('unique original archive required')
    record=records[0]
    if record['size']!=EXPECTED_SIZE or record['checksum']!='md5:'+EXPECTED_MD5:
        raise ValueError('official archive changed; do not silently replace it')
    if record['links']['self']!=RECORD+'/files/'+NAME+'/content':
        raise ValueError('unexpected archive origin')
    return record


def safe_member(info):
    path=PurePosixPath(info.filename)
    if path.is_absolute() or '..' in path.parts or '\\' in info.filename or ':' in info.filename:
        raise ValueError('unsafe ZIP member path')
    if stat.S_ISLNK(info.external_attr>>16):raise ValueError('ZIP symlink not accepted')
    if info.flag_bits&1:raise ValueError('encrypted ZIP not accepted')
    return dict(name=info.filename,bytes=info.file_size,compressed_bytes=info.compress_size,crc32=info.CRC)


def stream_download(url, destination, expected_size):
    offset=destination.stat().st_size if destination.exists() else 0
    if not 0<=offset<=expected_size:raise ValueError('partial size exceeds official archive')
    if offset==expected_size:return
    headers={'User-Agent':'DAVIANspeech-research/1.0','Accept-Encoding':'identity'}
    if offset:headers['Range']=f'bytes={offset}-'
    request=urllib.request.Request(url,headers=headers)
    with urllib.request.urlopen(request,timeout=30) as response:
        if offset:
            wanted=f'bytes {offset}-{expected_size-1}/{expected_size}'
            if response.status!=206 or response.headers.get('Content-Range')!=wanted:
                raise ValueError('server did not honor exact partial range; existing partial kept')
        elif response.status!=200:
            raise ValueError('unexpected initial download status')
        header=response.headers.get('Content-Length')
        if header is not None and int(header)!=expected_size-offset:
            raise ValueError('unexpected HTTP payload size')
        next_progress=offset+128*2**20
        with destination.open('ab' if offset else 'xb') as stream:
            while True:
                chunk=response.read(4*2**20)
                if not chunk:break
                if offset+len(chunk)>expected_size:raise ValueError('download exceeds official size')
                stream.write(chunk);offset+=len(chunk)
                if offset>=next_progress:
                    print(json.dumps(dict(stage='download',bytes=offset,total=expected_size)),flush=True)
                    next_progress=offset+128*2**20
    if offset!=expected_size:raise ValueError('incomplete transfer; partial file retained')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--resume',action='store_true')
    args=parser.parse_args()
    if args.output.exists() and not args.resume:raise FileExistsError(args.output)
    if args.resume and not args.output.is_dir():raise FileNotFoundError(args.output)
    if (args.output/'report.json').exists():raise FileExistsError('completed inventory already exists')
    with urllib.request.urlopen(urllib.request.Request(RECORD,headers={'User-Agent':'DAVIANspeech-research/1.0'}),timeout=30) as response:
        metadata=json.load(response)
    record=archive_record(metadata)
    args.output.mkdir(parents=True,exist_ok=args.resume)
    meta_path=args.output/'zenodo_record.json'
    if meta_path.exists():
        if archive_record(json.loads(meta_path.read_text()))!=record:
            raise ValueError('archive metadata changed since partial download')
    else:meta_path.write_text(json.dumps(metadata,indent=2)+'\n')
    partial=args.output/(NAME+'.part')
    required=EXPECTED_SIZE-(partial.stat().st_size if partial.exists() else 0)
    if shutil.disk_usage(args.output).free<required+2**30:raise ValueError('insufficient free disk')
    started=time.monotonic()
    stream_download(record['links']['self'],partial,EXPECTED_SIZE)
    print(json.dumps(dict(stage='downloaded_verifying_size_hash_crc')),flush=True)
    md5,sha=hashlib.md5(),hashlib.sha256()
    with partial.open('rb') as stream:
        for chunk in iter(lambda:stream.read(8*2**20),b''):md5.update(chunk);sha.update(chunk)
    if partial.stat().st_size!=EXPECTED_SIZE or md5.hexdigest()!=EXPECTED_MD5:
        raise ValueError('official archive checksum mismatch; invalid partial retained')
    with zipfile.ZipFile(partial) as archive:
        members=[safe_member(info) for info in archive.infolist()]
        if len({m['name'] for m in members})!=len(members):raise ValueError('duplicate ZIP member names')
        if sum(m['bytes'] for m in members)>16*10**9:raise ValueError('unexpected expanded archive size')
        bad=archive.testzip()
        if bad is not None:raise ValueError(f'ZIP CRC failed: {bad}')
        # Read small original README/texts for provenance, no waveform extraction.
        texts={}
        for info in archive.infolist():
            if info.filename.lower().endswith('.txt') and info.file_size<200_000 and '__MACOSX' not in info.filename:
                texts[info.filename]=archive.read(info).decode('utf-8',errors='replace')
    final=args.output/NAME
    if final.exists():raise FileExistsError(final)
    partial.rename(final)
    (args.output/'archive_members.json').write_text(json.dumps(members,indent=2)+'\n')
    (args.output/'original_texts.json').write_text(json.dumps(texts,indent=2)+'\n')
    wavs=[m for m in members if m['name'].lower().endswith('.wav') and '__MACOSX' not in m['name']]
    report=dict(status='complete_verified_archive_quarantined',official_record=RECORD,
        license='CC BY 4.0',archive=str(final.resolve()),bytes=EXPECTED_SIZE,md5=md5.hexdigest(),sha256=sha.hexdigest(),
        all_member_crc_verified=True,wav_members=len(wavs),total_expanded_bytes=sum(m['bytes'] for m in members),
        downloaded_seconds=time.monotonic()-started,script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        attribution=metadata['metadata'].get('creators'),audio_extracted=False,
        training_registered=False,models_trained=False,automatic_submission_allowed=False,
        required_before_training=['Audit existing third-party VocalSet subset use and original-recording overlap',
            'Reserve singer/recording disjoint roles and verify protected-source exclusions',
            'Retain original attribution; do not inherit third-party MIT label',
            'VocalSet supplies real vocals, NOT fake-vocal positive labels'])
    (args.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report),flush=True)


if __name__=='__main__':main()
