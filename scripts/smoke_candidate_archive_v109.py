"""Extract and run a candidate's actual entrypoint in the grader-core environment."""
import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import zipfile
import zlib


ROOT = Path(__file__).resolve().parents[1]
COLUMNS = ['ID', 'FILE_FAKE_PROB', 'VOICE_FAKE_PROB', 'MUSIC_FAKE_PROB',
           'VOICE_PRESENT_PROB', 'MUSIC_PRESENT_PROB']
RUNNER = '''import json, os, pathlib, resource, runpy, sys, time
def network_guard(event, args):
    if event == "socket.connect" and isinstance(args[1], tuple):
        raise RuntimeError("Network access attempted during offline inference")
    if event == "socket.sendto" and isinstance(args[-1], tuple):
        raise RuntimeError("Network datagram attempted during offline inference")
sys.addaudithook(network_guard)
import torch
torch.set_num_threads(6)
torch.cuda.reset_peak_memory_stats()
start=time.monotonic()
sys.argv=["script.py"]
runpy.run_path("script.py", run_name="__main__")
stats=dict(seconds=time.monotonic()-start, peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),
           peak_cuda_reserved_bytes=torch.cuda.max_memory_reserved(),
           max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
pathlib.Path("smoke_runtime.json").write_text(json.dumps(stats,indent=2))
print("SMOKE_RUNTIME "+json.dumps(stats),flush=True)
'''


def sha(path):
    digest=hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(8*1024*1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--python',type=Path,default=Path(sys.executable))
    parser.add_argument('--reuse-stage',action='store_true')
    args=parser.parse_args()
    archive=args.archive.resolve(); stage=args.output.resolve()
    if stage.exists() and not args.reuse_stage:
        raise FileExistsError(stage)
    if (stage/'report.json').exists():
        raise FileExistsError('Completed smoke already exists')
    stage.mkdir(parents=True,exist_ok=args.reuse_stage)
    with zipfile.ZipFile(archive) as handle:
        names=handle.namelist()
        assert len(set(names)) == len(names)
        assert {Path(name).parts[0] for name in names} == {'script.py','model','requirements.txt'}
        assert not any(name.startswith('/') or '..' in Path(name).parts for name in names)
        assert sum(info.file_size for info in handle.infolist()) < 32_000_000_000
        if args.reuse_stage:
            for info in handle.infolist():
                if info.is_dir():continue
                path=stage/info.filename
                if path.stat().st_size != info.file_size:
                    raise ValueError(f'Staged member size mismatch: {info.filename}')
                crc=0
                with path.open('rb') as stream:
                    for chunk in iter(lambda:stream.read(8*1024*1024),b''):
                        crc=zlib.crc32(chunk,crc)
                if crc != info.CRC:
                    raise ValueError(f'Staged member CRC mismatch: {info.filename}')
        else:
            handle.extractall(stage)  # CRC is checked on every extracted member.
    print(json.dumps({'archive':archive.name,'phase':'extracted_crc_passed'}),flush=True)
    fixture=ROOT/'reports/mixfake_joint_music_moe_v101/full_package_smoke/data'
    shutil.copytree(fixture,stage/'data',dirs_exist_ok=args.reuse_stage)
    import numpy as np
    import soundfile as sf
    # Extend the known RR/FR/RF fixtures to cover stereo, a full minute, and MP3.
    source,sr=sf.read(stage/'data/test/package_smoke_2.wav',dtype='float32')
    long=np.tile(source,int(np.ceil(60*sr/len(source))))[:60*sr]
    stereo=np.stack((long, .8*long),axis=1)
    sf.write(stage/'data/test/package_smoke_3.flac',stereo,sr,subtype='PCM_16')
    sys.path.insert(0,str(ROOT/'src'))
    from full_coverage_wpt import resolve_ffmpeg
    subprocess.run([str(resolve_ffmpeg()),'-v','error','-i',
                    str(stage/'data/test/package_smoke_1.wav'),'-codec:a','libmp3lame',
                    '-b:a','64k',str(stage/'data/test/package_smoke_4.mp3')],check=True)
    with (stage/'data/sample_submission.csv').open('w',newline='') as stream:
        writer=csv.writer(stream);writer.writerow(COLUMNS)
        for i in range(5):writer.writerow([f'package_smoke_{i}']+[.5]*5)
    before={str(p.relative_to(stage/'data')):sha(p) for p in (stage/'data').rglob('*') if p.is_file()}
    for p in (stage/'data').rglob('*'):
        if p.is_file():p.chmod(0o444)
    environment=dict(os.environ,HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',
                     HF_DATASETS_OFFLINE='1',PYTHONDONTWRITEBYTECODE='1',
                     OMP_NUM_THREADS='6',MKL_NUM_THREADS='6',OPENBLAS_NUM_THREADS='6')
    start=time.monotonic()
    affinity=sorted(os.sched_getaffinity(0))[:6]
    def constrain_cpu():os.sched_setaffinity(0,affinity)
    with (stage/'execution.log').open('w') as log:
        result=subprocess.run([str(args.python),'-c',RUNNER],cwd=stage,env=environment,
                              stdout=log,stderr=subprocess.STDOUT,preexec_fn=constrain_cpu)
    if result.returncode:
        raise RuntimeError(f'Packaged entrypoint failed: {stage / "execution.log"}')
    with (stage/'output/submission.csv').open(newline='') as stream:
        reader=csv.DictReader(stream);assert reader.fieldnames == COLUMNS
        rows=list(reader)
    assert [r['ID'] for r in rows] == [f'package_smoke_{i}' for i in range(5)]
    assert all(math.isfinite(float(row[c])) and 0 <= float(row[c]) <= 1
               for row in rows for c in COLUMNS[1:])
    after={str(p.relative_to(stage/'data')):sha(p) for p in (stage/'data').rglob('*') if p.is_file()}
    assert before == after,'input data changed'
    report=dict(status='passed',archive=archive.name,archive_sha256=sha(archive),
                rows=len(rows),formats=['wav','mp3','flac'],stereo_and_60s=True,
                crc_verified=True,network_connections_blocked=True,input_unchanged=True,
                python=str(args.python),seconds=time.monotonic()-start,
                runtime=json.loads((stage/'smoke_runtime.json').read_text()),
                limitation='B200 with 6 CPU affinity; full L4 1200-file runtime not measured')
    (stage/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report),flush=True)


if __name__=='__main__':main()
