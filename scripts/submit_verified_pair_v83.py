"""Submit the authorized diagnostic/candidate pair; credentials stay in memory."""
import datetime
import functools
import getpass
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
JOBS = [
    ('best_voice_probe_v82.zip', 'e9e35a18e720b72bf26f33f78e055bc5341a3c2f64762b525aecd0cf02fd58a9',
     'Diagnostic: exact 0.77743 v50+v57 anchor; only VOICE_FAKE_PROB=0.5, other four outputs unchanged. Not a performance candidate.'),
    ('best_eat_music_v83.zip', '92d5f2f3508f7b3d7748f2214a7319c5a9029d826f85779bb8ccb64f18ecf48f',
     'Best anchor + original EAT-large adapted Music only; File/Voice/CPS unchanged. Removed discarded Music residual/XLSR pass; full smoke and CRC verified.'),
]


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(8*1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def main():
    from importlib.metadata import version
    if version('dacon_submit_api') != '0.1.2':
        raise RuntimeError('Expected the supplied DACON SDK version 0.1.2')
    from dacon_submit_api import dacon_submit_api as api
    import requests
    api.tqdm = functools.partial(api.tqdm, mininterval=10)
    ledger = ROOT / 'reports/api_submissions_v83'
    ledger.mkdir(exist_ok=True)
    # Re-running after a timeout is unsafe: an accepted upload might already
    # consume a slot. Refuse even an incomplete previous attempt.
    for filename, digest, _ in JOBS:
        if (ledger/(filename+'.json')).exists():
            raise RuntimeError('Previous attempt exists; inspect before any retry')
        path = ROOT/filename
        if not path.is_file() or sha(path) != digest:
            raise ValueError(f'archive mismatch: {filename}')
    token = getpass.getpass('DACON token: ')
    for filename, digest, memo in JOBS:
        response = requests.post('https://app.dacon.io/api/v1/code-submission/validate',
            data={'cptId':'236749','teamName':'DAVIANspeech','apiToken':token},timeout=30)
        response.raise_for_status()
        status = response.json()
        quota = int(status.get('quota',0))
        print(json.dumps(dict(archive=filename,quota=quota)),flush=True)
        if quota <= 0:
            raise RuntimeError('No quota; stopped without submitting this file')
        path = ROOT/filename
        if path.stat().st_size > int(status['upload_filesize_limit']):
            raise ValueError('live platform upload limit exceeded')
        record = dict(archive=filename,sha256=digest,team='DAVIANspeech',competition='236749',
                      memo=memo,started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                      status='attempt_started',quota_before=quota)
        destination = ledger/(filename+'.json')
        destination.write_text(json.dumps(record,indent=2))
        result = api.post_code_submission_file(str(path),token,'236749','DAVIANspeech',memo)
        record.update(status='accepted' if result.get('isSubmitted') else 'not_confirmed',
                      result=result,finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
        destination.write_text(json.dumps(record,indent=2))
        print(json.dumps(dict(archive=filename,result=result),ensure_ascii=False),flush=True)
        if not result.get('isSubmitted'):
            raise RuntimeError('Submission not confirmed; remaining job stopped, no automatic retry')


if __name__ == '__main__':
    main()
