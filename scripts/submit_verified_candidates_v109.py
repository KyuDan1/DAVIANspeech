"""Submit validated, explicitly authorized candidates with an attempt ledger."""
import argparse
import datetime
import functools
import getpass
import hashlib
import json
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path)
    parser.add_argument('--quota-only', action='store_true')
    args = parser.parse_args()
    from importlib.metadata import version
    if version('dacon_submit_api') != '0.1.2':
        raise RuntimeError('Expected supplied DACON SDK 0.1.2')
    from dacon_submit_api import dacon_submit_api as api
    import requests
    token = os.environ.get('DACON_API_TOKEN') or getpass.getpass('DACON token: ')

    def quota():
        response = requests.post(
            'https://app.dacon.io/api/v1/code-submission/validate',
            data={'cptId': '236749', 'teamName': 'DAVIANspeech', 'apiToken': token},
            timeout=30,
        )
        response.raise_for_status()
        value = response.json()
        print(json.dumps({key: value.get(key) for key in (
            'quota', 'upload_filesize_limit', 'detail', 'message'
        )}), flush=True)
        return value

    current = quota()
    if args.quota_only:
        return
    if not args.plan:
        parser.error('--plan is required for submission')
    plan = json.loads(args.plan.read_text())
    jobs = plan['jobs']
    if not 1 <= len(jobs) <= 3 or int(current.get('quota', 0)) < len(jobs):
        raise RuntimeError('Insufficient live quota for authorized plan')
    ledger = ROOT / 'reports/api_candidates_20260907'
    ledger.mkdir(parents=True, exist_ok=True)
    for job in jobs:
        path = ROOT / job['archive']
        if path.name != job['archive'] or len(path.name) > 30:
            raise ValueError('Archive name must be a basename of at most 30 characters')
        if (ledger / (path.name + '.json')).exists():
            raise RuntimeError('Prior attempt exists; inspect it before any retry')
        report = json.loads((ROOT / job['smoke_report']).read_text())
        if report['status'] != 'passed' or report['archive_sha256'] != job['sha256']:
            raise RuntimeError('Exact packaged-entrypoint smoke is missing')
        if digest(path) != job['sha256']:
            raise RuntimeError('Archive hash differs from verified plan')
        if path.stat().st_size > int(current['upload_filesize_limit']):
            raise RuntimeError('Archive exceeds live upload size limit')
    api.tqdm = functools.partial(api.tqdm, mininterval=20)
    for job in jobs:
        current = quota()
        if int(current.get('quota', 0)) <= 0:
            raise RuntimeError('Quota consumed externally; stopped before next upload')
        path = ROOT / job['archive']
        record = dict(job, status='attempt_started', competition='236749',
                      team='DAVIANspeech', quota_before=int(current['quota']),
                      started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
        target = ledger / (path.name + '.json')
        with target.open('x') as stream:
            json.dump(record, stream, indent=2)
        print(json.dumps({'upload_start': path.name}), flush=True)
        result = api.post_code_submission_file(
            str(path), token, '236749', 'DAVIANspeech', job['memo']
        )
        record.update(status='accepted' if result.get('isSubmitted') else 'not_confirmed',
                      result=result, finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
        target.write_text(json.dumps(record, indent=2) + '\n')
        print(json.dumps({'archive': path.name, 'result': result}), flush=True)
        if not result.get('isSubmitted'):
            raise RuntimeError('Submission unconfirmed; no automatic retry')


if __name__ == '__main__':
    main()
