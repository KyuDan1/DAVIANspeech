"""Use the remaining authorized slot for a current-pipeline Music diagnostic."""
import datetime
import functools
import getpass
import json
from pathlib import Path

from build_v18_music_probe import sha256_file


def main():
    from dacon_submit_api import dacon_submit_api as api
    import requests
    root=Path(__file__).resolve().parents[1]
    archive=root/'current_music_probe_v84.zip'
    manifest=json.loads(archive.with_suffix('.report.json').read_text())
    assert manifest['status']=='complete_archive_validation' and manifest['all_crc_verified']
    assert sha256_file(archive)==manifest['sha256']
    ledger=root/'reports/api_submissions_v83/current_music_probe_v84.zip.json'
    if ledger.exists():
        raise RuntimeError('Prior attempt exists; do not automatically retry')
    token=getpass.getpass('DACON token: ')
    response=requests.post('https://app.dacon.io/api/v1/code-submission/validate',
        data={'cptId':'236749','teamName':'DAVIANspeech','apiToken':token},timeout=30)
    response.raise_for_status()
    status=response.json()
    quota=int(status.get('quota',0))
    print(json.dumps({'quota':quota}),flush=True)
    if quota<=0 or archive.stat().st_size>int(status['upload_filesize_limit']):
        raise RuntimeError('Quota or size check failed; not submitted')
    memo='Diagnostic only: submitted v83 pipeline, only final MUSIC_FAKE_PROB=0.5; other four outputs unchanged. Identifies current Music EER and, with Voice probe, File EER.'
    record=dict(archive=archive.name,sha256=manifest['sha256'],competition='236749',team='DAVIANspeech',
                started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),status='attempt_started',quota_before=quota,memo=memo)
    ledger.write_text(json.dumps(record,indent=2))
    api.tqdm=functools.partial(api.tqdm,mininterval=10)
    result=api.post_code_submission_file(str(archive),token,'236749','DAVIANspeech',memo)
    record.update(status='accepted' if result.get('isSubmitted') else 'not_confirmed',result=result,
                  finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
    ledger.write_text(json.dumps(record,indent=2))
    print(json.dumps(result,ensure_ascii=False),flush=True)
    if not result.get('isSubmitted'):
        raise RuntimeError('Not confirmed; no automatic retry')


if __name__=='__main__':
    main()
