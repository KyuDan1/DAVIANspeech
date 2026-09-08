import zipfile
import pytest

from scripts.run_exact_anchor_v74 import attest_package, sha256, stage_inputs


def test_attestation_checks_actual_package_bytes(tmp_path):
    package = tmp_path / 'package'
    (package / 'model').mkdir(parents=True)
    (package / 'model/head.pt').write_bytes(b'weights')
    (package / 'script.py').write_text('print(1)')
    archive = tmp_path / 'package.zip'
    with zipfile.ZipFile(archive, 'w') as stream:
        stream.write(package / 'script.py', 'script.py')
        stream.write(package / 'model/head.pt', 'model/head.pt')
    report = attest_package(package, archive, sha256(archive))
    assert report['archive_crc_matches_package']
    (package / 'model/head.pt').write_bytes(b'changed')
    with pytest.raises(ValueError):
        attest_package(package, archive, sha256(archive))


def test_stage_does_not_modify_audio_or_package(tmp_path):
    package = tmp_path / 'package'
    (package / 'model').mkdir(parents=True)
    audio = tmp_path / 'source.wav'
    audio.write_bytes(b'audio')
    output = tmp_path / 'stage'
    stage_inputs(output, package, [dict(ID='example', PATH=str(audio))])
    assert (output / 'model').resolve() == package / 'model'
    assert (output / 'data/test/example.wav').resolve() == audio
    assert audio.read_bytes() == b'audio'
    assert not (package / 'data').exists()
    with pytest.raises(FileExistsError):
        stage_inputs(output, package, [dict(ID='example', PATH=str(audio))])
