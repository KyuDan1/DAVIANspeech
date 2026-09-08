import pytest
from scripts.inventory_ctrsvdd_v75 import parse_protocol


def test_labels_follow_explicit_authenticity_not_filename():
    rows = parse_protocol('m4singer singer_1 real_name - A01 deepfake\n'
                          'm4singer singer_2 fake_name - - bonafide\n', 'train')
    assert [r['VOICE_FAKE'] for r in rows] == [1, 0]


@pytest.mark.parametrize('text', ['', 'bad row', 'a b c - - unknown',
    'a b c - A01 bonafide', 'a b c - - deepfake',
    'a b c - - bonafide\na b c - - bonafide'])
def test_ambiguous_or_inconsistent_metadata_rejected(text):
    with pytest.raises(ValueError):
        parse_protocol(text, 'train')
