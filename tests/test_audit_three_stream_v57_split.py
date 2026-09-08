import pandas as pd

from scripts.audit_three_stream_v57_split import excluded_rows, identity_tokens


def test_identity_tokens_include_both_components_and_archive_members():
    frame = pd.DataFrame([{
        "ID": "row", "VOICE_SOURCE_ID": "voice", "MUSIC_SOURCE_ID": "music",
        "VOICE_ARCHIVE_MEMBER": "v/member", "MUSIC_ARCHIVE_MEMBER": "m/member",
        "FMA_TRACK_ID": "track",
    }])
    tokens, counts = identity_tokens(frame)
    assert {"row", "voice", "music", "v/member", "m/member", "track"} <= tokens
    assert counts["VOICE_ARCHIVE_MEMBER"] == 1


def test_excluded_rows_match_identity_across_columns():
    frame = pd.DataFrame({
        "ID": ["a", "b"], "MUSIC_GROUP": ["safe", "leaked"],
    })
    selected, removed = excluded_rows(frame, {"leaked"})
    assert selected.ID.tolist() == ["a"]
    assert removed == ["b"]
