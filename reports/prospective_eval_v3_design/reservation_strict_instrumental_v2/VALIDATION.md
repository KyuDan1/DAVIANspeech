# Prospective mixed-phone v3 instrumental v2 validation

Validated: 2026-09-04 UTC

## Result

The completed build at
`data/eval/prospective_mixed_phone_v3_instrumental_v2` is valid and remains
unscored. The original 67 integrity checks plus two checks for the semantic
exclusion input give an effective result of **69/69 PASS**.

No authenticity-detector prediction was read, no detector was run on this
bank, and no ADS/CPS/EER or other authenticity score was computed. The build
provenance independently records `detector_inference: false` and
`score_computed: false`.

## Verified invariants

- 240 unique bases and 1,200 unique FLAC files are present.
- Each of the 12 `mix mode x RR/RF/FR/FF` cells has exactly 20 bases and 100
  rendered files.
- Every base has exactly the five declared paired channels: clean, G.711
  mu-law, G.722 wideband, Opus narrowband 8 kbit/s, and G.711-to-Opus
  transcoding. Each channel has 240 files.
- Every output is mono 16 kHz FLAC and is 8.000--19.901625 seconds long.
- `FILE_FAKE` equals the logical OR of `VOICE_FAKE` and `MUSIC_FAKE` on every
  row. Both components are present in every mixture.
- All 240 voice sources and all 240 canonical music groups are unique. ASV
  speaker/attack provenance is complete.
- Fake music contains no SONICS source. The five allowed FakeMusicCaps
  generators each contribute exactly 24 sources, all 120 sources pass the
  frozen MusicCaps caption/aspect rule, and all three acoustic replacements
  pass the unchanged PANNs/Demucs gates.
- All 480 selected source-file hashes and all 1,200 rendered-file hashes were
  recomputed from disk and match their manifests.
- Multi-root resolution used exactly 477 files from the original instrumental
  source root and three files from the isolated replacement root. No selected
  archive member was missing or multiply resolved.
- The submission template contains the exact 1,200 IDs and only neutral `0.5`
  placeholder probabilities.

## Hashes

| Artifact | SHA-256 |
| --- | --- |
| v2 reservation | `02ed5aa569078fba08a18155adef668ac6426cf9e9ad1bbd33bcdf4df4a310e1` |
| built truth | `d45c5a3ce0f20496e6d6d4754a94f8b43633cbe5cd1a273e1cf06599313cd388` |
| selected-source hash manifest | `ca549f463c0f0f23fb0fa83aa19c07de755a20bb3837cec4ac79d06f68eb23ec` |
| rendered-audio hash manifest | `544068f66075e0c8f9a31f8b96598c1e10e4d484fbb476c1c59ed45706e16c8a` |
| built provenance JSON | `b2ae36e0253ea344f4ea50156fee3da50183cfc052c90ee54e58901dcf3dfd51` |
| semantic exclusion evidence | `7810fa5167ef871c301532e65b9e6c9c9b5f200481a84a725529aa36cde95f1b` |
| replacement acoustic scores | `a3838c4345fdb265bec7f6da3e5fd6ca877456318f59a70ebdc28b8ffd50a833` |
| builder used at render time | `174368ea417817fcc8233b6fb8b9293d8745040b5dfbbb6431d88fa0dfc48259` |

## Post-build registration race

The reservation froze `configs/data_partitions.yaml` at SHA-256
`bacbc5dc7eb1d9ad09797bf1eb880ddaee554bb6e9cbfb9a4580db316d1bd962`.
The v2 truth completed at `07:52:57 UTC`. At `07:55:08 UTC`, while the
read-only full-file hash validation was running, the config was intentionally
updated to register v2 under `locked_eval` and v1 under `invalid_eval`; its new
SHA-256 became
`e3e58c80840371efe56653a3a7104201bb9608b8f7f414514c1059bb3b72fe8d`.

Consequently, the first long-running validation printed four raw overlap
alerts at its final stage: the check had hashed the old config near the start,
then read the newly registered manifests near the end. These are not evidence
of training/evaluation leakage. They are the expected self-overlap with v2 and
the deliberate 117/120-source inheritance from the never-authenticity-scored,
semantically invalid v1 predecessor.

The identity audit was repeated in memory against all other 45 registered
manifests while excluding only these two post-build registrations. Intersections
were exactly zero for all seven protected views: voice source, voice archive,
voice member, music source, music archive, music member, and canonical music
group. Thus the freeze-time leakage result is PASS, and post-build
self-registration correctly prevents future training use.

The acoustic audit is documented in
`reports/prospective_music_vocal_screen_v1/README.md`; it explicitly states
that the semantic models are not authenticity detectors and their outputs were
not used to estimate fake-detection performance.
