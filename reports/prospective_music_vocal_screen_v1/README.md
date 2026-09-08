# Prospective fake-music vocal semantic screen

## Outcome

The 120 unique fake-music sources in
`reservation_strict_instrumental/reservation.csv` were audited without reading
any XLS-R, EAT, SPEAR, File-fake, Music-fake, or submission score.

- 117/120 pass both acoustic screens.
- 3/120 require review.
- PANNs flags 3; Demucs flags 2, both a subset of the PANNs flags.
- Do not freeze the prospective truth until the three flags are listened to.

## Threshold freeze

Thresholds were fixed from the authorized development music bank
`echoes_fma_paired_v3/truth_dev.csv` before candidate audio was scored.

- PANNs any-voice threshold: `0.20`.
  Rule: ceil the maximum stored development `VOICE_SCREEN` (`0.193009`) to
  `0.01`.
- Demucs vocal/mix energy threshold: `-1.5 dB`.
  Rule: score all 111 development clips, then ceil the maximum (`-1.844733
  dB`) to `0.5 dB`.

The PANNs output is separated into 12 vocal/singing AudioSet labels and nine
speech labels, with the maximum taken across all 4.04-second windows. Demucs
uses the vocal-stem energy relative to the original mixture. A source is
flagged if either frozen threshold is exceeded.

## Flagged sources

| source | generator | PANNs evidence | Demucs vocal/mix | interpretation |
|---|---|---:|---:|---|
| `FM_mustango_QVpsAY9oR_4` | mustango | speech `0.202580` | `-34.6837 dB` | **Reject/replace.** MusicCaps calls it instrumental but also explicitly says “some vocalisation” and background chatter. The metadata regex missed `vocalisation`. |
| `FM_stable_audio_open_rGgvqyHKI4Y` | stable_audio_open | vocal `0.206062` | `-0.2223 dB` | Manual review. Caption says shofar and “no voices”; both models can confuse a monophonic wind instrument with voice/vocal stem. |
| `FM_audioldm2_Qe_KwKVDgoE` | audioldm2 | speech `0.252500` | `-0.0385 dB` | Manual review. Caption says a sustained saxophone note and “no voices”; this is another plausible wind-instrument false positive. |

The first source is a definite semantic-provenance failure even without model
judgment. The latter two must not be automatically relabeled: generated audio
can depart from its caption, but PANNs and Demucs are also particularly
vulnerable to voice-like solo wind timbres.

## Deterministic replacement recheck

All three flags are frozen in `acoustic_semantic_exclusions.csv`, including
their source-audio hashes and the hashes of the original reservation, audit
script, and full acoustic-evidence CSV. With seed `20260904`, the builder's
same-generator stable-rank replacement keeps all other base/voice/layout
assignments and 117/120 fake-music identities unchanged. It replaces only:

| excluded source | deterministic replacement | PANNs any voice | Demucs vocal/mix | result |
|---|---|---:|---:|---|
| `FM_mustango_QVpsAY9oR_4` | `FM_mustango_UXDzDV_f1jw` | `0.069375` | `-46.3646 dB` | pass |
| `FM_stable_audio_open_rGgvqyHKI4Y` | `FM_stable_audio_open_F1X7egd8Us0` | `0.023448` | `-59.1759 dB` | pass |
| `FM_audioldm2_Qe_KwKVDgoE` | `FM_audioldm2_WNmAlKrAPjQ` | `0.025155` | `-48.4147 dB` | pass |

The replacement identities were fixed before scoring. The thresholds were not
retuned: all three pass the same PANNs `0.20` and Demucs `-1.5 dB` rules, for
zero replacement failures. The canonical v2 reservation SHA-256 is
`02ed5aa569078fba08a18155adef668ac6426cf9e9ad1bbd33bcdf4df4a310e1`.
The replacement score CSV SHA-256 is
`a3838c4345fdb265bec7f6da3e5fd6ca877456318f59a70ebdc28b8ffd50a833`;
full provenance is in `replacement_recheck_v2/summary.json`.

## Generator summary

| generator | sources | flags | passes |
|---|---:|---:|---:|
| MusicGen_medium | 24 | 0 | 24 |
| audioldm2 | 24 | 1 | 23 |
| musicldm | 24 | 0 | 24 |
| mustango | 24 | 1 | 23 |
| stable_audio_open | 24 | 1 | 23 |

## Bias and limits

This is a semantic data-quality screen, not model evaluation. The development
reference was itself constructed using PANNs `VOICE_SCREEN <= 0.20`, so its
distribution is right-censored; the threshold is an anomaly envelope, not a
measured sensitivity/specificity operating point. The reservation was already
selected using MusicCaps instrumental/no-vocal text, so the 117/120 pass rate
is conditional on that text selection. PANNs and Demucs are weak, correlated
semantic evidence rather than human truth.

The v2 replacements clear the frozen acoustic gate, but PANNs and Demucs do not
certify vocal absence. Before freezing truth, listen to the three replacements
and a generator-stratified random sample of the 117 retained passes. Do not
adjust either threshold after listening to prospective items.

Exact source scores and hashes are in `source_scores.csv`; calibration values
are in `demucs_development_calibration.csv`; run provenance and limitations are
in `summary.json`. The audit is reproduced by
`scripts/audit_prospective_music_vocals.py`.
