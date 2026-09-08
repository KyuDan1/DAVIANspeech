# Prospective mixed + telephone evaluation v3: provenance audit and design

Date: 2026-09-04

## Decision

The current `phone_factorial_1200_v1` is not a prospective holdout.  It has no
role in `configs/data_partitions.yaml`, all 300 clean parents come from existing
development banks, and it has already appeared in 32 experiment scripts.  Keep
it only as a contaminated channel-mechanics diagnostic; do not use it to select
another weight.

The strongest unused pool already present locally is the **official MixFake
`eval` split**.  Metadata-only inspection found 40,000 speech+music rows.  After
excluding every exact identity in every configured role, including a canonical
SONICS song group, 35,578 rows remain:

| Voice/Music cell | Eligible rows | Unique voice IDs | Unique music groups |
| --- | ---: | ---: | ---: |
| RR | 9,811 | 2,500 | 9,811 |
| RF | 7,938 | 2,494 | 6,372 |
| FR | 9,811 | 2,500 | 9,811 |
| FF | 8,018 | 2,493 | 6,439 |

This is ample for a new source-disjoint bank.  It is **not a fully
generator-disjoint bank**: its fake-music families (AudioLDM2, MusicGen,
MusicLDM, Mustango, Stable Audio Open, Suno and Udio) have all appeared in the
current experiment universe.  The local MixFake protocol also omits the
ASVspoof attack and speaker fields, so the official ASVspoof protocol must be
joined before claiming fake-voice generator or speaker disjointness.

No complete, genuinely generator-OOD voice+music pool is currently usable
locally.  RTCFake would be the best real-communication voice source, but both
local label/pair files are zero bytes and no audio is present.  A new music
generator family must also be acquired or generated after the split is frozen.

## Proposed 1,200-file holdout

Build `data/eval/prospective_mixed_phone_v3` only after all model choices for
the next round are frozen.

- Make 12 cells: RR/RF/FR/FF crossed with `concurrent`,
  `partial_overlap`, and `sequential`.
- Select 20 source-disjoint base mixtures per cell: 240 bases total.
- Render every completed mixture through five paired conditions: `clean`,
  `g711_ulaw`, `g722_wb`, `opus_nb_8k`, and `transcode_g711_opus`.
  This gives exactly 240 x 5 = 1,200 files.
- Use a voice and a canonical music group in only one base mixture.  Reuse is
  allowed only for the five paired channel variants of that base.
- Balance fake music across the seven available generator families.  Join the
  official ASVspoof2019 LA protocol and balance fake voice across eval attacks;
  reject the build if attack/speaker provenance cannot be recovered.
- For concurrent/partial modes, cycle voice-to-music SNR through
  -10/-5/0/+5/+10 dB.  Cycle partial overlap through 25/50/75 percent and both
  orders.  Cycle sequential order and 0/0.2/0.5 s gaps.
- Apply the telephone transform **after** the final mixture.  Do not separate
  components and do not transform them independently before mixing.
- Keep every output between 4 and 60 seconds at 16 kHz.  Label File as the
  logical OR of Voice and Music fake labels.

The clean member is essential: it makes channel damage measurable within the
same content.  Metrics must cluster by base ID; the five variants are not five
independent observations.

## Freeze protocol

1. Import provenance, canonicalize IDs, sample once with a declared seed, and
   write a source reservation manifest **without running a detector**.
2. Record SHA-256 for the source archives, protocol files, reservation
   manifest, builder commit and output truth.
3. Add the future truth path to `locked_eval` before any feature extraction.
   A new role name is unsafe until `src/data_guard.py` is taught to protect it.
4. Store at least `BASE_ID`, `VOICE_SOURCE_ID`, `VOICE_SPEAKER`,
   `VOICE_GENERATOR`, `MUSIC_SOURCE_ID`, `MUSIC_GROUP_ID`, `MUSIC_GENERATOR`,
   `MIX_MODE`, `CHANNEL`, and source split/license in truth.
5. Do not inspect model predictions, distributions or scores until the
   submission candidate and all fusion weights are frozen.  Score once and
   then retire the bank from model selection.

This audit did not open or score prospective audio.  It read only existing
truth/protocol metadata and archive directory/manifest metadata.

## Existing protocol violations and ambiguities

1. **Unprotected phone audit.** `phone_factorial_1200_v1/truth.csv` is absent
   from every role.  Its 1,200 rows are four transforms of 300 parents taken
   from `source_disjoint_mixed_equal_v1`, `source_disjoint_music_v1`, and
   `asvspoof_voice_v1`; the first two are development banks.  It has been used
   repeatedly for selection, so adding it to `locked_eval` now protects future
   training but cannot restore prospective status.
2. **The full factorial manifest crosses roles.** All 1,200 rows of
   `factorial_eval_1200_v2/truth.csv` are declared locked while its exact 400-row
   dev and holdout subsets are declared development and OOD holdout.  Remove
   the full manifest from role lists; keep only the three split manifests in
   their intended roles.
3. **Raw factorial banks are mislabeled as wholly locked.** The complete
   `multigen_voice_v2` and `echoes_fma_paired_v3` manifests contain
   dev/holdout/locked rows but are both declared wholly locked.  Factorial dev
   reuses 85 voice and 98 music IDs from those paths.  Split the raw manifests
   by their `SPLIT` column and register each split separately.
4. **The old `prospective` music split is no longer prospective.** The full
   `source_disjoint_music_v1` manifest, including 200 `prospective` rows, is
   declared development and has been scored by selection scripts.  The locked
   `forensic_call_audit_v1` also reuses 103 identities from this development
   path.  Treat both only as retrospective diagnostics.
5. **Internal dev rows are declared train.** Full manifests for
   `forensic_call_train_v1` and temporal mixed v1/v2 are under `train` even
   though they contain `SPLIT=dev`.  Some trainers filter correctly, but the
   partition contract does not enforce that.  Register `truth_train.csv` and
   `truth_dev.csv` explicitly instead.
6. **Guard coverage is incomplete.** `data_guard.py` checks train against
   protected roles, but not development against locked, and its identity list
   omits `MUSIC_GROUP_ID`, `VOICE_SPEAKER`, `PAIR_GROUP`, and `PARENT_ID`.
   Therefore the guard currently prints PASS despite the cross-role overlaps
   above.

Intentional telephone derivatives under `stress_eval` also share identities
with development sources.  That is valid for paired robustness measurement,
but their scores are not evidence of source-generalization.

## Pool triage

Detailed counts are in `candidate_pools.csv`; concrete overlap evidence is in
`protocol_violations.csv`.

- **Use now, with caveat:** MixFake official eval components.  They are exact
  source-disjoint after filtering, but generator-family-seen.
- **Real-music reserve:** 7,385 unused FMA-small track IDs remain.  They need a
  frozen instrumental/voice screen after reservation.
- **Too small:** only 15 unseen FMA-matched Echoes groups (191 renderings)
  remain after current role exclusions, before voice screening.
- **Do not call holdout:** 2,086 locally available unused SONICS song groups
  all come from SONICS train, all have vocals, and the repository includes a
  SONICS-trained detector.  They are neither music-only nor model-unseen.
- **CPS-only reserve:** 24 Squillo and 60 VocalSet real a-cappella files are
  unused, but they are singing, not a representative voice-phishing call pool.
- **Blocked pending acquisition:** RTCFake metadata exists, but audio and labels
  do not.

