# Prospective mixed-phone v2: frozen v47/v48 result

Evaluation date: 2026-09-04

This bank was reserved, built, semantics-screened, registered as `locked_eval`,
and hashed before either prediction was scored.  The scorer has no sweep,
threshold, routing, or weight-selection interface.  After this single scoring
event the bank was moved to `retrospective_diagnostic` and must not be used for
future checkpoint or fusion selection.

## Overall result

| model | File EER | Voice EER | Music EER | ADS |
|---|---:|---:|---:|---:|
| v47 | 0.28333 | 0.25500 | 0.24667 | 0.73333 |
| v48, WPT Voice 5% | 0.28333 | **0.24500** | 0.24667 | **0.73533** |

The independent v47 ADS is only `0.00301` below the official v18 ADS
`0.736341`, so this bank is much better aligned to the leaderboard than the
previous repeatedly inspected local banks.  The frozen WPT Voice residual
transfers in the intended direction (`+0.00200` ADS) but is much too small to
close the gap to the leaderboard target ADS `0.821160`.

All 1,200 rows contain both components, so Presence AUC, CPS, and Total are
mathematically undefined.  The scorer reports `NaN` and `CPS_DEFINED=false`
rather than inventing a surrogate.

## Channel result

| channel | v47 File | v47 Voice | v47 Music | v47 ADS | v48 ADS |
|---|---:|---:|---:|---:|---:|
| clean | 0.15000 | 0.16667 | 0.10833 | **0.85917** | 0.86083 |
| G.722 wideband | 0.16667 | 0.15000 | 0.14167 | **0.84417** | 0.84583 |
| G.711 mu-law | 0.28333 | 0.19167 | 0.23333 | 0.75000 | 0.75000 |
| Opus narrowband 8 kHz | 0.36667 | 0.37500 | 0.34167 | 0.63917 | 0.64583 |
| G.711 then Opus | 0.40278 | 0.42500 | 0.40000 | 0.59361 | 0.59528 |

Clean and G.722 already exceed the target ADS.  Narrowband Opus and repeated
telephone transcoding destroy the ranking of all three authenticity outputs;
this is the dominant gap, not a hard phone-router threshold.

## Layout interaction

The table below shows v47 ADS within each paired channel and layout.

| channel | concurrent | partial overlap | sequential |
|---|---:|---:|---:|
| clean | 0.867 | 0.845 | 0.860 |
| G.722 | 0.867 | 0.840 | 0.823 |
| G.711 | 0.790 | 0.738 | 0.825 |
| Opus-NB | **0.515** | 0.695 | 0.728 |
| G.711 then Opus | **0.487** | 0.623 | 0.674 |

The catastrophic cell is concurrent mixing after Opus or repeated transcoding:
File/Voice/Music EER are `0.500/0.425/0.500` for Opus-NB and
`0.550/0.475/0.475` after G.711-to-Opus.  Sequential clips are materially
easier because at least one local window contains a less-masked component.
This directly motivates preserving ordered XLS-R windows and training
component-query residuals with clean/codec pairs instead of averaging all
windows or routing an entire file to one phone expert.

## Frozen artifacts

- evaluation fingerprint:
  `7b50454cd3606f7af75cb2a1372cfb9b1f65fa83b61ba88e39b8962b3c3d81f0`
- truth SHA-256:
  `d45c5a3ce0f20496e6d6d4754a94f8b43633cbe5cd1a273e1cf06599313cd388`
- v47 prediction SHA-256:
  `1efe13c41fd4814e0961e405cc47fc07551ec1b051a4020492e1939b51ce983e`
- v48 prediction SHA-256:
  `dbaecae25fede872dfc5bd0a0e656457cc0804d3d86d0f017dfc3a3b9ff8ac5a`

Detailed official-EER outputs are in `overall.csv`, `channels.csv`,
`cell_contrasts.csv`, `paired_deltas.csv`, and `paired_rank_flips.csv`.
