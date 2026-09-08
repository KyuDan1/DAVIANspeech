# Exact v47 + packaged v48 fixed-expert audit

> **Selection note:** the standalone `0.075/0.01` comparator below came from a
> locked-response sweep and is not a valid selection. The corrected
> development-only standalone policy is Voice `0.05`, File `0.0`; see
> `../v47_standalone_devselect/report.md`.

## Decision

The packaged v48 policy is **aggregate-stronger but not controlled-cell safe**. It improves ADS over exact v47 on every locked bank and beats the standalone Voice 0.075 / File 0.01 candidate on factorial and phone, tying it on YuE. However, it introduces one regression among the 57 controlled contrasts: factorial partial-overlap, real-voice/fake-music File EER rises from `0.24` to `0.28`.

For a conservative first hidden submission, use the leakage-free development-selected **standalone Voice 0.05 / File 0.0** policy documented in the linked audit. Use packaged v48 as the higher-upside alternative only if accepting that known File failure-mode regression. The 0.075/0.01 results below remain a historical locked-surface comparator, not a valid selection.

## Exact policy audited

The frozen fixed expert and outer residual were reconstructed exactly as packaged:

```text
fixed Voice = sigmoid(0.10 * logit(Unified Voice) + 0.90 * logit(WPT Voice))
fixed File  = sigmoid(0.20 * logit(Unified File)  + 0.80 * logit(WPT File))
v48 Voice   = sigmoid(0.95 * logit(v47 Voice) + 0.05 * logit(fixed Voice))
v48 File    = sigmoid(0.95 * logit(v47 File)  + 0.05 * logit(fixed File))
v48 Music   = v47 Music
```

The fixed-expert path uses `1e-5` clipping and rounds the outer outputs to 10 decimals, matching `wpt_spectra_inference.py`. Exact v47 was rebuilt with its deployed `1e-6` clipping and per-stage rounding: EAT patch 5%, component query File 2.5% / Music 5%, then File component noisy-OR 30%. Its aggregate EERs and ADS match `v47_locked_audit.csv` within `5e-11` on every bank.

## Locked aggregates

| bank | method | File EER | Voice EER | Music EER | ADS | delta vs v47 | delta vs standalone |
|---|---|---:|---:|---:|---:|---:|---:|
| factorial | exact v47 | 0.232364 | 0.222857 | 0.211429 | 0.775818 | - | -0.002286 |
| factorial | standalone 0.075/0.01 | 0.232364 | 0.211429 | 0.211429 | 0.778104 | +0.002286 | - |
| factorial | **v48 fixed 0.05** | **0.224727** | 0.217143 | 0.211429 | **0.780779** | **+0.004961** | **+0.002675** |
| phone | exact v47 | 0.178286 | 0.167500 | 0.170000 | 0.826357 | - | -0.002714 |
| phone | standalone 0.075/0.01 | 0.175857 | **0.160000** | 0.170000 | 0.829071 | +0.002714 | - |
| phone | **v48 fixed 0.05** | **0.165857** | 0.162500 | 0.170000 | **0.833571** | **+0.007214** | **+0.004500** |
| YuE | exact v47 | 0.122283 | 0.102083 | 0.177419 | 0.865216 | - | -0.003750 |
| YuE | standalone 0.075/0.01 | 0.122283 | 0.083333 | 0.177419 | 0.868966 | +0.003750 | - |
| YuE | v48 fixed 0.05 | 0.122283 | 0.083333 | 0.177419 | 0.868966 | +0.003750 | 0.000000 |

Across the three banks, v48 fixed has mean ADS **0.827772**, a mean gain of **+0.005308** over v47 and **+0.002392** over standalone. Its worst bank gain over v47 is **+0.003750**. The standalone comparator has mean ADS **0.825381**, mean gain **+0.002917**, and worst bank gain **+0.002286**.

## All 57 controlled contrasts

| comparison | improved | tied | regressed | mean EER delta | worst EER delta |
|---|---:|---:|---:|---:|---:|
| standalone vs v47 | 9 | 48 | 0 | -0.009298 | 0.000000 |
| v48 fixed vs v47 | 14 | 42 | **1** | -0.010351 | **+0.040000** |
| v48 fixed vs standalone | 8 | 44 | 5 | -0.001053 | +0.125000 |

The sole v48 regression relative to v47 is:

- factorial `partial_overlap|file_rr_vs_voice_real+music_fake`, File: `0.24 -> 0.28` (`+0.04`).

The other 14 changed v48 cells are improvements. Notable File gains include:

- factorial concurrent fake-voice/fake-music: `0.24 -> 0.20`;
- factorial sequential real-voice/fake-music: `0.36 -> 0.28`;
- phone speech-only File: `0.045 -> 0.035` and music-only File: `0.135 -> 0.125`;
- phone simultaneous fake-voice/music-real: `0.22 -> 0.21`, real-voice/music-fake: `0.29 -> 0.25`, and both fake: `0.17 -> 0.16`;
- YuE concurrent real-voice/fake-music: `0.50 -> 0.375`.

Relative to standalone, v48 gains eight File contrasts but gives back five standalone advantages: two factorial Voice gains are smaller, phone simultaneous/music-fake Voice returns from `0.19` to the v47 value `0.20`, YuE concurrent/music-fake Voice returns from `0.125` to the v47 value `0.25`, and the factorial File regression appears. Thus the higher aggregate result is chiefly a File trade rather than uniform cell dominance.

Every controlled row is recorded in `controlled_contrasts.csv`; `aggregate.csv` contains the full official metrics, including counts, AUC, CPS, and SCORE.

## Alignment and frozen evidence

All joins used `(dataset, canonical ID)` rather than row position. Coverage is complete for the 400 factorial holdout, 1,200 codec-specific phone IDs, and 124 YuE IDs, with no missing or duplicate keys across v18 reconstruction, patch, query, WPT batch-8, and Unified predictions. `alignment.csv` records these checks.

The inner expert reconstructed from `seed06_submit_batch8` and `joint_router_banks` differs from `fixed_moe_export/fixed_predictions.csv` by at most `4.05e-8`, consistent with the previously verified batch-size numerical variation. The packaged checkpoint provenance is seed `20260906`, epoch 7. `predictions.csv` preserves the exact per-row v47, standalone, and v48 Voice/File values used here.
