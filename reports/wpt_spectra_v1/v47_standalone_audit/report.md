# Exact v47 + standalone WPT low-weight audit

> **Superseded selection:** this report mapped the locked response surface and
> incorrectly used it to recommend weights. The leakage-free development-only
> selection is documented in `../v47_standalone_devselect/report.md` and selects
> Voice `0.05`, File `0.0`.

## Recommendation

Fuse the standalone `seed06_submit_batch8` WPT prediction into exact v47 at **Voice 0.075 / File 0.01**, leaving **Music unchanged**:

`logit(p_out) = (1 - w) * logit(p_v47) + w * logit(p_wpt)`

This is the maximin-safe point in the requested grid: it strictly improves ADS on every locked bank, maximizes the smallest bank gain among settings with no controlled hard-cell regression, and improves 9 of 57 controlled contrasts while regressing none. This audit does **not** use the unified/fixed expert.

| locked bank | v47 File EER | fused File EER | v47 Voice EER | fused Voice EER | Music EER | v47 ADS | fused ADS | ADS delta |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| factorial holdout | 0.232364 | 0.232364 | 0.222857 | 0.211429 | 0.211429 | 0.775818 | 0.778104 | +0.002286 |
| phone factorial | 0.178286 | 0.175857 | 0.167500 | 0.160000 | 0.170000 | 0.826357 | 0.829071 | +0.002714 |
| YuE cross-component | 0.122283 | 0.122283 | 0.102083 | 0.083333 | 0.177419 | 0.865216 | 0.868966 | +0.003750 |

Mean ADS rises from **0.822464 to 0.825381** (`+0.002917`); the minimum per-bank gain is **+0.002286**. Music probabilities are unchanged.

## Alignment and exactness

The published v47 locked audit contains aggregate metrics, so I reconstructed its per-row probabilities from exact v18 plus the documented EAT patch, component-query residuals, and File noisy-OR. The reconstructed aggregate metrics equal `v47_locked_audit.csv` on all three banks.

Rows were joined by `(dataset, source_key)` derived from the actual audio filename/ID, never by row order. Factorial uses the 400 locked holdout keys from a 1,200-file audio directory; phone uses all 1,200 codec-specific IDs (not the 300 repeated parent IDs); YuE uses all 124 IDs. Query and WPT coverage is complete with no duplicate or missing keys. See `alignment.csv`.

## Sweep result

The complete independent 6 x 6 Voice/File grid is in `sweep.csv`, with one row per bank, and summarized in `grid_summary.csv`. Of 36 settings, 21 strictly improve all three banks. Six do so with zero regression across the 57 controlled contrasts:

| Voice weight | File weight | mean ADS delta | min bank delta | hard gains / regressions |
|---:|---:|---:|---:|---:|
| 0.050 | 0.000 | +0.002131 | +0.001143 | 6 / 0 |
| 0.050 | 0.010 | +0.002536 | +0.001143 | 7 / 0 |
| 0.075 | 0.000 | +0.002512 | +0.001500 | 8 / 0 |
| **0.075** | **0.010** | **+0.002917** | **+0.002286** | **9 / 0** |
| 0.100 | 0.000 | +0.002726 | +0.001000 | 8 / 0 |
| 0.100 | 0.010 | +0.003131 | +0.002214 | 9 / 0 |

Voice 0.10 / File 0.01 has a slightly larger mean gain, but its worst-bank gain is smaller because phone Voice EER is 0.1625 rather than 0.1600. Therefore 0.075 / 0.01 is the more robust exact choice.

## Controlled hard cells

At the recommended point, the nine strict improvements are:

- factorial Voice: partial-overlap/music-real `0.32 -> 0.20`, sequential/music-real `0.24 -> 0.16`, partial-overlap/music-fake `0.28 -> 0.24`;
- phone Voice: speech-only `0.065 -> 0.055`, simultaneous/music-real `0.24 -> 0.23`, simultaneous/music-fake `0.20 -> 0.19`;
- phone File: simultaneous real-voice/fake-music `0.29 -> 0.28`;
- YuE Voice: concurrent/music-fake `0.25 -> 0.125`, sequential/music-fake `0.125 -> 0.0`.

All other controlled cells are unchanged, including factorial concurrent fake-voice/music-real File EER `0.44`, phone simultaneous fake-voice/music-real `0.22`, and YuE concurrent fake-voice/music-real `0.25`. Full selected-cell evidence is in `hard_cells_selected.csv`.

Every tested File weight at or above 0.025 causes one regression regardless of Voice weight: factorial partial-overlap real-voice/fake-music File EER worsens from `0.24` to `0.28`. That rules out the larger aggregate-gain File settings for the conservative submission.
