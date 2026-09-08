# Leakage-free v47 + standalone WPT development selection

## Corrected recommendation

Select **Voice 0.05 / File 0.0**, with Music unchanged. This is the best candidate in the requested bounded grid after requiring both channel safety and controlled-contrast safety on the only non-locked development bank with an exact reconstructible v47 anchor.

```text
logit(Voice_out) = 0.95 * logit(v47 Voice) + 0.05 * logit(WPT Voice)
File_out          = v47 File
Music_out         = v47 Music
```

The earlier Voice 0.075 / File 0.01 recommendation is withdrawn because it was selected by looking across the three locked banks. Its locked result remains useful only as an observed comparison, not as selection evidence.

## Development coverage

Patch, component-query, and WPT prediction CSVs cover all seven model-development banks (3,800 rows), but exact v47 requires a complete frozen exact-v18 anchor chain before applying patch/query/noisy-OR. That chain is available only for `factorial_eval_1200_v2_dev` (400 rows).

| development bank | rows | patch/query/WPT | exact v47 | used to select |
|---|---:|---:|---:|---:|
| factorial dev | 400 | complete | yes | **yes** |
| external mixed | 400 | complete | no | no |
| MixFake music | 1,600 | complete | no | no |
| source-disjoint mixed equal | 200 | complete | no | no |
| source-disjoint mixed | 200 | complete | no | no |
| source-disjoint music | 400 | complete | no | no |
| telephone mixed | 600 | complete | no | no |

The six incomplete banks were not approximated with a different anchor and their standalone WPT scores were not treated as v47 residual results. `coverage.csv` gives the exact inventory. Consequently, the mean-plus-worst-domain score is based on one valid domain and degenerates to that domain's ADS; this is materially weaker evidence than a seven-domain selection.

## Selection protocol and result

I swept Voice and File independently over `0, 0.01, 0.025, 0.05, 0.075, 0.10` in logit space, keeping Music exact. The corrected filter and ranking use development rows only:

1. reject any candidate that lowers ADS on any of the six channel subgroups;
2. reject any candidate that worsens any of 23 factorial controlled task contrasts;
3. maximize `0.5 * (mean domain ADS + worst domain ADS)`;
4. break ties using the worst channel ADS.

Only four of 36 candidates pass both subgroup gates, and all have File weight zero:

| Voice | File | dev ADS | ADS delta | worst channel ADS | controlled gains/regressions |
|---:|---:|---:|---:|---:|---:|
| **0.050** | **0.000** | **0.765481** | **+0.003429** | **0.715924** | **2 / 0** |
| 0.010 | 0.000 | 0.763195 | +0.001143 | 0.715924 | 0 / 0 |
| 0.025 | 0.000 | 0.763195 | +0.001143 | 0.715924 | 0 / 0 |
| 0.000 | 0.000 | 0.762052 | 0.000000 | 0.715924 | 0 / 0 |

Every nonzero File weight causes exactly one controlled dev regression at every Voice setting, even though many improve aggregate ADS. Therefore File must remain zero under the stated safety rule. The full grid is in `dev_sweep.csv` and the dev-only selection record is in `selection.json`.

At Voice 0.05 / File 0.0, factorial-dev Voice EER improves from `0.285714` to `0.268571`; File EER stays `0.238182` and Music EER stays `0.205714`. Four channel ADS values tie v47, while `ogg_48k` improves by `+0.013563` and `telephone_flac` by `+0.006897`; none regress. The two controlled gains are Voice EER on concurrent/music-real (`0.32 -> 0.28`) and partial-overlap/music-fake (`0.36 -> 0.32`). See `dev_channels_selected.csv` and `dev_controlled_selected.csv`.

## Locked outcome, excluded from corrected selection

The already-computed locked tables show that the development-selected policy transfers positively to all three banks:

| locked bank | v47 ADS | Voice 0.05 / File 0 ADS | delta | File EER | Voice EER |
|---|---:|---:|---:|---:|---:|
| factorial holdout | 0.775818 | 0.776961 | +0.001143 | 0.232364 | 0.217143 |
| phone factorial | 0.826357 | 0.827857 | +0.001500 | 0.178286 | 0.160000 |
| YuE | 0.865216 | 0.868966 | +0.003750 | 0.122283 | 0.083333 |

Mean locked ADS gain is **+0.002131** and the minimum bank gain is **+0.001143**. Six of 57 locked controlled contrasts improve and none regress.

For context only, the invalidly locked-selected 0.075/0.01 setting had mean gain `+0.002917`, minimum gain `+0.002286`, and 9/57 gains with no regression. Those stronger numbers cannot justify choosing it because they were the selection target. `locked_outcome_observed.csv` and `locked_controlled_observed.csv` retain both outcomes with their selection basis labeled explicitly.

Because the locked response surface had already been inspected in the superseded audit, this correction cannot retroactively claim a pristine blind evaluation. The corrected weight is nevertheless the deterministic output of the stated dev-only rule: locked values are neither filter inputs nor ranking inputs. A future candidate comparison should preserve blind isolation from the outset.

## Final status

Voice 0.05 / File 0 is the only defensible standalone recommendation from currently reconstructible non-locked evidence. It is a **provisional one-domain selection**, not evidence of seven-domain robustness. A stronger decision requires frozen exact-v47 predictions (or the missing exact-v18 chain) for the other six development banks, followed by the same mean-plus-worst-domain and subgroup-safe sweep without revisiting locked weights.
