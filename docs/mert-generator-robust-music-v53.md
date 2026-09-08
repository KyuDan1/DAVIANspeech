# MERT generator-robust Music head v53

## Decision

**Reject for submission.**  The model was selected without looking at codec
banks v4--v7, then audited once with a pre-declared `0.025` Music-logit
residual.  Relative to the exact v50 anchor it changes Music EER by
`-0.00333 / 0 / 0 / +0.00556` on v4/v5/v6/v7, respectively (negative is an
improvement).  The mean Music EER is therefore **0.00056 worse**, corresponding
to approximately **-0.00017 ADS** and **-0.00015 total score** if all other
axes are fixed.  The standalone expert is also worse than the anchor on every
retrospective bank.

This is evidence against deploying the current low-capacity temporal MERT
head, not evidence that long-horizon musical evidence is intrinsically
useless.  Its authorized-development and LOGO scores are not predictive enough
of the codec/source shifts represented by the held-out banks.

## What was tested

The detector uses the original mixture; it does not separate voice and music.
It combines two views of the same MERT representation:

- a local tower for shallow-layer, within-window fakeprints; and
- a long tower for deep-layer start/middle/end direction, curvature, and
  dispersion, intended to retain rhythm/harmony-scale evidence.

The input tensor has shape `[3 views, 13 layers, mean/std, 768]`.  A fixed
random projection limits coordinate-specific generator shortcuts, after which
two small MLP towers produce the Music logit.  Training uses source-balanced
BCE, generator GroupDRO, and directed clean-teacher-to-codec-student
consistency.  Exact parent pairs are used when available.

The training set contains 17,760 Music-present examples from:

- `external_mixed_train_v1`
- `mixed_devvoice_train_v1`
- `mixed_fmc_music_train_v1`
- `mixfake_music_train_v1`
- `telephone_mixed_train_v1`
- the `truth_train.csv` partition of `temporal_mixed_train_v2`
- `channel_invariant_factorial_train_v1`

No evaluation row from `temporal_mixed_train_v2` is admitted into training.
Repeated codec children are source-balanced so that one source does not gain
extra loss mass merely by having more channel variants.

## No extra MERT pass

The current package already calls `apply_sofia_mert_fusion` on each original
audio mixture.  `SofiaMertDetector.score_and_statistics_path` returns the
public SOFIA score and temporal tensor from the **same encoder forward**.  The
new head consumes that tensor after all files are scored, so its incremental
MERT forward count is zero.  `test_mert_runtime_reuse.py` replaces the detector
with a sentinel that fails if the old independent `score_path` is called and
asserts exactly one combined call per file.

## Selection protocol

Selection used only the authorized development sets and source-disjoint
leave-one-generator-out (LOGO) folds.  The robust score is

`0.50 * authorized maximin + 0.25 * LOGO mean + 0.25 * LOGO worst`.

| Candidate | Authorized | LOGO mean | LOGO worst | Robust |
|---|---:|---:|---:|---:|
| compressed local | 0.63823 | 0.68425 | 0.48935 | 0.61252 |
| compressed long | 0.62829 | 0.67561 | 0.55457 | 0.62169 |
| **compressed dual** | **0.68909** | 0.71972 | **0.59008** | **0.67199** |
| rich dual | 0.68416 | **0.72260** | 0.57008 | 0.66525 |

The richer layer-preserving feature did not beat the smaller dual summary, so
no further architecture sweep was run.  The chosen compressed-dual checkpoint
was frozen before any v4--v7 result was read.

Authorized development EER for the frozen model:

| Dataset | Music EER |
|---|---:|
| mixfake music dev | 0.20875 |
| external mixed | 0.37500 |
| source-disjoint mixed | 0.19000 |
| source-disjoint mixed equal | 0.15000 |
| source-disjoint music | 0.07500 |
| factorial dev | 0.36571 |
| telephone mixed dev | 0.36333 |

LOGO Music scores were Suno 0.74707, Udio 0.74821, MusicGen 0.66086,
AudioLDM 0.59008, MusicLDM 0.66426, and Mubert 0.90786.  AudioLDM is the
selection bottleneck.

## Frozen artifact

- checkpoint: `reports/mert_generator_robust_v53_dual/mert_generator_robust_music.pt`
- SHA-256: `73d4e398382fd806cf150bde77be8554abbff82e7e816c3bee5e951f0fd46cb9`
- config: `configs/mert_generator_robust_v53_frozen.yaml`
- allowed output change: `MUSIC_FAKE_PROB` only
- fixed fusion: `97.5%` anchor logit + `2.5%` expert logit
- post-audit tuning: forbidden

## One-shot retrospective audit

v4, v5, v6, and v7 were never used for training, checkpoint selection, model
choice, or residual-weight selection.  In particular, v7 is diagnostic only.

| Bank | v50 EER | Expert EER | v50 + .025 EER | EER reduction |
|---|---:|---:|---:|---:|
| codec mixed dev v4 | 0.27333 | 0.33000 | 0.27000 | +0.00333 |
| codec mixed blind v5 | 0.19167 | 0.38333 | 0.19167 | 0.00000 |
| codec mixed blind v6 | 0.25833 | 0.30833 | 0.25833 | 0.00000 |
| codec mixed blind v7 | 0.30556 | 0.42778 | 0.31111 | **-0.00556** |

The conservative v51 Music anchor remains unchanged on v4 and v5:

| Bank | v51 EER | v51 + .025 EER |
|---|---:|---:|
| codec mixed dev v4 | 0.25333 | 0.25333 |
| codec mixed blind v5 | 0.18333 | 0.18333 |

The small v4 improvement is not broad.  It appears in transcode G.711/Opus
(`0.4167 -> 0.4000`), partial overlap (`0.36 -> 0.35`), and sequential mixtures
(`0.21 -> 0.20`).  In contrast, the same residual worsens v5 transcode
G.711/Opus (`0.1667 -> 0.2083`) and v7 G.722 (`0.1389 -> 0.1667`).  All other
codec and mixture-mode slice EERs are essentially unchanged.  This sign flip
is the reason the candidate is rejected as non-generalizing.

Full predictions and slice metrics are in
`reports/mert_generator_robust_v53_retrospective/`.

## Gap to the target

The official leader target is ADS `0.821160`; the best confirmed local
submission baseline used for this calculation is ADS `0.7363412698`.  The ADS
gap is **0.08481873**.  If Music were the only axis improved, its coefficient
of 0.30 means an absolute Music EER reduction of **0.28273** would be required.
This head delivers `-0.00056` on the retrospective-bank mean, so it is orders
of magnitude short and moves in the wrong direction.

The next Music experiment should therefore change the evidence or supervision,
not tune this residual: retain longer raw temporal sequences rather than only
three pooled views, expand truly source-disjoint real and fake music, and use a
locked cross-corpus selection set whose ranking agrees with codec banks before
considering deployment.

## Reproduction and tests

Training and evaluation entry points:

```bash
python scripts/train_mert_generator_robust_music_v53.py --help
python scripts/evaluate_mert_generator_robust_v53.py \
  --statistics output/mert_retrospective_v53 \
  --checkpoint reports/mert_generator_robust_v53_dual/mert_generator_robust_music.pt \
  --output-dir reports/mert_generator_robust_v53_retrospective --device cuda
```

Relevant tests, together with the MERT-statistics and data-guard tests, pass:
**30 passed**.  They cover feature geometry, gradient flow, numerical
stability, source balancing, directed codec pairs, residual guards, and
single-forward runtime reuse.
