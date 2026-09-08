# Source-restricted query/WPT Music consensus

## Question

Can the Music logits already produced by component-query EAT/SPEAR and the WPT
Spectra model be combined to improve unseen-source Music detection without an
additional backbone pass?

This is deliberately narrower than prior work. MERT temporal Group-DRO and
leave-one-generator-out, long-horizon SSL statistics, X-Codec CoMoE/ngrams,
SOFIA/MERT, SONICS, EAT segmental/hierarchical heads, and real-only EAT density
were already evaluated. The WPT forward is already present in v48, while the
component-query forward is inherited from v47.

## Protocol

- Inputs: frozen seed04 component-query and seed06 WPT development predictions.
- Candidate: convex logit consensus with WPT weights
  `[0, 0.1, 0.2, 0.3, 0.4, 0.5]`.
- Selection: maximize half mean plus half worst-bank Music quality, subject to
  zero Music-EER regression on every fitted bank.
- Source-restricted check: repeat selection seven times, each time withholding
  one authorized development bank and measuring only that withheld bank.
- Deployment safety check: substitute the selected consensus for the raw query
  Music expert in the exactly reconstructed factorial-dev v47 chain, before
  its existing 5% query residual and 30% File noisy-OR.
- Data used: seven `development` banks from `configs/data_partitions.yaml`.
  No factorial holdout, phone, YuE, or other locked truth/prediction was opened.

## Standalone expert result

The selected expert consensus is 50% query / 50% WPT in logit space.

| authorized development bank | query Music EER | consensus Music EER |
|---|---:|---:|
| MixFake Music | 0.086250 | 0.075000 |
| External mixed | 0.110000 | 0.100000 |
| Source-disjoint mixed | 0.130000 | 0.120000 |
| Source-disjoint mixed equal | 0.070000 | 0.060000 |
| Source-disjoint music | 0.040000 | 0.035000 |
| Factorial dev | 0.182857 | 0.171429 |
| Telephone mixed dev | 0.180000 | 0.173333 |

Mean Music EER improves `0.114158 -> 0.104966`; worst-bank Music EER improves
`0.182857 -> 0.173333`; the robust selection score improves
`0.851492 -> 0.860850`. All seven leave-one-domain-out trials improve or tie
their unseen bank. Selected fold weights are 0.5 in five folds, 0.4 in one,
and 0.3 in one.

## Deployment safety and decision

**Reject; do not change or package the current submission.** Despite its clean
standalone result, inserting the consensus at the existing query-expert point
causes destructive rank interaction with the v18/patch Music anchor:

| exact factorial-dev v47 chain | File EER | Music EER | ADS |
|---|---:|---:|---:|
| Current query expert | 0.238182 | 0.205714 | 0.762052 |
| Query/WPT consensus | 0.238182 | 0.240000 | 0.751766 |

Music EER regresses by `+0.034286`, overall ADS falls by `-0.010286`, and the
minimum channel ADS delta is `-0.020392` (telephone FLAC). Thus expert-level
OOD complementarity is real but is not composable at the current nested 5%
fusion location. A future attempt should train or select against the final
anchor residual directly, with exact authorized-dev anchors for all banks,
instead of optimizing the raw expert in isolation.

`selection.json`, `bank_sweep.csv`, `leave_one_domain_out.csv`, and the
factorial-v47 tables contain the exact reproducible results. The experiment is
implemented in `scripts/experiment_source_restricted_query_wpt_music.py`.
