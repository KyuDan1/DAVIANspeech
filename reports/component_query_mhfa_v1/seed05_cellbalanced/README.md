# Seed05 cell-balanced component-query MHFA

This directory contains one bounded training run of the existing component-query
MHFA model. No shared source file was changed for the run. The imported training
sampler was wrapped at launch time so that each
`(dataset, layout, voice/music presence, voice/music fake)` cell has equal
expected mass within its corpus while retaining generator-aware weights inside
each cell.

## Clean checkpoint and selection

- Seed: `20260905`
- Checkpoint: `component_query_mhfa.pt`
- Best epoch: `42`
- Selection score: `0.8209143089053803`
- Mean authorized-dev ADS: `0.8312801226551226`
- Worst authorized-dev ADS: `0.7725974025974027`
- Selection data: the same seven authorized development banks used by the base
  trainer; locked factorial holdout, phone, and YuE banks were not used to train
  or select this checkpoint.

`summary.json`, `history.csv`, `dev_metrics.csv`, `dev_predictions.csv`, and
`audit_provenance.json` document this clean run. The checkpoint is therefore a
valid frozen candidate for a future development-only fusion experiment.

## Locked diagnostics are not selection evidence

Files under the sibling directory `../seed05_cellbalanced_locked/`, including
`deployment_comparison.csv`, `fusion_grid_diagnostic.csv`, and
`hybrid_deployment_metrics.csv`, were produced after checkpoint selection for
diagnosis. The hybrid weights were inspected against locked factorial holdout,
phone, and YuE results. In particular, the conservative File weight was favored
partly because it preserved locked YuE performance. These diagnostics are
therefore contaminated for hyperparameter/model selection and must not be used
to package or claim a deployable hybrid.

## Decision

Do not replace the current v47 deployment and do not package either diagnostic
hybrid. Cell balancing remains a promising training intervention, but the hard
concurrent cell moved from `0.28` to `0.40` EER on factorial dev while moving
from `0.44` to `0.40` on locked factorial holdout, indicating substantial
split/seed variance. Any future fusion must use a predeclared objective and only
authorized development predictions, then be frozen before one-time evaluation
on a genuinely untouched bank.
