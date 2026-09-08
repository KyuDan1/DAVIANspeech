# Exact-v47 label-blind anchor cache runner

`scripts/run_v47_anchor_cache.py` keeps preparation, GPU inference, and merge
as three explicit phases. It accepts only truth manifests declared under
`train` or `development` in `configs/data_partitions.yaml`. Router-training,
training-validation, locked, OOD, stress, retrospective, invalid, unknown, or
multiply-declared manifests are rejected. The repository data
guard runs before every preparation, launch, and merge validation.

Preparation reads only the truth `ID` column after the data guard completes.
It writes a label-free dataset map, hashes every truth/audio/package input,
creates globally unique combined IDs, and symlinks the selected audio into one
corpus. Each shard receives an isolated round-robin audio directory and
sample-submission containing only its own IDs, its own output directory, an
exact-package entrypoint overlay, and symlinks to the read-only
`component_query_or_v47/model` assets. The package directory and its existing
`data/test` link are never modified.

The physical shard isolation is required because only the base pipeline
understands its `--num-shards` arguments; later EAT/SPEAR/MERT fusion calls
enumerate every file visible in `data/test`. Each isolated process therefore
runs internally as shard `0/1`. A regression test requires disjoint audio and
CSV views for every prepared shard, preventing downstream ID drift.

The model-source overlay starts from the exact package `pipeline.py`. It
imports only the opt-in XLS-R archive implementation from current
`src/pipeline.py` and adds one original-mixture XLS-R pass after the base
prediction CSV is written. The pass logs `CACHE_EXPORT_TIMING` with encoder
and archive seconds and verifies that the base CSV hash did not change. The
entrypoint remains the exact v47 fusion sequence; its opt-in delta retains the
already-produced EAT patch graph and SPEAR component-bin archives before the
package deletes its temporary files. It logs each retained hash and verifies
the final exact-v47 CSV hash before and after retention. No extra EAT or SPEAR
encoder pass is added.

Example (preparation only; this does not launch inference):

```bash
python scripts/run_v47_anchor_cache.py prepare \
  --work-dir reports/v47_anchor_cache_run_01 \
  --num-shards 8 \
  --dataset data/eval/channel_invariant_factorial_train_v1/truth.csv \
  --dataset data/eval/factorial_eval_1200_v2/truth_dev.csv
```

GPU execution requires both the explicit `launch` subcommand and confirmation
token `--confirm RUN_EXACT_V47_CACHE`. After all shards finish, `merge`
revalidates roles and every config/truth/audio/package hash, requires the exact
round-robin row set from each shard, checks finite `[0,1]` predictions, and
strictly validates all cache schemas before atomically publishing results.

Schema v1 of this runner incorrectly exposed the full corpus to every process.
Its base pipeline and optional XLS-R pass completed correctly, but the first
unsharded post-pipeline fusion rejected the extra IDs. Those expensive base
artifacts can be recovered without trusting partial fusion output through
`adopt-schema-v1-base`: it accepts only four conditions together—identical
dataset/package provenance, the exact all-shard nonzero launch status, exactly
one known ID-mismatch traceback after `CACHE_EXPORT_TIMING`, and a prediction
hash matching the logged pre-fusion hash. `launch-post-fusions` then runs the
unchanged remaining v47 fusion sequence against schema-v2 isolated views.
Both recovery actions require explicit confirmation tokens, record hashes,
and refuse existing post-fusion caches.

Per-dataset output is under `cache/datasets/<dataset-key>/` and restores the
original truth-manifest ID order. It contains:

- `exact_v47_predictions.csv`: `ID` plus the five competition probability
  columns.
- `xlsr_windows.npz`: `ids [N]`, `embeddings [N,W,1920]` float32, prefix
  `mask [N,W]`, sample `starts [N,W]`, scalar `window`, and scalar
  `sample_rate=16000`.
- `eat_patch_graph.npz`: `ids [N]`, `temporal [N,V,L,2,T,128]`,
  `spectral [N,V,L,2,F,128]`, boolean `view_mask [N,V]`, fixed
  `projection [768,128]`, and `layers [L]` metadata. Exact v47 currently
  writes the graph tensors as float16.
- `spear_component_bins.npz`: `ids [N]`,
  `features [N,V,B,L,4,64]`, boolean view/bin `mask [N,V,B]`, fixed
  `projection [1280,64]`, `layers [L]`, and scalar `bins=B` metadata. Exact
  v47 currently writes features as float16.

`merge_provenance.json` records all shard and published artifact SHA-256
digests. Neither the combined map nor any cache contains target labels.

The canonical files above are also exposed without duplication through
relative symlinks matching the three-stream trainer's cache-root convention:
`anchor/<dataset>/predictions.csv`, `xlsr/<dataset>/features.npz`,
`eat/<dataset>/features.npz`, and `spear/<dataset>/features.npz`. Pass those
four parent directories as the corresponding cache roots. The SPEAR root uses
the trainer's per-dataset exact-v47 reader because these package-generated
archives intentionally do not invent the `ranges` field found in older
standalone extraction caches.

The combined audio and model trees are symlinks, so they consume negligible
additional storage. With exact-v47 dimensions the uncompressed fixed feature
payload is 473,088 bytes per row (423,936 EAT plus 49,152 SPEAR), plus 7,680
bytes for each valid XLS-R window. Completed shard archives and merged caches
coexist, so steady-state storage is roughly twice one feature copy; merge also
uses one temporary uncompressed disk-backed EAT/SPEAR spool. For the current
238-row codec bank this is about 107 MiB per uncompressed fixed-feature copy,
plus XLS-R windows, ZIP metadata, and compression variance.
