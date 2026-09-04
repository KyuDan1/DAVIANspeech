# Low-weight WPT residual over exact-v18 + EAT patch-graph 5%

## Recommendation

Use **one fixed 5% outer WPT residual on both Voice and File**, applied after the
5% EAT patch-graph File/Music residual:

```text
anchor = exact v18
anchor.File, anchor.Music <- 0.95 * anchor + 0.05 * EAT-patch  (logit space)
final.Voice, final.File   <- 0.95 * anchor + 0.05 * WPT-fixed-expert (logit space)
final.Music               = patch anchor Music
presence/CPS              = exact v18
```

Do not use 2.5%: it leaves measurable gain unused and regresses three broad
channel subgroups versus patch-only. Do not use 10% for the first combined
submission: although it has the best aggregate ADS, it newly worsens two of 57
controlled contrasts, including the known hard `concurrent voice-fake +
music-real` File contrast. The 5% setting improves every aggregate bank and
none of the 57 controlled contrasts.

## Exact aggregate audit

The audit used only frozen CSV predictions and the repository's official EER /
ADS implementation. Fusion order and clipping match deployment: patch fusion
uses `1e-6`, followed by WPT fusion using `1e-5`. The WPT fixed expert is the
v39 internal task-wise MoE (`Unified/WPT = 0.10/0.90` for Voice and
`0.20/0.80` for File), not WPT standalone. The same outer weight was applied
to Voice and File; Music and both presence outputs were untouched by WPT.

| bank | exact v18 ADS | patch 5% ADS | +WPT 2.5% | +WPT 5% | +WPT 10% |
|---|---:|---:|---:|---:|---:|
| factorial dev | 0.697922 | 0.726104 | 0.727247 | **0.734260** | 0.745325 |
| factorial locked | 0.745506 | 0.757896 | 0.762623 | **0.763766** | 0.768961 |
| phone locked | 0.733857 | 0.790929 | 0.794857 | **0.796929** | 0.802429 |
| YuE locked | 0.828448 | 0.849252 | 0.849252 | **0.855719** | 0.855719 |
| four-bank mean | 0.751433 | 0.781045 | 0.783495 | **0.787668** | 0.793108 |

Relative to patch-only, mean ADS changes are `+0.002450`, `+0.006623`, and
`+0.012063` for 2.5%, 5%, and 10%. Their worst aggregate-bank changes are
`+0.000000`, `+0.005870`, and `+0.006467`. Thus every tested dose is
aggregate-safe, and 5%/10% improve all four banks.

At the recommended 5% dose, task EERs are:

| bank | File EER | Voice EER | Music EER | ADS delta vs v18 | ADS delta vs patch 5% |
|---|---:|---:|---:|---:|---:|
| factorial dev | 0.262909 | 0.268571 | 0.268571 | +0.036338 | +0.008156 |
| factorial locked | 0.238182 | 0.217143 | 0.245714 | +0.018260 | +0.005870 |
| phone locked | 0.194143 | 0.162500 | 0.245000 | +0.063071 | +0.006000 |
| YuE locked | 0.148777 | 0.083333 | 0.177419 | +0.027271 | +0.006467 |

Coverage is intentionally limited to the one dev and three locked corpora for
which the repository can reconstruct **exact v18** and both frozen experts by
ID. The EAT/WPT CSVs contain six additional common dev corpora, but there is no
exact-v18 reconstruction for those banks; their standalone scores were not
misrepresented as combined-v18 results.

## Controlled-cell safety

Using the existing locked cell-audit definitions (57 valid task/contrast
comparisons):

| WPT outer dose | improved | regressed | mean EER change vs patch-only | worst new EER change |
|---|---:|---:|---:|---:|
| 2.5% | 12 | 0 | -0.005263 | 0.000000 |
| **5%** | **15** | **0** | **-0.011842** | **0.000000** |
| 10% | 18 | 2 | -0.017544 | +0.040000 |

The 10% regressions are both factorial File contrasts:

- concurrent, RR versus voice-fake + music-real: EER `0.48 -> 0.52`;
- partial overlap, RR versus voice-real + music-fake: EER `0.20 -> 0.24`.

Broad subgroup ADS is quantized because several cells are small. Versus
patch-only, 5% has two negative channel groups (factorial dev and locked
`stereo_wav`, each about `-0.0068/-0.0054` ADS), whereas 10% has one
(`factorial locked telephone_flac`, `-0.0068`). This does not overturn the 5%
choice because 10% damages two controlled component contrasts central to the
known File failure mode, and previous larger residuals have transferred poorly
to hidden evaluation.

## v39 package and runtime evidence

The inspected `wpt_fixed_moe_v39.zip` is structurally usable as the source of
the WPT fixed expert, but **not directly as this candidate**: its entrypoint
uses v38 as the outer anchor and deployed outer weights Voice/File =
`0.10/0.60`. A combined package must preserve exact-v18, export the unified
expert without applying the v38 Music residual, apply patch 5%, then apply the
recommended WPT 5% Voice/File residual.

Current v39 archive evidence:

- compressed `8,286,763,669` bytes; uncompressed `9,189,597,377` bytes;
- 134 entries, no duplicate or unsafe paths, top-level only `model/`,
  `script.py`, `requirements.txt`;
- sole requirement `onnxruntime-gpu==1.23.2`;
- WPT uses three 4.04-second views, file batch 8;
- isolated B200 peak allocated/reserved `3.26/3.30 GiB`;
- batch-8 and batch-4 frozen predictions agree to at most `4.1e-8` on their
  2,124 common rows;
- adding the patch head/code costs only about 1.94 MB, so the combined archive
  remains comfortably below the 10 GB compressed and 32 GB expanded limits.

Runtime should remain below 60 minutes on L4, but this is an evidence-based
estimate rather than an L4 end-to-end measurement. The repository records
`31-33 min / 1,200` for the exact-v18 family and about `41 min` for the later
EAT-statistics pipeline. WPT 5-view/batch-6 scoring took about 28 seconds for
2,124 files on B200; the proposed 3-view/batch-8 pass is lighter. Patch graph
and unified heads reuse the existing EAT/SPEAR forwards. Allowing several-fold
L4 slowdown for the WPT pass and overhead gives a conservative combined range
of roughly **42-50 minutes**, leaving about 10 minutes of margin. Outer weight
does not affect runtime.

Final go/no-go should still require a full 1,200-file L4 wall-clock run (or a
representative linear extrapolation), because v39 has only CUDA smoke,
B200 throughput, and memory evidence—not a recorded full L4 timing.

## Inputs and reproducibility

- exact-v18 reconstruction: `scripts/evaluate_wpt_v38_residual.py`
- metric implementation: `scripts/evaluate_diagnostic.py`
- patch expert: `reports/eat_patch_graph_v1/seed01_balanced_fm/dev_predictions.csv`
  and `seed01_balanced_fm_locked/predictions.csv`
- WPT fixed expert: `reports/wpt_spectra_v1/fixed_moe_export/fixed_predictions.csv`
- controlled contrasts: `reports/eat_patch_graph_v1/cell_audit_v1/audit.py`
- deployed fusion references: `eat_patch_graph_v44/model/src/eat_patch_graph_inference.py`
  and `wpt_fixed_moe_v39/model/src/wpt_spectra_inference.py`

