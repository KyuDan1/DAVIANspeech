# Codec mixed development v4 validation

## Result

**30/30 checks PASS.**
The bank is built, unscored, validated, and intentionally absent from 
`configs/data_partitions.yaml` pending an explicit later registration decision.
No authenticity detector was run and no prospective-v2 prediction or score was read.

## Counts

- 120 unique bases and 600 paired codec renders.
- 12 cells with 10 bases / 50 renders each.
- 240 exact extracted sources; no missing or extra members.
- 60/60 fake-music sources pass PANNs 0.20 and Demucs -1.5 dB.
- Five FakeMusicCaps generators contribute exactly 12 fake-music sources each.
- 67/67 archive volume hashes match the fixed repository revision.
- 124 identity-view comparisons across 47 configured manifests have zero overlap.

## Core hashes

| Artifact | SHA-256 |
| --- | --- |
| Reservation | `cf7f8e9fe355432ed4d87c3031e45b71049013ff697720a59e2fdbae34bcded5` |
| Acoustic source scores | `a11e3a99d610bb8249978143430142c59b2491bceb9de1e4faa4e4d30beb861c` |
| Truth | `6a87bc333eb88306d3f31002ac519bb484f903dd1c419824bbe1c28a5b41881b` |
| Selected-source hash manifest | `d421901c04baec6a34dae4e8a3e8a3c7b40c4f327bc2c478b1e550bda6893048` |
| Rendered-audio hash manifest | `69b07208e2ed7c6f0a0de4367a612190904956b5a958e081a353e4d4837fe341` |
| Builder | `051b0cd82fd11827820e6017b6a3cb0b6e11612f54d80f03b7f2211b01d8f5a7` |
| Validation JSON | `fae42e8b0516f020c3ee7f58aa69ddf387b80083a1222ed4c866fd95d23977f7` |

Full paths, tool/model hashes, archive revision and validation evidence are in 
`full_provenance.json`, `archive_volume_validation.csv`, and 
`role_overlap_by_column.csv`.
