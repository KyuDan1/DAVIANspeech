# Development role registration

The authoritative `codec_mixed_dev_v4` bank was registered under
`development` only after the frozen pre-registration validation reported
30/30 PASS.

- Truth SHA-256:
  `6a87bc333eb88306d3f31002ac519bb484f903dd1c419824bbe1c28a5b41881b`
- Pre-registration validation SHA-256:
  `fae42e8b0516f020c3ee7f58aa69ddf387b80083a1222ed4c866fd95d23977f7`
- Partition config SHA-256 after registration:
  `5a3da174e5f5f300c4953b3bf443d82cf541d0f2b8375bb1964528afcbba2c9c`
- Repository data guard after registration: all 12 train/router manifests PASS.

This bank may be used repeatedly for development checkpoint and loss
selection.  It is not a blind/locked performance claim.  The retired
prospective v2 bank remains under `retrospective_diagnostic` and is forbidden
for selection.
