# External generalist audit and v54 Voice decision

## Decision

Keep only a **7.5% logit residual from Forensics XLS-R Wild** for
`VOICE_FAKE_PROB`, evaluated on the original mixture with three deterministic
5-second crops.  Apply the already-frozen component-consistent File update
after this Voice update.  Do not add Spectra-AASIST3 or DF Arena 500M.

This is deliberately a low-weight diversity expert, not a replacement for the
v50 Voice detector.  Both external models were much weaker when music was
present even when their pure-speech scores were excellent.

## Sources and licences

| model | fixed revision | licence | released purpose |
|---|---|---|---|
| Spectra-AASIST3 | `bc0ded888080ddad493177bb53aa6f5b95219d7c` | Apache-2.0 | cross-corpus speech anti-spoofing |
| Forensics XLS-R Wild | `aa10055181a35eb7b4916175d2d72266def78c1b` | CC-BY-NC-4.0 | real-world/codec-augmented speech deepfake detection |
| DF Arena 500M | `8258fa8e74ff9b8ad20d4c939c1a7f694a6e4080` | non-commercial research | speech plus singing/environmental universal anti-spoofing |

Primary release pages:

- <https://huggingface.co/lab260/Spectra-AASIST3>
- <https://huggingface.co/eliya/forensics_0.3B_xlsr_wild_deepfake_classifier>
- <https://huggingface.co/Speech-Arena-2025/DF_Arena_500M_V_1>

The local Forensics checkpoint is pinned by SHA-256
`39e37fa5a958c2b46ebb6a5937874c36c39003906dcdd5d0aba6154bc6b2dc21`.
Its vendored loader constructs the included XLS-R architecture locally, so
evaluation does not need a network connection.

## Pure-speech and channel audit

All results below use fake-oriented scores and the competition EER rule.

| model | clean Voice EER | paired telephone Voice EER | clean/phone score correlation |
|---|---:|---:|---:|
| existing Spectra-AASIST | 0.00000 | 0.01137 | 0.9757 |
| Spectra-AASIST3 | 0.03411 | 0.01137 | 0.8326 |
| Forensics XLS-R Wild | 0.00000 | 0.01137 | **0.9957** |
| DF Arena 500M | 0.00000 | 0.00000 | 0.9801 |

Spectra-AASIST3 was rejected because its newer public benchmark result did not
transfer to this preprocessing/domain, and its paired channel stability was
substantially worse than the existing model.

## Mixed-audio audit

The pure-speech result cannot justify a hard route.  On the 600-row prospective
codec mixed development bank, direct Voice EER was `0.3800` for Forensics and
`0.3300` for DF Arena 500M, versus `0.2067` for frozen v50.  On the factorial
development bank, their direct Voice EERs were `0.2800` and `0.3686`.
DF Arena 500M also had Music EER `0.5333` on codec-mixed development and
`0.4629` on factorial development, so its singing/environmental training does
not make it a usable Music detector here.

Weights were selected on development only.  A fixed Forensics weight `0.075`
gave:

| bank | role | v50 Voice EER | + Forensics 7.5% | delta |
|---|---|---:|---:|---:|
| codec mixed v4 | selection | 0.20667 | **0.19667** | -0.01000 |
| codec mixed v5 | retrospective | 0.11667 | **0.10833** | -0.00833 |
| codec mixed v6 | retrospective | 0.08333 | **0.07500** | -0.00833 |
| codec mixed v7 | retrospective | 0.20000 | **0.19444** | -0.00556 |

No retrospective bank was used to alter the weight.  The fixed residual
improved aggregate Voice EER on every bank.  It is still intentionally small:
Opus alone worsened on v5 and v7 even though aggregate EER improved.

The same fixed `0.075` RAPTOR/DF-Arena residual improved v4, v5, and v7 but
worsened v6 (`0.08333` to `0.09167`), so it was rejected.  Its 1.745 GB
checkpoint also costs more deployment space than the selected 1.264 GB
Forensics checkpoint.

## v54 execution order

1. Run frozen v50 unchanged.
2. Score three deterministic 5-second crops of the **original mixture** with
   Forensics XLS-R Wild.
3. Update Voice in logit space:
   `0.925 * logit(v50_voice) + 0.075 * forensics_fake_logit`.
4. Leave Music and both presence outputs unchanged.
5. Run the already-frozen component-consistent File update last, so File sees
   the final Voice/Music probabilities.

The expected gain is modest but unusually consistent.  The larger remaining
gap to ADS 0.82116 must still come mainly from generator-unseen Music and the
File decision derived from it.
