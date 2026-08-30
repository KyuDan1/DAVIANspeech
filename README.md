# DAVIANspeech

AI-generated audio detection for the *AI 생성 오디오 탐지* competition.

This is the competition baseline with two of its three stages replaced:

| Stage | Baseline | Here |
| --- | --- | --- |
| Voice / music presence | PANNs Cnn14 | **PANNs Cnn14** (unchanged) |
| Source separation | HTDemucs | HTDemucs **or SAM-Audio** |
| Spoof detection | DF-Arena 1B | **XLS-R-2B + ArtifactNet** |

The presence stage is left alone deliberately — it already scores ~0.989 on
the public leaderboard, so there is nothing to win there.

```text
INPUT AUDIO
|
+-- PANNs Cnn14 ------> VOICE_PRESENT_PROB (VP), MUSIC_PRESENT_PROB (MP)
|
+-- Separator --------> voice stem --> XLS-R-2B --> VOICE_FAKE_PROB (VF)
    (HTDemucs |         music stem --> XLS-R-2B --+
     SAM-Audio)
                       full audio --> ArtifactNet-+--> MUSIC_FAKE_PROB (MF)

FILE_FAKE_PROB = max(fake probabilities for components with presence >= 0.7)
```

XLS-R and ArtifactNet contribute 50% each to the music score. ArtifactNet is a
small codec-residual detector that complements the large learned audio
classifier. Presence
scores are used as gates rather than multipliers: they are strong ranking
scores for CPS but are not calibrated probabilities, and multiplication was
found to damage file EER on music-only and sequential mixed audio.

## Why the checkpoint is remapped onto transformers

`nii-yamagishilab/xls-r-2b-anti-deepfake` ships a **fairseq** `Wav2Vec2Model`
under an `m_ssl.model.` prefix plus a `proj_fc` `Linear(1920, 2)` head. Its
`config.json` advertises `Wav2Vec2ForPreTraining`, but the parameter names are
fairseq's (`self_attn.k_proj`, `fc1`, `fc2`, `pos_conv.0.weight_g`), so
`from_pretrained` will not load it.

Upstream pins `fairseq==0.12.2`, which only installs on Python 3.9 and does not
run against torch 2.6. Rather than freeze the whole project on that,
[`src/xlsr_antideepfake.py`](src/xlsr_antideepfake.py) renames the fairseq
parameters onto `transformers.Wav2Vec2Model`. The architectures line up exactly
once fairseq's `layer_norm_first=True` is matched with HF's
`do_stable_layer_norm=True`. The loader is strict: every source tensor must
find a destination of identical shape, and any missing or unexpected key raises
rather than silently leaving weights at their random init.

Two details that fail silently if you get them wrong, both taken from
[AntiDeepfake](https://github.com/nii-yamagishilab/AntiDeepfake):

- **Logit order is `[fake, real]`.** `dataio.py` labels real audio `1` and fake
  `0`; `evaluation.py` scores with `softmax(...)[:, 1]` as the real-class
  probability. So `P(fake) = softmax(logits)[0]`.
- **Input is utterance-normalised** with `F.layer_norm(wav, wav.shape)` before
  it reaches the encoder.

## Setup

```bash
scripts/setup_environment.sh
conda activate davianspeech
```

The setup script uses PyTorch 2.8 with CUDA 12.8 and has been
smoke-tested on NVIDIA B200. `requirements.txt` remains available as a looser,
hardware-independent dependency list; install PyTorch for the target CUDA
version before using it.

Fetch the checkpoints:

```bash
huggingface-cli download nii-yamagishilab/xls-r-2b-anti-deepfake \
    --local-dir models/xls-r-2b-anti-deepfake          # 8.65 GB
huggingface-cli download intrect/artifactnet \
    --local-dir models/artifactnet                     # 17 MB
curl -L -o 'models/panns/Cnn14_mAP=0.431.pth' \
    'https://zenodo.org/record/3987831/files/Cnn14_mAP%3D0.431.pth?download=1'
```

HTDemucs downloads itself on first use. PANNs needs a
`models/panns/component_labels.json` naming the AudioSet classes that count as
voice and as music; the competition package ships one, and
[`configs/component_labels.fallback.json`](configs/component_labels.fallback.json)
stands in when it is unavailable.

**SAM-Audio** is gated (request access on the
[model page](https://huggingface.co/facebook/sam-audio-large)) and needs its
own environment — its dependency set cannot coexist with the detector stack.
See [docs/samaudio-environment.md](docs/samaudio-environment.md); note that
the model card's install line points at the wrong repository.

It therefore runs as a separate pass that writes stems for the detector to
pick up:

```bash
# in the samaudio env
python scripts/separate_sam.py --test-dir data/test --out-dir stems/sam-large \
    --checkpoint models/sam-audio-large

# back in the pipeline env
python src/pipeline.py --separator precomputed --stems-dir stems/sam-large
```

## Running

Single GPU:

```bash
python src/pipeline.py \
    --test-dir data/test --sample-submission data/sample_submission.csv \
    --output output/submission.csv --separator htdemucs
```

Across several GPUs (shards are round-robin, and the merged output is
bit-identical to a single-GPU run):

```bash
GPUS=1,2,3,4,5,6,7 scripts/run_sharded.sh
```

Score a run against labelled data:

```bash
python src/evaluate.py output/submission.csv data/ground_truth.csv
```

Run the model-independent regression tests:

```bash
python -m pytest -q
```

## Building the submission

The competition grades code, not predictions: you upload a zip holding
`script.py` and a `model/` directory, and the organisers run it against a test
set you never see. `open.zip` ships only three (byte-identical) example clips
and no labels, so **there is nothing to score locally** — the leaderboard is
the only evaluator.

```bash
python scripts/build_submission.py \
    --xlsr-dir     models/xls-r-2b-anti-deepfake \
    --panns-dir    models/panns \
    --htdemucs-dir baseline/model/htdemucs \
    --artifactnet-dir models/artifactnet \
    --output-dir   submission --zip
```

The current archive is **4.09 GiB** (4.43 GiB unpacked), under the competition
limits, because the XLS-R weights ship as fp16 while inference still runs in
fp32.

Verify it the way the grader will — from the package root, with no network:

```bash
cd run_dir && ln -s /path/to/data data
HTTP_PROXY=http://127.0.0.1:9 HTTPS_PROXY=http://127.0.0.1:9 \
    python /path/to/submission/script.py
```

## Measuring where the loss is

The leaderboard reports only `0.5*File + 0.2*Voice + 0.3*Music`, so a submission
tells you "better" or "worse" and nothing about which term moved. Pinning one
probability column to a constant fixes that column's EER at exactly 0.5 and
leaves the others alone, so the drop from an unprobed run names the term:

```bash
python scripts/build_submission.py --probe-column MUSIC_FAKE_PROB ...
python scripts/decode_probes.py --anchor 0.7083888889 \
    --music-probe 0.6698174603 --voice-probe 0.6515
```

Run against the current pipeline this gave **File 0.2741, Music 0.3714,
Voice 0.2156** — see [docs/probe-decomposition.md](docs/probe-decomposition.md)
for the submissions, the derivation, and why the anchor has to be the same
package.

## Layout

```
src/xlsr_antideepfake.py   fairseq -> transformers remap + spoof scoring
src/separation.py          HTDemucs and SAM-Audio behind one interface
src/presence.py            PANNs Cnn14 voice/music presence
src/pipeline.py            end-to-end inference, sharding-aware
src/evaluate.py            ROC-AUC and EER per probability column
scripts/run_sharded.sh     multi-GPU fan-out
scripts/merge_shards.py    reassemble shards in submission order
scripts/build_eval_korean.py  Korean voice eval set from FLEURS + synthetic fakes
scripts/gen_fake_audio8.py    Korean fakes via Audio8-TTS speaker cloning
scripts/build_mixtures.py     speech+music mixtures for separator comparison
scripts/run_eval_set.sh       score an eval set and print the diagnostic table
scripts/submit_dacon.py       upload a 4.4 GB zip from this machine
scripts/decode_probes.py      leaderboard ADS readings -> component EERs
```

## Licensing

The XLS-R-2B-AntiDeepfake weights are **CC BY-NC-SA 4.0** (research and
educational use). SAM-Audio is under the SAM License. Both are obligations on
the weights, not on this code — check them against your intended use.
