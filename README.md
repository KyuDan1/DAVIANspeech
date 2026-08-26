# DAVIANspeech

AI-generated audio detection for the *AI 생성 오디오 탐지* competition.

This is the competition baseline with two of its three stages replaced:

| Stage | Baseline | Here |
| --- | --- | --- |
| Voice / music presence | PANNs Cnn14 | **PANNs Cnn14** (unchanged) |
| Source separation | HTDemucs | HTDemucs **or SAM-Audio** |
| Spoof detection | DF-Arena 1B | **XLS-R-2B-AntiDeepfake** |

The presence stage is left alone deliberately — it already scores ~0.989 on
the public leaderboard, so there is nothing to win there.

```text
INPUT AUDIO
|
+-- PANNs Cnn14 ------> VOICE_PRESENT_PROB (VP), MUSIC_PRESENT_PROB (MP)
|
+-- Separator --------> voice stem --> XLS-R-2B --> VOICE_FAKE_PROB (VF)
    (HTDemucs |         music stem --> XLS-R-2B --> MUSIC_FAKE_PROB (MF)
     SAM-Audio)

FILE_FAKE_PROB = max(VP * VF, MP * MF)
```

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
conda create -n davianspeech -c conda-forge python=3.11
conda activate davianspeech
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
conda install -c conda-forge ffmpeg   # needed for .m4a / .wma / .aac
```

Fetch the checkpoints:

```bash
huggingface-cli download nii-yamagishilab/xls-r-2b-anti-deepfake \
    --local-dir models/xls-r-2b-anti-deepfake          # 8.65 GB
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
    --output-dir   submission --zip
```

That lands at **4.41 GiB**, under the 5.0 GiB baseline package, because the
XLS-R weights ship as fp16 while inference still runs in fp32.

Verify it the way the grader will — from the package root, with no network:

```bash
cd run_dir && ln -s /path/to/data data
HTTP_PROXY=http://127.0.0.1:9 HTTPS_PROXY=http://127.0.0.1:9 \
    python /path/to/submission/script.py
```

## Layout

```
src/xlsr_antideepfake.py   fairseq -> transformers remap + spoof scoring
src/separation.py          HTDemucs and SAM-Audio behind one interface
src/presence.py            PANNs Cnn14 voice/music presence
src/pipeline.py            end-to-end inference, sharding-aware
src/evaluate.py            ROC-AUC and EER per probability column
scripts/run_sharded.sh     multi-GPU fan-out
scripts/merge_shards.py    reassemble shards in submission order
```

## Licensing

The XLS-R-2B-AntiDeepfake weights are **CC BY-NC-SA 4.0** (research and
educational use). SAM-Audio is under the SAM License. Both are obligations on
the weights, not on this code — check them against your intended use.
