# SAM-Audio environment

SAM-Audio cannot share an environment with the detector stack, so it gets its
own and runs as a separate pass (`scripts/separate_sam.py`) that leaves stems
on disk for `pipeline.py --separator precomputed`.

## What the model card gets wrong

It says `pip install git+https://github.com/facebookresearch/sam3.git`. That
installs SAM 3 (image/video segmentation), which contains no `sam_audio`
module — and its `numpy<2` pin plus an unpinned `torch` dependency will drag
the detector environment onto a cu130 build this driver cannot run. The real
package is **[facebookresearch/sam-audio](https://github.com/facebookresearch/sam-audio)**.

## Working combination

Every pin below is load-bearing; the version that pip resolves on its own does
not work.

```bash
conda create -n samaudio -c conda-forge python=3.11 pip
conda install -n samaudio -c conda-forge 'ffmpeg=7'   # torchcodec 0.4 wants libavutil 56-59

# Install torch BEFORE sam-audio, or pip pulls a cu130 build.
pip install torch==2.8.0 torchaudio==2.8.0 torchvision==0.23.0 \
    --index-url https://download.pytorch.org/whl/cu128

git clone https://github.com/facebookresearch/sam-audio.git && pip install ./sam-audio

pip install torchcodec==0.4.0            # 0.2.x lacks AudioDecoder; 0.16 is a CUDA-13 build
pip install xformers==0.0.32.post2 --index-url https://download.pytorch.org/whl/cu128
pip install 'transformers>=4.54,<5' 'huggingface_hub<1.0'
pip install 'setuptools<81'              # ImageBind imports pkg_resources
```

| Pin | Why |
| --- | --- |
| `torch 2.8 + cu128` | cu130 wheels need a CUDA 13 driver; this host has 12.9. torchcodec 0.4 is built against 2.8. |
| `torchcodec==0.4.0` | 0.2.x (the torch-2.6 pair) has no `AudioDecoder`; 0.16.0 wants `libnvrtc.so.13`. |
| `ffmpeg=7` | torchcodec 0.4 links `libavutil.so.56-59`; ffmpeg 9 ships `.60`. |
| `xformers==0.0.32.post2` | The torch-2.6 build calls a triton API that torch 2.8's triton rejects. |
| `huggingface_hub<1.0` | `sam_audio.model.base.BaseModel._from_pretrained` still declares the pre-1.0 keyword-only `proxies`/`resume_download`. |
| `transformers<5` | transformers 5.x requires `huggingface_hub>=1.5`, which contradicts the row above. |
| `setuptools<81` | 81 removed `pkg_resources`, which ImageBind imports at module load. |

Run it with the conda env's libraries on the loader path:

```bash
LD_LIBRARY_PATH=$CONDA_PREFIX/lib python scripts/separate_sam.py \
    --test-dir data/test --out-dir stems/sam-large \
    --checkpoint models/sam-audio-large
```

## Checkpoint sizes

| Model | checkpoint.pt |
| --- | --- |
| `facebook/sam-audio-large` | 14.9 GB |
| `facebook/sam-audio-base` | 7.7 GB |

Both are gated: request access on the model page and pass an approved
`HF_TOKEN`.
