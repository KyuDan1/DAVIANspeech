#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="${1:-davianspeech}"

# Restrict creation to conda-forge. Recent conda releases otherwise inspect
# configured Anaconda defaults and can require their Terms of Service even
# though this project does not need those channels.
conda create -y -n "$ENV_NAME" --override-channels -c conda-forge \
    python=3.11 pip 'ffmpeg=7'

conda run -n "$ENV_NAME" python -m pip install \
    torch==2.8.0 torchaudio==2.8.0 \
    --index-url https://download.pytorch.org/whl/cu128

conda run -n "$ENV_NAME" python -m pip install -r requirements.txt \
    pytest==9.1.1 pyarrow==21.0.0 remotezip==0.12.5 yt-dlp==2026.8.19

echo "Environment ready. Activate it with: conda activate $ENV_NAME"
