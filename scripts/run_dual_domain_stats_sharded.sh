#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 4 || $# -gt 6 ]]; then
  echo "usage: $0 DATASET STREAM CHANNEL OUTPUT_ROOT [NUM_SHARDS] [GPU_OFFSET]" >&2
  exit 2
fi

dataset=$1
stream=$2
channel=$3
output_root=$4
num_shards=${5:-8}
gpu_offset=${6:-0}

root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
python_bin=${PYTHON_BIN:-/home/nas_main/kyudanjung/conda_envs/envs/davianspeech/bin/python}
audio_dir="$root/data/eval/$dataset/audio"
truth="$root/data/eval/$dataset/truth.csv"
destination="$output_root/$dataset/$stream/$channel"
mkdir -p "$destination"

training_args=()
case "$dataset" in
  external_mixed_train_v1|mixed_devvoice_train_v1|mixed_fmc_music_train_v1|\
  phone_router_voice_train_v1|multigen_music_presence_train_v1)
    training_args=(--training-truth "$truth")
    ;;
esac

pids=()
for ((shard=0; shard<num_shards; shard++)); do
  gpu=$((gpu_offset + shard))
  CUDA_VISIBLE_DEVICES=$gpu "$python_bin" "$root/scripts/extract_dual_domain_stats.py" \
    --test-dir "$audio_dir" \
    --output "$destination/shard_$shard.npz" \
    --stream "$stream" \
    --channel-variant "$channel" \
    --num-shards "$num_shards" \
    --shard-index "$shard" \
    "${training_args[@]}" \
    >"$destination/shard_$shard.log" 2>&1 &
  pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    status=1
  fi
done
if [[ $status -ne 0 ]]; then
  tail -n 20 "$destination"/*.log >&2
  exit "$status"
fi

"$python_bin" - "$destination" <<'PY'
from pathlib import Path
import sys
import numpy as np

directory = Path(sys.argv[1])
paths = sorted(directory.glob("shard_*.npz"))
count = sum(len(np.load(path, allow_pickle=False)["ids"]) for path in paths)
print(f"complete: {directory} ({count} examples in {len(paths)} shards)")
PY
