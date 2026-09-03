#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 || $# -gt 4 ]]; then
  echo "usage: $0 OUTPUT_DIR [NUM_SHARDS] [GPU_COUNT] [GPU_OFFSET]" >&2
  exit 2
fi

destination=$1
num_shards=${2:-8}
gpu_count=${3:-4}
gpu_offset=${4:-0}
root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
python_bin=${PYTHON_BIN:-/home/nas_main/kyudanjung/conda_envs/envs/davianspeech/bin/python}
mkdir -p "$destination"

datasets=(
  external_mixed_train_v1 mixed_devvoice_train_v1 mixed_fmc_music_train_v1
  mixfake_music_train_v1 telephone_mixed_train_v1 temporal_mixed_train_v2
  channel_invariant_factorial_train_v1 mixfake_music_dev_v1 external_mixed_v1
  source_disjoint_mixed_v1 source_disjoint_mixed_equal_v1
  source_disjoint_music_v1 telephone_mixed_dev_v1
)
training=(
  external_mixed_train_v1 mixed_devvoice_train_v1 mixed_fmc_music_train_v1
  mixfake_music_train_v1 telephone_mixed_train_v1 temporal_mixed_train_v2
  channel_invariant_factorial_train_v1
)

pids=()
for ((shard=0; shard<num_shards; shard++)); do
  gpu=$((gpu_offset + shard % gpu_count))
  CUDA_VISIBLE_DEVICES=$gpu "$python_bin" \
    "$root/scripts/extract_spear_temporal_bins.py" \
    --datasets "${datasets[@]}" --training-datasets "${training[@]}" \
    --output "$destination/shard_$shard.npz" \
    --num-shards "$num_shards" --shard-index "$shard" \
    >"$destination/shard_$shard.log" 2>&1 &
  pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then status=1; fi
done
if [[ $status -ne 0 ]]; then
  tail -n 30 "$destination"/*.log >&2
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
