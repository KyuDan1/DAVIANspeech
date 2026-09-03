#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 3 ]]; then
  echo "usage: $0 TEST_BANK OUTPUT_CSV SHARD_DIR" >&2
  exit 2
fi

root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
bank=$1
output=$2
shard_dir=$3
package=${LME_PACKAGE:-$root/submit_lme_spear_v1}
python_bin=${PYTHON_BIN:-/home/nas_main/kyudanjung/conda_envs/envs/davianspeech/bin/python}
num_shards=${NUM_SHARDS:-8}
mkdir -p "$shard_dir" "$(dirname "$output")"

pids=()
for ((shard=0; shard<num_shards; shard++)); do
  CUDA_VISIBLE_DEVICES=$shard PYTHONNOUSERSITE=1 "$python_bin" \
    "$package/model/src/pipeline.py" \
    --test-dir "$bank/audio" \
    --sample-submission "$bank/sample_submission.csv" \
    --output "$shard_dir/shard_$shard.csv" \
    --panns-dir "$package/model/panns" \
    --xlsr-dir "$package/model/xlsr" \
    --artifactnet-dir "$package/model/artifactnet" \
    --separator htdemucs \
    --htdemucs-repo "$package/model/htdemucs" \
    --pooling logmeanexp --temperature 5 \
    --num-shards "$num_shards" --shard-index "$shard" \
    >"$shard_dir/shard_$shard.log" 2>&1 &
  pids+=("$!")
done
status=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then status=1; fi
done
if [[ $status -ne 0 ]]; then
  tail -n 30 "$shard_dir"/*.log >&2
  exit "$status"
fi

"$python_bin" "$root/scripts/merge_shards.py" \
  --shard-dir "$shard_dir" \
  --sample-submission "$bank/sample_submission.csv" \
  --output "$output"

# SPEAR is much lighter than XLS-R-2B; one B200 post-pass is sufficient and
# exactly matches the verified competition package's final operation.
CUDA_VISIBLE_DEVICES=0 PYTHONPATH="$package/model/src" "$python_bin" - \
  "$bank/audio" "$output" "$package" <<'PY'
from pathlib import Path
import sys
from anchor_spear_fusion import apply_fusion

audio, output, package = map(Path, sys.argv[1:])
apply_fusion(
    audio, output, package / "model/spear",
    package / "model/spear-mixed-music-head.npz",
    package / "model/spear-cross-component-joint-v1.npz",
    device="cuda", weight=0.10,
)
PY
