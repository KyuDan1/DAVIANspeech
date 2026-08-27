#!/usr/bin/env bash
# Fan the test set out across several GPUs, then merge the shards back into
# one submission ordered like sample_submission.csv.
#
#   GPUS=1,2,3,4,5,6,7 scripts/run_sharded.sh [extra pipeline.py args...]
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# An optional env file may define cluster-specific paths. With no env file,
# use the currently activated environment and the repository-local models.
if [ -n "${ENV_FILE:-}" ]; then
  source "$ENV_FILE"
fi

GPUS="${GPUS:-1,2,3,4,5,6,7}"
PY="${PY:-python}"
TEST_DIR="${TEST_DIR:-$HERE/data/test}"
SAMPLE_SUBMISSION="${SAMPLE_SUBMISSION:-$HERE/data/sample_submission.csv}"
OUTPUT="${OUTPUT:-$HERE/output/submission.csv}"
SEPARATOR="${SEPARATOR:-htdemucs}"
SHARD_DIR="${SHARD_DIR:-$HERE/output/shards}"

IFS=',' read -ra GPU_LIST <<< "$GPUS"
N=${#GPU_LIST[@]}
mkdir -p "$SHARD_DIR" "$(dirname "$OUTPUT")"

echo "Running $N shards on GPUs: $GPUS"
pids=()
for i in "${!GPU_LIST[@]}"; do
  CUDA_VISIBLE_DEVICES="${GPU_LIST[$i]}" \
  PYTHONNOUSERSITE=1 \
  "$PY" "$HERE/src/pipeline.py" \
      --test-dir "$TEST_DIR" \
      --sample-submission "$SAMPLE_SUBMISSION" \
      --output "$SHARD_DIR/shard_$i.csv" \
      --panns-dir "${PANNS_DIR:-$HERE/models/panns}" \
      --xlsr-dir "${XLSR_DIR:-$HERE/models/xls-r-2b-anti-deepfake}" \
      --sonics-dir "${SONICS_DIR:-$HERE/models/sonics-spectttra-gamma-5s}" \
      --separator "$SEPARATOR" \
      --num-shards "$N" --shard-index "$i" \
      "$@" > "$SHARD_DIR/shard_$i.log" 2>&1 &
  pids+=($!)
done

status=0
for i in "${!pids[@]}"; do
  if wait "${pids[$i]}"; then
    echo "shard $i done"
  else
    echo "shard $i FAILED -- see $SHARD_DIR/shard_$i.log" >&2
    status=1
  fi
done
[ "$status" -eq 0 ] || exit "$status"

PYTHONNOUSERSITE=1 "$PY" "$HERE/scripts/merge_shards.py" \
    --shard-dir "$SHARD_DIR" \
    --sample-submission "$SAMPLE_SUBMISSION" \
    --output "$OUTPUT"
