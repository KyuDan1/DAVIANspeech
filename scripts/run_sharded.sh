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
      --xlsr-music-head "${XLSR_MUSIC_HEAD:-$HERE/model_heads/xlsr-music-head.npz}" \
      --xlsr-echoes-music-head "${XLSR_ECHOES_MUSIC_HEAD:-$HERE/model_heads/xlsr-echoes-music-head.npz}" \
      --xlsr-echofake-voice-head "${XLSR_ECHOFAKE_VOICE_HEAD:-$HERE/model_heads/xlsr-echofake-voice-head.npz}" \
      --eat-dir "${EAT_DIR:-$HERE/models/eat-base-as2m}" \
      --eat-head "${EAT_HEAD:-$HERE/model_heads/eat-music-head.npz}" \
      --eat-echoes-head "${EAT_ECHOES_HEAD:-$HERE/model_heads/eat-echoes-music-head.npz}" \
      --spear-dir "${SPEAR_DIR:-$HERE/models/spear-xlarge-speech-audio-v2}" \
      --spear-music-head "${SPEAR_MUSIC_HEAD:-$HERE/model_heads/spear-v3-music-head.npz}" \
      --spear-mixed-voice-head "${SPEAR_MIXED_VOICE_HEAD:-$HERE/model_heads/spear-mixed-voice_fake-head.npz}" \
      --spear-mixed-music-head "${SPEAR_MIXED_MUSIC_HEAD:-$HERE/model_heads/spear-mixed-music_fake-head.npz}" \
      --spear-mixture-present-head "${SPEAR_MIXTURE_PRESENT_HEAD:-$HERE/model_heads/spear-mixture-present-head.npz}" \
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
