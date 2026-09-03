#!/usr/bin/env bash
set -euo pipefail

root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
python_bin=${PYTHON_BIN:-/home/nas_main/kyudanjung/conda_envs/envs/davianspeech/bin/python}
output_dir=${1:-"$root/output/mert_temporal_v2"}
log_dir=${2:-"$root/logs/mert_temporal_v2"}
num_shards=${NUM_SHARDS:-8}
mkdir -p "$output_dir" "$log_dir"

datasets=(
  external_mixed_train_v1 mixed_devvoice_train_v1 mixed_fmc_music_train_v1
  mixfake_music_train_v1 telephone_mixed_train_v1 temporal_mixed_train_v2
  channel_invariant_factorial_train_v1 mixfake_music_dev_v1 external_mixed_v1
  source_disjoint_mixed_v1 source_disjoint_mixed_equal_v1
  source_disjoint_music_v1 factorial_eval_1200_v2_dev telephone_mixed_dev_v1
  factorial_eval_1200_v2_holdout phone_factorial_1200_v1
  yue_cross_component_audit_v1 suno_vocals_v1
)

pids=()
for ((shard=0; shard<num_shards; shard++)); do
  CUDA_VISIBLE_DEVICES=$shard "$python_bin" "$root/scripts/extract_mert_embeddings.py" \
    --datasets "${datasets[@]}" \
    --model-root "$root/models/sofia-mert-v1" \
    --output-dir "$output_dir" \
    --shard-index "$shard" --num-shards "$num_shards" \
    --temporal-statistics --music-present-only \
    >"$log_dir/shard_$shard.log" 2>&1 &
  pids+=("$!")
done

status=0
for index in "${!pids[@]}"; do
  if wait "${pids[$index]}"; then
    echo "shard $index done"
  else
    echo "shard $index failed" >&2
    tail -n 30 "$log_dir/shard_$index.log" >&2
    status=1
  fi
done
exit "$status"
