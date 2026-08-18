#!/bin/bash
# Shared by every model's train.sh / test.sh / finetune.sh.
# Callers must set MODEL first, e.g.:  MODEL=UFNO; source "$(dirname "$0")/../_common.sh"
#
# Conventions:
#   Training runs on level2/3/4 only -- level5 is a pure extrapolation set and takes no
#   part in training.
#   Evaluation covers level2/3/4/5; for level5 the level4 weights are picked up
#   automatically for zero-shot prediction.
#   Fine-tuning runs on level5 only. K=0 means no fine-tuning, baseline only (the first
#   point of the curve).

set -u
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${PY:-python}"   # override with PY=/path/to/python if needed
LOG="$ROOT/logs"
RESULTS="$ROOT/results"
mkdir -p "$LOG" "$RESULTS"

TRAIN_LEVELS="${TRAIN_LEVELS:-level2 level3 level4}"
TEST_LEVELS="${TEST_LEVELS:-level2 level3 level4 level5}"
SHOTS="${SHOTS:-0 10 50 100 250 500}"

# Therm-FM needs 4-GPU DDP: batch_size=40 is defined as the total across 4 GPUs, so
# changing the GPU count changes the effective batch and the results stop being
# comparable to the benchmark. Every other model is single-GPU.
is_thermfm() { [[ "$MODEL" == ThermFM-* ]]; }
if is_thermfm; then
  GPUS="${GPUS:-0,1,2,3}"
  PORT="${PORT:-29815}"
  GPU_ARGS=(--gpus "$GPUS" --port "$PORT")
else
  GPUS="${GPUS:-0}"
  export CUDA_VISIBLE_DEVICES="$GPUS"
  GPU_ARGS=()
fi

_run() {   # _run <log name> <run.py args...>
  local tag=$1; shift
  echo "[$(date '+%F %T')] START $tag"
  if "$PY" "$ROOT/run.py" "$@" >> "$LOG/$tag.log" 2>&1; then
    echo "[$(date '+%F %T')] DONE  $tag"
  else
    echo "[$(date '+%F %T')] FAIL  $tag   log: $LOG/$tag.log"
    return 1
  fi
}

do_train() {
  local levels="${*:-$TRAIN_LEVELS}"
  for lv in $levels; do
    if [[ "$lv" == "level5" ]]; then
      echo "skipping level5: it is a pure extrapolation set and is not trained on"
      continue
    fi
    _run "${lv}_${MODEL}_train" --model "$MODEL" --data "$lv" --task train "${GPU_ARGS[@]}"
  done
}

do_test() {
  local levels="${*:-$TEST_LEVELS}"
  for lv in $levels; do
    _run "${lv}_${MODEL}_test" --model "$MODEL" --data "$lv" --task test \
         --output "$RESULTS/${lv}_${MODEL}.json" "${GPU_ARGS[@]}"
  done
}

do_finetune() {
  local shots="${*:-$SHOTS}"
  for k in $shots; do
    _run "level5_${MODEL}_ft_k${k}" --model "$MODEL" --data level5 --task finetune \
         --shots "$k" "${GPU_ARGS[@]}"
  done
}
