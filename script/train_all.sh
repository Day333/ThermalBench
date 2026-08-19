#!/bin/bash
# Train all 8 models on level2/3/4 in one go.
#
#   bash script/train_all.sh              # default GPU assignment
#   OP_GPUS="4 5" TFM_GPUS=0,1,2,3 bash script/train_all.sh
#
# Layout: the 5 operator models are spread across the GPUs named in OP_GPUS, balanced by
# runtime and serial within each GPU; the three Therm-FM sizes each occupy 4 GPUs, so
# they run serially and on different GPUs from the operator models.
# Reference timings (RTX A6000, 15000 samples, 100 epochs): SAU-FNO ~6h,
# U-FNO ~3.2h, FNO ~2h, UNet <0.25h, DeepONet <0.5h, and Therm-FM T/B/L
# about half a day in total. Wall time also depends on accelerator throughput and I/O.

set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LOG="$ROOT/logs"; mkdir -p "$LOG"
OP_GPUS="${OP_GPUS:-4 5}"          # GPUs for the operator models (space separated)
TFM_GPUS="${TFM_GPUS:-0,1,2,3}"    # GPUs for Therm-FM (comma separated, must be 4)

# slow first, fast last, so the two queues finish at roughly the same time
QUEUE_A=(SAUFNO UNet DeepONet)
QUEUE_B=(UFNO FNO)

i=0
for q in QUEUE_A QUEUE_B; do
  gpu=$(echo $OP_GPUS | cut -d" " -f$((i + 1)))
  [ -n "$gpu" ] || { echo "OP_GPUS needs at least 2 GPUs"; exit 1; }
  declare -n models=$q
  (
    for m in "${models[@]}"; do
      GPUS=$gpu bash "$ROOT/script/$m/train.sh"
    done
    echo "[QUEUE-COMPLETE] gpu$gpu"
  ) > "$LOG/train_all_gpu${gpu}.log" 2>&1 &
  i=$((i + 1))
done

(
  port=29815
  for m in ThermFM-T ThermFM-B ThermFM-L; do
    GPUS="$TFM_GPUS" PORT=$port bash "$ROOT/script/$m/train.sh"
    port=$((port + 1))
  done
  echo "[QUEUE-COMPLETE] thermfm"
) > "$LOG/train_all_thermfm.log" 2>&1 &

echo "launched: operator models -> GPU $OP_GPUS, Therm-FM -> GPU $TFM_GPUS"
echo "progress: tail -f $LOG/train_all_*.log"
echo "results:  $ROOT/checkpoints/<level>_<model>/test_metrics.json"
wait
echo "ALL_TRAIN_DONE"
