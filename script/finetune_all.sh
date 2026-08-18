#!/bin/bash
# Produce the level5 few-shot fine-tuning curves for all 8 models.
#
#   bash script/finetune_all.sh            # K = 0 10 50 100 250 500
#   SHOTS="0 10" bash script/finetune_all.sh
#
# Each model fine-tunes at one tenth of its own training lr (1e-4 for the operator
# family, 5e-6 for Therm-FM), not at one shared value -- Therm-FM diverges outright at
# 1e-4. See finetune_lr in MODEL_ZOO.
# Results land in checkpoints/fewshot/level5_<model>_k<K>.json with all six metrics per
# recorded epoch.

set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
GPUS="${GPUS:-0}"
TFM_GPUS="${TFM_GPUS:-$GPUS}"

for m in FNO UFNO SAUFNO UNet DeepONet; do
  GPUS="$GPUS" bash "$ROOT/script/$m/finetune.sh"
done
for m in ThermFM-T ThermFM-B ThermFM-L; do
  GPUS="$TFM_GPUS" bash "$ROOT/script/$m/finetune.sh"
done
echo "ALL_FINETUNE_DONE"
