#!/bin/bash
# Evaluate all 8 models on level2/3/4/5 and print a summary table.
#
#   bash script/test_all.sh           # evaluate using existing weights
#   GPUS=6 TFM_GPUS=7 bash script/test_all.sh
#
# Evaluation only, no training, so this is much faster than a training sweep (about an
# hour end to end). For level5 the level4 weights are picked up automatically for
# zero-shot prediction.

set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PY="${PY:-python}"   # override with PY=/path/to/python if needed
GPUS="${GPUS:-0}"
TFM_GPUS="${TFM_GPUS:-$GPUS}"

for m in FNO UFNO SAUFNO UNet DeepONet; do
  GPUS="$GPUS" bash "$ROOT/script/$m/test.sh"
done
for m in ThermFM-T ThermFM-B ThermFM-L; do
  GPUS="$TFM_GPUS" bash "$ROOT/script/$m/test.sh"
done

echo
"$PY" "$ROOT/utils/summarize.py"
