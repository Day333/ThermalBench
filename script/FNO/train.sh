#!/bin/bash
# Train FNO (single GPU). Defaults to level2/3/4; or name them:  ./train.sh level2
# Pick GPUs with:  GPUS=3 ./train.sh
MODEL=FNO
source "$(dirname "$0")/../_common.sh"
do_train "$@"
