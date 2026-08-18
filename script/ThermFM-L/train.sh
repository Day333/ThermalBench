#!/bin/bash
# Train ThermFM-L (Therm-FM, 4-GPU DDP). Defaults to level2/3/4; or name them:  ./train.sh level2
# Pick GPUs with:  GPUS=0,1,2,3 ./train.sh
MODEL=ThermFM-L
source "$(dirname "$0")/../_common.sh"
do_train "$@"
