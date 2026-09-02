#!/bin/bash
# Train ThermFM-L (Therm-FM, single GPU). Defaults to level2/3/4; or: ./train.sh level2
# Pick a GPU with: GPUS=0 ./train.sh
MODEL=ThermFM-L
source "$(dirname "$0")/../_common.sh"
do_train "$@"
