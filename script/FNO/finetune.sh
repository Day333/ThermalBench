#!/bin/bash
# Few-shot fine-tuning of FNO on the level5 extrapolation set.
# Defaults to K=0/10/50/100/250/500; or name them:  ./finetune.sh 10 50
# K is the number of samples per case used for fine-tuning; K=0 evaluates the baseline
# only. The learning rate defaults to one tenth of this model's own training lr --
# do not casually replace it with one shared value.
MODEL=FNO
source "$(dirname "$0")/../_common.sh"
do_finetune "$@"
