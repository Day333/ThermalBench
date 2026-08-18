#!/bin/bash
# Evaluate FNO. Defaults to level2/3/4/5; or name them:  ./test.sh level5
# level5 is a pure extrapolation set, so the level4 weights are used automatically for
# zero-shot prediction.
# Results are written to results/<level>_FNO.json
MODEL=FNO
source "$(dirname "$0")/../_common.sh"
do_test "$@"
