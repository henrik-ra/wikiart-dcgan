#!/bin/bash
# The runs of the report: the DCGAN and four modifications, each for 200 epochs, and all four combined.
# Runs one after the other; finished runs (runs/<name>/metrics.json) are skipped and interrupted ones resume.
#
#   bash run_200.sh
cd "$(dirname "$0")"
run() {
    name=$1; shift
    [ -f "runs/$name/metrics.json" ] || bash run_one.sh "$name" --epochs 200 "$@"
}
run e200
run ema200      --ema 0.999 --save-every 10
run sn200       --sn --save-every 10
run ttur200     --lr-d 4e-4 --lr-g 1e-4 --save-every 10
run diffaug200  --diffaug --save-every 10
run combo200    --sn --ema 0.999 --diffaug --lr-d 4e-4 --lr-g 1e-4
