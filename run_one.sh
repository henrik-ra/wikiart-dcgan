#!/bin/bash
# One experiment: train -> FID over all generator snapshots (2k images) -> final IS/FID (10k images).
# Running the same command again after an interruption continues from runs/<name>/last.pt.
#
#   bash run_one.sh <name> [train.py options]
#   PYTHON=/path/to/python bash run_one.sh ...     # default: python
cd "$(dirname "$0")"
name=$1; shift
PY=${PYTHON:-python}
log() { echo "[$(date '+%H:%M:%S')] $*" >> runs/queue.log; }
mkdir -p "runs/$name"
log "start $name: $*"
t0=$(date +%s)
if $PY train.py --name "$name" --save-every 5 --resume "$@" >> "runs/$name/train.log" 2>&1 &&
   $PY evaluate.py --run "$name" --all --n 2000 > "runs/$name/eval_all.log" 2>&1 &&
   $PY evaluate.py --run "$name" --n 10000 > "runs/$name/eval.log" 2>&1; then
    log "done $name in $(( ($(date +%s) - t0) / 60 )) min: $(grep '^{' "runs/$name/eval.log" | tr '\n' ' ' | cut -c1-300)"
else
    log "FAILED $name, see runs/$name/*.log"
fi
