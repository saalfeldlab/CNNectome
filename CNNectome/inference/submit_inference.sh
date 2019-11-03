#!/usr/bin/env bash
WD=$(pwd)
WDTL=$(basename $(pwd))
DP=$(basename $3)
echo "Submitting job"
bsub -J "$3-$4-$WDTL-prep" -P cosem -n $2 -o "$DP-$4-$WDTL-output.log" -e "$DP-$4-$WDTL-error.log" -We 100 ./run_inference_cluster.sh ${@}
