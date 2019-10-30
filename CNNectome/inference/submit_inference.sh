#!/usr/bin/env bash
WD=$(pwd)
WDTL=$(basename $(pwd))

bsub -J "$3-$4-$WDTL-prep" -P cosem -n $2 -o "$3-$4-$WDTL-output.log" -e "$3-$4-$WDTL-error.log" -We 100
./run_inference_cluster.sh ${@}
