#!/usr/bin/env bash
WD=$(pwd)
WDTL=$(basename $(pwd))
DP=$(basename $3)
for i in {0..15}; do
    while : ; do
            running_jobs=`bjobs -noheader -o "job_name" | grep "$3-$4-$WDTL" | wc -l`
	    echo $3-$4-$WDTL
            echo "waiting for $running_jobs jobs to finish"
            [[ $running_jobs -gt 0 ]] || break
            sleep 360
    done
    complete=`python check_inference_complete.py ${@}`;
    echo $complete;
    if [  $complete -eq 1 ]; then
       echo ending
       exit 0
    else
echo "Submitting job"
       bsub -J "$3-$4-$WDTL-prep" -P cosem -n 2 -gpu "num=1" -q gpu_short  -o "$DP-$4-$WDTL-output.log" -e "$DP-$4-$WDTL-error.log" -W 100 ./run_inference_cluster.sh ${@}
    fi
done
