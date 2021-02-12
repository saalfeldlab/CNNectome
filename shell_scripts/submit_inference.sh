#!/usr/bin/env bash
SETUPDIR=${PWD#${PWD%/*/*/*}/}
SETUP=${SETUPDIR////_}
DP=$(basename $3)
for i in {0..15}; do
    while : ; do
            running_jobs=`bjobs -noheader -o "job_name" | grep "$3-$4-$SETUP" | wc -l`
	    echo $3-$4-$SETUP
            echo "waiting for $running_jobs jobs to finish"
            [[ $running_jobs -gt 0 ]] || break
            sleep 120
    done
    complete=`python3 check_inference_complete.py ${@}`;
    echo $complete;
    if [  $complete -eq 1 ]; then
       echo ending
       exit 0
    else
echo "Submitting job"
       bsub -J "$3-$4-$SETUP-prep" -P cosem -n 2 -gpu "num=1" -q gpu_m6000  -o "$DP-$4-$SETUP-output.log" -e "$DP-$4-$SETUP-error.log" -We 100 ./run_inference_cluster.sh ${@}
    fi
done
