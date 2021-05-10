#!/usr/bin/env bash
WD=$(pwd)
SETUPDIR=${PWD#${PWD%/*}/}
SETUP=${SETUPDIR////_}
RUNSCRIPT=${@}
echo $RUNSCRIPT
for i in {0..15}; do
    while : ; do
        running_jobs=`bjobs -noheader -o "job_name" | grep "$3-$4-$SETUP" | wc -l`
	echo $3-$4-$SETUP
        echo "waiting for $running_jobs jobs to finish"
        [[ $running_jobs -gt 0 ]] || break
        sleep 60
    done
    complete=`check_inference_complete ${@}`;
    echo $complete;
    if [  $complete -eq 1 ]; then
        echo ending
        exit 0
    else
        echo "Submitting job"
        bsub -J "$3-$4-$SETUP-prep" -P cosem -n 2 -gpu "num=1" -q gpu_short  -o "$3-$4-$SETUP-output.log" -e "$3-$4-$SETUP-error.log" -We 240 \
          singularity exec \
                     --nv \
                     --containall \
                     -B /scratch/$USER:/tmp,/groups/saalfeld,/nrs/saalfeld,/groups/turaga,/groups/cosem/cosem,/nrs/cosem \
                     --pwd $WD \
                     /groups/saalfeld/home/heinrichl/singularity-builds/cnnectome.sif \
                     /bin/bash --norc -c "export OMP_NUM_THREADS=1; umask 0002; unet_inference prepare $RUNSCRIPT ";
        echo "Running inference split over $1 jobs"
        prepjobid=`bjobs -J "$3-$4-$SETUP-prep" -o "jobid" -noheader`
        for i in $(seq 0 `expr $1 - 1`); do
            bsub -w $prepjobid -J "$3-$4-$SETUP-$i" -P cosem -n 2 -gpu "num=1" -q gpu_short -o "$3-$4-$SETUP-$i-output.log" -e "$3-$4-$SETUP-$i-error.log" -W 240 \
              singularity exec \
                        --nv \
                        --containall \
                        -B /scratch/$USER:/tmp,/groups/saalfeld,/nrs/saalfeld,/groups/turaga,/groups/cosem/cosem,/nrs/cosem \
                        --pwd $WD \
                        /groups/saalfeld/home/heinrichl/singularity-builds/cnnectome.sif \
                        /bin/bash --norc -c "export OMP_NUM_THREADS=1; umask 0002; unet_inference inference $i ${*:2} ";
        done;
    fi
done
