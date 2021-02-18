#!/usr/bin/env bash
WD=$(pwd)
SETUPDIR=${PWD#${PWD%/*/*/*}/}
SETUP=${SETUPDIR////_}
DP=$(basename $3)
RUNSCRIPT=${@}
echo $RUNSCRIPT
echo "Starting as user ${USER_ID}"
echo "Preparing inference"
singularity exec \
            --nv \
            --containall \
            -B /scratch/$USER:/tmp,/groups/saalfeld,/nrs/saalfeld,/groups/turaga,/groups/cosem/cosem,/nrs/cosem \
            --pwd $WD \
            /groups/saalfeld/home/heinrichl/singularity-builds/cnnectome.sif \
            /bin/bash --norc -c "export OMP_NUM_THREADS=1; umask 0002; python -u inference_config.py prepare $RUNSCRIPT"
echo "Running inference split over $1 jobs"

for i in $(seq 0 `expr $1 - 1`);
do
    bsub -J "$3-$4-$SETUP-$i" -P cosem -n 2 -gpu "num=1" -q gpu_short -o "$DP-$4-$SETUP-$i-output.log" \
    -e "$DP-$4-$SETUP-$i-error.log" -W 240 \
    singularity exec \
                --nv \
                --containall \
                -B /scratch/$USER:/tmp,/groups/saalfeld,/nrs/saalfeld,/groups/turaga,/groups/cosem/cosem,/nrs/cosem \
                --pwd $WD \
                /groups/saalfeld/home/heinrichl/singularity-builds/cnnectome.sif \
                /bin/bash --norc -c "export OMP_NUM_THREADS=1; umask 0002; python -u inference_config.py inference $i ${*:2} ";
done;
