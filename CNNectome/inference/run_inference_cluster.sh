#!/usr/bin/env bash
WD=$(pwd)
WDTL=$(basename $(pwd))
RUNSCRIPT=${@}

echo "Starting as user ${USER_ID}"
echo "Preparing inference"
singularity exec \
            --nv \
            --containall \
            -B /groups/saalfeld,/nrs/saalfeld,/groups/turaga,/groups/cosem/cosem,/nrs/cosem \
            --pwd $WD
            /groups/saalfeld/home/heinrichl/singularity-build/cnnectome.sif \
            /bin/bash --norc -c "export OMP_NUM_THREADS=1; python -u unet_inference_template.py prepare ${@}"
echo "Running inference split over $1 jobs"

for i in {0..$1};
do
    bsub -J "$3-$4-$WDTL-$i" -P cosem -n $2 -gpu "num=1" -q gpu_any -o "$3-$4-$WDTL-$i-output.log" \
    -e "$3-$4-$WDTL-$i-error.log" -We 300 \
    singularity exec \
                --nv \
                --containall \
                -B /groups/saalfeld,/nrs/saalfeld,/groups/turaga,/groups/cosem/cosem,/nrs/cosem \
                -pwd $WD
                /groups/saalfeld/home/heinrichl/singularity-build/cnnectome.sif \
                /bin/bash --norc -c "export OMP_NUM_THREADS=1; python -u unet_inference_template.py inference $i
                ${@:2}";
done;