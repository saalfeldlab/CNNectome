#!/usr/bin/env bash
WD = $(pwd)
RUNSCRIPT=${@}

echo "Starting as user ${USER_ID}"
echo "Preparing inference"
singularity exec \
            --nv \
            --containall \
            -B /groups/saalfeld,/nrs/saalfeld,/groups/turaga,/groups/cosem/cosem,/nrs/cosem \
            --pwd $WD
            /groups/saalfeld/home/heinrichl/singularity-build/cnnectome.sif \
            /bin/bash --norc -c "export OMP_NUM_THREADS=1; python -u unet_inference_template.py ${@}"

for i in {0..$1}
do
    bsub singularity exec blabla
done;