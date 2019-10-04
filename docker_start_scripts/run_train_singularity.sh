#!/usr/bin/env bash
WD=$(pwd)
NAME=$(basename $(pwd))
NAME=$(basename $(dirname $(pwd)))-$NAME-training
USER_ID=${UID}
GPU=${*: -1}
RUNSCRIPT=${@:1:$#-1}

echo "Starting as user ${USER_ID}"

singularity exec \
            --nv \
	    --containall \
	    -B /groups/saalfeld,/nrs/saalfeld,/groups/turaga,/groups/cosem/cosem,/nrs/cosem \
	    --pwd $WD \
	    /groups/saalfeld/home/heinrichl/singularity-builds/cnnectome.sif \
	    /bin/bash --norc -c "export CUDA_VISIBLE_DEVICES=$GPU; export OMP_NUM_THREADS=1; nvidia-smi;
			                 mprof run -T 1 -CM python -u $RUNSCRIPT 2>&1 | tee -a logfile"

