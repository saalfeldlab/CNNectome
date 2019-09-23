#!/usr/bin/env bash
WD=$(pwd)
NAME=$(basename $(pwd))
NAME=$(basename $(dirname $(pwd)))-$NAME-training
TRAIN_PATH=$(readlink -f $HOME/dev/CNNectome/)
GP_PATH=$(readlink -f $HOME/dev/gunpowder/)
USER_ID=${UID}
GPU=$2
RUNSCRIPT=$1

echo $TRAIN_PATH


echo "Starting as user ${USER_ID}"

singularity exec \
            --nv \
	    --containall \
	    -B /groups/saalfeld,/nrs/saalfeld,/groups/turaga,/groups/cosem/cosem \
	    --pwd $WD \
	    /groups/saalfeld/home/heinrichl/singularity-builds/cnnectome.sif \
	    /bin/bash --norc -c "export CUDA_VISIBLE_DEVICES=$GPU; export OMP_NUM_THREADS=1;
	    	                 PYTHONPATH=${TRAIN_PATH}:${GP_PATH}:\$PYTHONPATH;
			                 mprof run -CM python -u $RUNSCRIPT 2>&1 | tee -a logfile"

