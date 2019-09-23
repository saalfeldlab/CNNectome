#!/usr/bin/env bash
WD=$(pwd)
NAME=$(basename $(pwd))
NAME=$(basename $(dirname $(pwd)))-$NAME-training
USER_ID=${UID}
LARISSA_DIR=/groups/saalfeld/home/heinrichl/Projects/git_repos/
GP_PATH=$LARISSA_DIR/gunpowder/
GP_NODES_PATH=$LARISSA_DIR/gunpowder-nodes/
GPU=$2
TRAIN_PATH=/groups/ahrens/home/bennettd/forked/cnnectome/
RUNSCRIPT=$1

echo $TRAIN_PATH


echo "Starting as user ${USER_ID}"

singularity exec \
            --nv \
	    --containall \
	    -B /groups/ahrens/,/groups/saalfeld,/nrs/saalfeld \
	    --pwd $WD \
	    docker://bennettd/cnnectome:latest \
	    /bin/bash --norc -c "export CUDA_VISIBLE_DEVICES=$GPU; export OMP_NUM_THREADS=1;
	    	          PYTHONPATH=${TRAIN_PATH}:${GP_PATH}:${GP_NODES_PATH}:\$PYTHONPATH; 
			  python -u $RUNSCRIPT 2>&1 | tee -a logfile"

