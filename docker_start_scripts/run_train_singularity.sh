#!/usr/bin/env bash
WD=$(pwd)
NAME=$(basename $(pwd))
NAME=$(basename $(dirname $(pwd)))-$NAME-training
USER_ID=${UID}
LARISSA_DIR=/groups/saalfeld/home/heinrichl/Projects/git_repos/
GP_PATH=$LARISSA_DIR/gunpowder/
GP_NODES_PATH=$LARISSA_DIR/gunpowder-nodes/

TRAIN_PATH=/groups/ahrens/home/bennettd/forked/cnnectome/
echo $TRAIN_PATH

echo "Starting as user ${USER_ID}"

singularity exec \
            --nv \
	    --containall \
	    -B /groups/turaga,/groups/saalfeld,/nrs/saalfeld,/groups/ahrens \
	    --pwd $WD \
	    /tmp/cnnectome.sif \
	    /bin/bash -c "export CUDA_VISIBLE_DEVICES=$2; export OMP_NUM_THREADS=1;
	    	          PYTHONPATH=${TRAIN_PATH}:${GP_PATH}:${GP_NODES_PATH}:\$PYTHONPATH; 
			  python -u $1 2>&1 | tee -a logfile"
#neptunes5thmoon/gunpowder:v0.3-pre6-dask1 \ 
#nvidia-docker run --rm \
#    -u ${USER_ID} \
#    -v /groups/turaga:/groups/turaga:rshared \
#    -v /groups/saalfeld:/groups/saalfeld:rshared \
#    -v /nrs/saalfeld:/nrs/saalfeld:rshared \
#    -w ${PWD} \
#    --name ${NAME} \
#    neptunes5thmoon/gunpowder:v1.0-dist-z5py \
#    /bin/bash -c 
#
