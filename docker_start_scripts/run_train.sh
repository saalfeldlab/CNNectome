#!/usr/bin/env bash
WD=$(pwd)
NAME=$(basename $(pwd))
NAME=$(basename $(dirname $(pwd)))-$NAME-training
USER_ID=${UID}
GUNPOWDER_PATH=$(readlink -f $HOME/Projects/git_repos/gunpowder/)
GUNPOWDER_NODES_PATH=$(readlink -f $HOME/Projects/git_repos/gunpowder-nodes/)
TRAIN_PATH=$(readlink -f $HOME/Projects/CNNectome/)
Z5_PATH=/groups/saalfeld/home/papec/Work/my_projects/z5/bld27/python/
#Z5_PATH=/groups/saalfeld/home/heinrichl/Brew/z5/bld27/python/
docker rm -f $NAME
#rm snapshots/*
echo "Starting as user ${USER_ID}"
cd /groups/turaga
cd /groups/saalfeld
cd /nrs/saalfeld
cd $WD
#neptunes5thmoon/gunpowder:v0.3-pre6-dask1 \
nvidia-docker run --rm \
    -u ${USER_ID} \
    -v /groups/turaga:/groups/turaga:rshared \
    -v /groups/saalfeld:/groups/saalfeld:rshared \
    -v /nrs/saalfeld:/nrs/saalfeld:rshared \
    -w ${PWD} \
    --name ${NAME} \
    neptunes5thmoon/gunpowder:v1.0-dist-z5py \
    /bin/bash -c "export CUDA_VISIBLE_DEVICES=$2; export OMP_NUM_THREADS=1;
    PYTHONPATH=${TRAIN_PATH}:${GUNPOWDER_PATH}:${GUNPOWDER_NODES_PATH}:\$PYTHONPATH;
    python -u $1 2>&1 | tee -a logfile"
