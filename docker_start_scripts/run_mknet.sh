#!/usr/bin/env bash

NAME=$(basename $(pwd))
NAME=$(basename $(dirname $(pwd)))-$NAME-mknet
USER_ID=${UID}
GUNPOWDER_PATH=$(readlink -f $HOME/Projects/git_repos/gunpowder)
TRAIN_PATH=$(readlink -f $HOME/Projects/CNNectome)
Z5_PATH=$(readlink -f $HOME/../papec/Work/my_projects/z5/bld27/python)
docker rm -f $NAME
#rm snapshots/*

echo "Starting as user ${USER_ID}"

nvidia-docker run --rm \
    -u ${USER_ID} \
    -v /groups/turaga:/groups/turaga \
    -v /groups/saalfeld:/groups/saalfeld \
    -v /nrs/saalfeld:/nrs/saalfeld \
    -w ${PWD} \
    --name ${NAME} \
    neptunes5thmoon/gunpowder:v0.3-pre6-dask1 \
    /bin/bash -c "export CUDA_VISIBLE_DEVICES=0;
    PYTHONPATH=${GUNPOWDER_PATH}:${TRAIN_PATH}:${Z5_PATH}:\$PYTHONPATH;
    python -u $1"

