#!/usr/bin/env bash

NAME=$(basename $(pwd)-convert)
USER_ID=${UID}
GUNPOWDER_PATH=$(readlink -f $HOME/Projects/mygunpowder/gunpowder)
SRC_PATH=$(readlink -f $HOME/Projects/gunpowder_training/utils/)
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
    funkey/gunpowder:v0.3-pre5 \
    /bin/bash -c "export CUDA_VISIBLE_DEVICES=0; PYTHONPATH=${GUNPOWDER_PATH}:${TRAIN_PATH}:\$PYTHONPATH;
    python -u ${SRC_PATH}/saved_model_api_convert.py"