#!/usr/bin/env bash

NAME=$(basename $(pwd)-prediction)
USER_ID=${UID}
GUNPOWDER_PATH=$(readlink -f $HOME/Projects/mygunpowder/gunpowder/)
TRAIN_PATH=$(readlink -f $HOME/Projects/CNNectomics/tf_training/)
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
    funkey/gunpowder:v0.3-pre4 \
    /bin/bash -c "export CUDA_VISIBLE_DEVICES='2'; PYTHONPATH=${GUNPOWDER_PATH}:\$PYTHONPATH;
    python -u ${TRAIN_PATH}/python_scripts/predict_dist.py 'grayscale_z_968_2024_sub.h5'" 2>&1 | tee -a logfile
