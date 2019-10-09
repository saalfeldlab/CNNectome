#!/usr/bin/env bash
WD=$(pwd)
NAME=$(basename $(pwd))
NAME=$(basename $(dirname $(pwd)))-$NAME-training
RUNSCRIPT=${@:1:$#-1}
GPU="device=${*: -1}"
USER_ID=${UID}
docker rm -f $NAME
#rm snapshots/*
echo "Starting as user ${USER_ID}"
cd /groups/turaga
cd /groups/saalfeld
cd /nrs/saalfeld
cd $WD
#neptunes5thmoon/gunpowder:v0.3-pre6-dask1 \
echo $RUNSCRIPT
docker run \
    --gpus $GPU  \
    --rm \
    -u ${USER_ID} \
    -v /groups/turaga:/groups/turaga:rshared \
    -v /groups/saalfeld:/groups/saalfeld:rshared \
    -v /groups/cosem/cosem:/groups/cosem/cosem:rshared \
    -v /nrs/cosem:/nrs/cosem:rshared \
    -v /nrs/saalfeld:/nrs/saalfeld:rshared \
    -w ${PWD} \
    -ti \
    --name ${NAME} \
    neptunes5thmoon/cnnectome:v2.0.dev5 \
    /bin/bash -c "export OMP_NUM_THREADS=1; nvidia-smi;
    mprof run -T 1 -CM python -u $RUNSCRIPT 2>&1 | tee -a logfile"