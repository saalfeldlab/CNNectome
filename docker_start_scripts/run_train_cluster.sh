#!/usr/bin/env bash
WD=$(pwd)
RUNSCRIPT=$1

echo "Starting as user ${USER_ID}"

singularity exec \
            --nv \
	    --containall \
	    -B /groups/saalfeld,/nrs/saalfeld,/groups/turaga,/groups/cosem/cosem \
	    --pwd $WD \
	    /groups/saalfeld/home/heinrichl/singularity-builds/cnnectome.sif \
	    /bin/bash --norc -c "export OMP_NUM_THREADS=1; mprof run -C python -u $RUNSCRIPT 2>&1 | tee -a logfile"
