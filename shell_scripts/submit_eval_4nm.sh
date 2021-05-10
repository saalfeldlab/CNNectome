#!/usr/bin/env bash

# input arguments: num_gpus_per_inference_job iteration db_username db_password
set -e

if [ $1 -gt 20 ]; then
  echo "Too many gpus, dont' do that"
  exit 1
fi

./submit_inference.sh $1 2 jrc_hela-2 $2 --mask_ds volumes/masks/validation/0003
./submit_inference.sh $1 2 jrc_hela-3 $2 --mask_ds volumes/masks/validation/0003
./submit_inference.sh $1 2 jrc_macrophage-2 $2 --mask_ds volumes/masks/validation/0003
./submit_inference.sh $1 2 jrc_jurkat-2 $2 --mask_ds volumes/masks/validation/0003

completehela2=`check_inference_complete $1 2 jrc_hela-2 $2 --mask_ds volumes/masks/validation/0003`
completehela3=`check_inference_complete $1 2 jrc_hela-3 $2 --mask_ds volumes/masks/validation/0003`
completemac=`check_inference_complete $1 2 jrc_macrophage-2 $2 --mask_ds volumes/masks/validation/0003`
completejurkat=`check_inference_complete $1 2 jrc_jurkat-2 $2 --mask_ds volumes/masks/validation/0003`

if [ $completehela2 -eq 1 ] && [ $completehela3 -eq 1 ] && [ $completemac -eq 1 ] && [ $completejurkat -eq 1 ]; then
  echo "Submitting evaluation"
  WD=$(pwd)
  SETUPDIR=${PWD#${PWD%/*/*}/}
  SETUP=${SETUPDIR////_}

  bsub -J "$SETUP-$2-eval" -P cosem -n 5 -o "$SETUP-$2-eval-output.log" -e "$SETUP-$2-eval-error.log" -W 300 \
    singularity exec \
                --containall \
                -B /scratch/$USER:/tmp,/groups/saalfeld,/nrs/saalfeld,/groups/turaga,/groups/cosem/cosem,/nrs/cosem \
                --pwd $PWD \
                /groups/saalfeld/home/heinrichl/singularity-builds/cnnectome.sif \
                /bin/bash --norc -c "export OMP_NUM_THREADS=1; umask 0002; auto_evaluation $2 $3 $4";
  exit 0
else
  COUNTER=${5-0}
  if [ $COUNTER -eq 3 ]; then
    echo "Failed 3 times, something's probably going wrong. Exiting"
    exit 1
  else
    echo "Failed, evaluation not submitted. Resubmitting inference."
    ./$0 $1 $2 $3 $4 $COUNTER
  fi
fi
