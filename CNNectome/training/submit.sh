bsub -J "setup01" -P cosem -n 12 -gpu "num=1" -q gpu_tesla_large -o output.log -e error.log -W 30240 ./run_train_cluster.sh unet_template.py ${@}
