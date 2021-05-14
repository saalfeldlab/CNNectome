# CNNectome

CNNectome is a collection of scripts for building, training and validating Convolutional Neural Networks (CNNs) for
*Connectomics*. It relies on [gunpowder](https://github.com/funkey/gunpowder) and 
[Tensorflow](https://github.com/tensorflow/tensorflow).

## Setup

### Docker

Docker images can be built with the Dockerfile in `docker/` or downloaded from 
[this dockerhub](https://hub.docker.com/r/neptunes5thmoon/cnnectome).

### Git + Pip

Distribution via pypi will be available soon, but for now you can install the repo by cloning it and then installing via 
`pip install .` 

### Data

Many parts of this repo depend on specific data for organelle and synapse segmentations as laid out below. In order to 
run data-dependent parts of the pipeline you will have to download the relevant data and configure the repo to point 
to your download directories.
An example config file is included in `etc/config_local.ini`. By calling `init_CNNectome_config` (or calling any script 
trying to access the configuration) the example config file will be copied to the system-dependent default location 
for config files (if it doesn't yet exist).  

## Organelle Segmentation

**Preprint:** Larissa Heinrich, Davis Bennett, David Ackerman, Woohyun Park, John Bogovic, Nils Eckstein, 
Alyson Petruncio, Jody Clements, C. Shan Xu, Jan Funke, Wyatt Korff, Harald F. Hess, Jennifer Lippincott-Schwartz, 
Stephan Saalfeld, Aubrey V. Weigel, COSEM Project Team. *Automatic whole cell organelle segmentation in volumetric 
electron microscopy.* bioRxiv (2020) 10.1101/2020.11.14.382143

Find links to all the code involved in this poject at [github.com/janelia-cosem/heinrich-2021a](https://github.com/janelia-cosem/heinrich-2021a).

### Training

The training data for organelle segmentation can be downloaded from [s3://janelia-cosem-publications/heinrich-2021a/](https://open.quiltdata.com/b/janelia-cosem-publications/tree/heinrich-2021a/) 
 - specifically you need the raw data and groundtruth blocks for `jrc_hela-2`, `jrc_hela-3`, 
`jrc_jurkat-1`, `jrc_macrophage-2` and `jrc_sum159-1`. Their directories should be located in ***data_path***. Further, 
some of the metadata for the groundtruth blocks is organized in a mongo database. Since you only need read access here, 
you can use the public facing instance of our database 
(`mongodb+srv://cosemRead:xYF1fCyVHrIfujKm@cosem-public.5a8vu.mongodb.net`, configurable via ***database-public***).

The file `CNNectome/training/unet_template.py` can serve as a starting point for setting up your own trainings. 
The setups from the paper can be found in a separate repo: 
[github.com/janelia-cosem/training_setups](https://github.com/janelia-cosem/training_setups) with all checkpoints 
used to reconstruct the cells on [openorganelle](https://openorganelle.janelia.org) in a corresponding [s3 bucket](s3://janelia-cosem-networks).
These can be distributed across several directories, configured via a comma-separated list in ***training_setups_paths***.

### Evaluation

All evaluations presented in the paper (and more) are also saved in the [same database](mongodb+srv://cosemRead:xYF1fCyVHrIfujKm@cosem-public.5a8vu.mongodb.net) as the metadata for 
groundtruth blocks (***database-public***). To add evaluations of your own 
(`CNNectome/validation/organelles/run_evaluation.py`) you'll need your own mongodb instance to which 
you have write access; this is configurable via ***database-private***. Of course you'll also need the validation crops (which are part of the groundtruth) in ***data_path*** as well as 
their metadata in ***database-public***. 

In order to reproduce the comparisons presented in the paper, in addition to ***database-public*** and 
***training_setups_paths***, you'll need to download the results of the manual evaluation procedures from 
the s3 bucket: [s3://janelia-cosem-publications/heinrich-2021a/evaluations](https://open.quiltdata.com/b/janelia-cosem-publications/tree/heinrich-2021a/evaluations). ***evaluation_path*** should point 
directly to that directory. \
The scripts for running and visualizing the comparisons are in 
`CNNectome/validation/organelles/comparisons.py` and `CNNectome/visualization/organelles/plot_comparisons.py`, 
respectively.

*coming soon*: find refined segmentations in ***data_path*** instead of separate ***refined_seg_path***.

###Other notes

The names used for organelle classes in this repo and in the paper differ. The mapping can be found in 
`CNNectome/utils/hierarchy.py`.

The preprocessing for the raw data has been improved since these setups have been trained. If you start training from 
scratch consider using the raw data linked on openorganelle.  

## Synapse Segmentation

*Instructions coming soon*




