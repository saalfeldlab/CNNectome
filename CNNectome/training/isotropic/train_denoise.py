from gunpowder import *
from gunpowder.tensorflow import *
from gunpowder.contrib import IntensityCrop, ImpulseNoiseAugment
from gunpowder.ext import zarr
from gunpowder.compat import ensure_str

import fuse
import tensorflow as tf
import math
import time
import itertools
import json
import os
import numpy as np
import logging
import pymongo



def network_setup(net_name, trainable=True):
    # load graph's io names
    with open("{0:}_io_names.json".format(net_name), "r") as f:
        net_io_names = json.load(f)
    if trainable:
        # find checkpoint from previous training, start a new one if not found
        if tf.train.latest_checkpoint("."):
            start_iteration = int(tf.train.latest_checkpoint(".").split("_")[-1])
            if start_iteration >= max_iteration:
                logging.info("Network has already been trained for {0:} iterations".format(start_iteration))
            else:
                logging.info("Resuming training from {0:}".format(start_iteration))
        else:
            start_iteration = 0
            logging.info("Starting fresh training")
    else:
        start_iteration=None
    # define network inputs
    inputs = dict()
    inputs[net_io_names["raw_input"]] = ArrayKey("RAW_INPUT")
    inputs[net_io_names["raw_target"]] = ArrayKey("RAW_TARGET")
    # define network outputs
    outputs = dict()
    outputs[net_io_names["raw_predicted"]] = ArrayKey("RAW_PREDICTED")
    return net_io_names, start_iteration, inputs, outputs


def batch_generator(
        gt_version,
        input_shape,
        output_shape,
        target_matches,
        db_username,
        db_password,
        augmentations,
        db_name="crops",
        completion_min=6,
        data_path=None,
        intensity_scale_range=(0.75, 1.25),
        intensity_shift_range=(-0.2, 0.2),
        gamma_range=(0.75, 4 / 3.),
        impulse_noise_prob=0.05,
        prob_missing=0.05,
        prob_low_contrast=0.05,
        contrast_scale=0.1,
        min_masked_voxels=17561.,
        voxel_size=Coordinate((8, 8, 8)),
):
    def make_variant_source(n5_raw, subsample_variants):
        n5file = zarr.open(n5_raw, mode='r')
        mask_ds_foreground = "volumes/masks/foreground"
        mask_ds_training = "volumes/masks/training"

        # add sources for all groundtruth labels
        voxel_size_mask_attr = Coordinate(n5file[mask_ds_foreground].attrs["pixelResolution"]["dimensions"])
        # add raw source
        logging.debug("Adding ZarrSource ({file:}/{ds:}) providing {ak_inp}, {ak_tar}, {ak_mask_fg}".format(
            file=n5_raw, ds=subsample_variants[0], ak_inp=ArrayKey("RAW_INPUT"), ak_tar=ArrayKey("RAW_TARGET"),
            ak_mask_fg=ArrayKey("MASK_FOREGROUND")))

        if voxel_size_mask_attr != voxel_size_mask:
            src = (
                ZarrSource(
                    n5_raw,
                    {ArrayKey("RAW_INPUT"): subsample_variants[0],
                     ArrayKey("RAW_TARGET"): subsample_variants[1],
                     ArrayKey("MASK_UP"): mask_ds_foreground,
                     ArrayKey("MASK_TRAINING"): mask_ds_training
                     },
                    array_specs={ArrayKey("RAW_INPUT"):  ArraySpec(voxel_size=voxel_size),
                                 ArrayKey("RAW_TARGET"): ArraySpec(voxel_size=voxel_size),
                                 ArrayKey("MASK_UP"):    ArraySpec(voxel_size=voxel_size_mask_attr, interpolatable=False),
                                 ArrayKey("MASK_TRAINING"): ArraySpec(voxel_size=voxel_size, interpolatable=False)
                                 }
                )
            )
            src += DownSample(ArrayKey("MASK_UP"), voxel_size_mask/voxel_size_mask_attr, ArrayKey("MASK_FOREGROUND"))
        else:
            src = (
                ZarrSource(
                    n5_raw,
                    {ArrayKey("RAW_INPUT"): subsample_variants[0],
                     ArrayKey("RAW_TARGET"): subsample_variants[1],
                     ArrayKey("MASK_FOREGROUND"): mask_ds_foreground,
                     ArrayKey("MASK_TRAINING"): mask_ds_training
                     },
                    array_specs={ArrayKey("RAW_INPUT"):  ArraySpec(voxel_size=voxel_size),
                                 ArrayKey("RAW_TARGET"): ArraySpec(voxel_size=voxel_size),
                                 ArrayKey("MASK_FOREGROUND"): ArraySpec(voxel_size=voxel_size_mask_attr, interpolatable=False),
                                 ArrayKey("MASK_TRAINING"): ArraySpec(voxel_size=voxel_size, interpolatable=False)
                                 }
                )
            )
        # combine all sources and pick a random location
        crop_src = (
            src
            + RandomLocation()
            + Reject(ArrayKey("MASK_FOREGROUND"), min_masked=keep_thr, reject_probability=0.95)
            + Reject(ArrayKey("MASK_TRAINING"), min_masked=1.))

        # contrast adjustment
        contr_adj = n5file["volumes/raw/s0"].attrs["contrastAdjustment"]
        scale = 255.0 / (float(contr_adj["max"]) - float(contr_adj["min"]))
        shift = - scale * float(contr_adj["min"])
        logging.debug("Adjusting contrast with scale {scale:} and shift {shift:}".format(scale=scale, shift=shift))
        crop_src += IntensityScaleShift(ArrayKey("RAW_INPUT"),
                                        scale,
                                        shift
                                        )
        crop_src += IntensityScaleShift(ArrayKey("RAW_TARGET"),
                                        scale,
                                        shift)

        return crop_src



    keep_thr = float(min_masked_voxels) / np.prod(output_shape)

    # ArrayKey("RAW_INPUT")
    # ArrayKey("RAW_TARGET")
    # ArrayKey("RAW_PREDICTED")
    # ArrayKey("MASK_FOREGROUND")
    # ArrayKey("MASK_TRAINING")
    # ArrayKey("MASK_UP")
    voxel_size_mask = Coordinate((16, 16, 16))
    input_size = Coordinate(input_shape) * voxel_size
    output_size = Coordinate(output_shape) * voxel_size

    client = pymongo.MongoClient("cosem.int.janelia.org:27017", username=db_username, password=db_password)
    db = client[db_name]
    collection = db[gt_version]
    filter = {"completion": {"$gte": completion_min}}
    skip = {"_id": 0, "parent":1}
    raw_datasets = set()
    for crop in collection.find(filter, skip):
        raw_datasets.add(crop["parent"])

    raw_datasets = list(raw_datasets)
    prefix = os.path.commonprefix(raw_datasets)

    if data_path is not None:
        for k, raw_ds in enumerate(raw_datasets):
            raw_datasets[k] = raw_ds.replace(prefix, data_path)



    # construct batch request
    request = BatchRequest()
    request.add(ArrayKey("RAW_INPUT"), input_size, voxel_size=voxel_size)
    request.add(ArrayKey("RAW_TARGET"), output_size, voxel_size=voxel_size)
    request.add(ArrayKey("RAW_PREDICTED"), output_size, voxel_size=voxel_size)
    request.add(ArrayKey("MASK_FOREGROUND"), output_size, voxel_size=voxel_size_mask)
    request.add(ArrayKey("MASK_TRAINING"), output_size, voxel_size=voxel_size)

    # specify specs for output
    array_specs_pred = {ArrayKey("RAW_PREDICTED"): ArraySpec(voxel_size=voxel_size, interpolatable=True)}


    srcs = []
    for raw_ds in raw_datasets:
        for subsample_variants in target_matches:
            srcs.append(make_variant_source(raw_ds, subsample_variants))

    pipeline = (tuple(srcs) + RandomProvider())

    pipeline += Normalize(ArrayKey("RAW_INPUT"), 1.0/255)
    pipeline += Normalize(ArrayKey("RAW_TARGET"), 1.0/255)
    pipeline += IntensityCrop(ArrayKey("RAW_INPUT"), 0., 1.)
    pipeline += IntensityCrop(ArrayKey("RAW_TARGET"), 0., 1.)

    # augmentations
    for aug in augmentations:
        if aug == "simple":
            pipeline += fuse.SimpleAugment()
        elif aug == "elastic":
            pipeline += fuse.ElasticAugment(voxel_size,
                                            (100, 100, 100),
                                            (10., 10., 10.),
                                            (0, math.pi / 2.),
                                            spatial_dims=3,
                                            subsample=8
                                            )
        # elif aug == "intensity_narrow" or aug == "intensity":
        #     pipeline += fuse.IntensityAugment([ArrayKey("RAW_INPUT"), ArrayKey("RAW_TARGET")], 0.75, 1.25, -0.2, 0.2)
        #     pipeline += GammaAugment([ArrayKey("RAW_INPUT"), ArrayKey("RAW_TARGET")], 0.75, 4/3.)
        # elif aug == "intensity_wide":
        #     pipeline += fuse.IntensityAugment([ArrayKey("RAW_INPUT"), ArrayKey("RAW_TARGET")], 0.25, 1.75, -0.5, 0.35)
        #     pipeline += GammaAugment([ArrayKey("RAW_INPUT"), ArrayKey("RAW_TARGET")], 0.5, 2.)
        elif aug == "intensity":
            pipeline += fuse.IntensityAugment(
                [ArrayKey("RAW_INPUT"), ArrayKey("RAW_TARGET")],
                intensity_scale_range[0],
                intensity_scale_range[1],
                intensity_shift_range[0],
                intensity_shift_range[1]
            )
        elif aug == "gamma":
            pipeline += GammaAugment([ArrayKey("RAW_INPUT"), ArrayKey("RAW_TARGET")], gamma_range[0], gamma_range[1])
        elif aug == "poisson":
            pipeline += NoiseAugment(ArrayKey("RAW_INPUT"), 'poisson', clip=True)
        elif aug == "impulse_noise":
            pipeline += ImpulseNoiseAugment(ArrayKey("RAW_INPUT"), impulse_noise_prob)
        elif aug == "defect":
            pipeline += DefectAugment(
                ArrayKey("RAW_INPUT"),
                prob_missing=prob_missing,
                prob_low_contrast=prob_low_contrast,
                contrast_scale=contrast_scale
            )
        else:
            raise ValueError("")

    pipeline += IntensityScaleShift(ArrayKey("RAW_INPUT"), 2, -1)
    pipeline += IntensityScaleShift(ArrayKey("RAW_TARGET"), 2, -1)
    return pipeline, request, array_specs_pred


def train_until(
    max_iteration,
    gt_version,
    net_name,
    input_shape,
    output_shape,
    loss_name,
    target_matches,
    db_username,
    db_password,
    augmentations,
    db_name="crops",
    completion_min=6,
    data_path=None,
    cache_size=5,
    num_workers=10,
    intensity_scale_range=(0.75, 1.25),
    intensity_shift_range=(-0.2, 0.2),
    gamma_range=(0.75, 4 / 3.),
    impulse_noise_prob=0.05,
    prob_missing=0.05,
    prob_low_contrast=0.05,
    contrast_scale=0.1,
    min_masked_voxels=17561.,
    voxel_size=Coordinate((8, 8, 8)),
):

    pipeline, request, array_specs_pred = batch_generator(
        gt_version,
        input_shape,
        output_shape,
        target_matches,
        db_username,
        db_password,
        augmentations,
        db_name=db_name,
        completion_min=completion_min,
        data_path=data_path,
        intensity_scale_range=intensity_scale_range,
        intensity_shift_range=intensity_shift_range,
        gamma_range=gamma_range,
        impulse_noise_prob=impulse_noise_prob,
        prob_missing=prob_missing,
        prob_low_contrast=prob_low_contrast,
        contrast_scale=contrast_scale,
        min_masked_voxels=min_masked_voxels,
        voxel_size=voxel_size)
    net_io_names, start_iteration, inputs, outputs = network_setup(net_name)
    # specify snapshot data layout
    snapshot_data = dict()
    snapshot_data[ArrayKey("RAW_INPUT")] = "volumes/raw_input"
    snapshot_data[ArrayKey("MASK_FOREGROUND")] = "volumes/masks/all"
    snapshot_data[ArrayKey("RAW_TARGET")] = "volumes/raw_target"
    snapshot_data[ArrayKey("RAW_PREDICTED")] = "volumes/raw_predicted"

    # specify snapshot request
    snapshot_request = BatchRequest()
    pipeline = (pipeline
                + PreCache(cache_size=cache_size,
                           num_workers=num_workers)
                + Train(net_name,
                        optimizer=net_io_names["optimizer"],
                        loss=net_io_names["loss"],
                        inputs=inputs,
                        summary=net_io_names["summary"],
                        log_dir="log",
                        outputs=outputs,
                        gradients={},
                        log_every=10,
                        save_every=500,
                        array_specs=array_specs_pred
                        )
                + Snapshot(snapshot_data,
                           every=500,
                           output_filename="batch_{iteration}.hdf",
                           output_dir="snapshots/",
                           additional_request=snapshot_request,
                           )
                + PrintProfilingStats(every=50)
                )

    logging.info("Starting training...")
    with build(pipeline) as pp:
        for i in range(start_iteration, max_iteration+1):
            start_it = time.time()
            pp.request_batch(request)
            time_it = time.time() - start_it
            logging.info("it{0:}: {1:}".format(i+1, time_it))
        pp.internal_teardown()
    logging.info("Training finished")


def evaluate_blur(
        iterations,
        gt_version,
        net_name,
        input_shape,
        output_shape,
        loss_name,
        target_matches,
        db_username,
        db_password,
        augmentations,
        db_name="crops",
        completion_min=6,
        data_path=None,
        cache_size=5,
        num_workers=10,
        intensity_scale_range=(0.75, 1.25),
        intensity_shift_range=(-0.2, 0.2),
        gamma_range=(0.75, 4 / 3.),
        impulse_noise_prob=0.05,
        prob_missing=0.05,
        prob_low_contrast=0.05,
        contrast_scale=0.1,
        min_masked_voxels=17561.,
        voxel_size=Coordinate((8, 8, 8)),
):
    pipeline, request, array_specs_pred = batch_generator(
        gt_version,
        input_shape,
        output_shape,
        target_matches,
        db_username,
        db_password,
        augmentations,
        db_name=db_name,
        completion_min=completion_min,
        data_path=data_path,
        intensity_scale_range=intensity_scale_range,
        intensity_shift_range=intensity_shift_range,
        gamma_range=gamma_range,
        impulse_noise_prob=impulse_noise_prob,
        prob_missing=prob_missing,
        prob_low_contrast=prob_low_contrast,
        contrast_scale=contrast_scale,
        min_masked_voxels=min_masked_voxels,
        voxel_size=voxel_size)
    net_io_names, _, inputs, outputs = network_setup(net_name, trainable=False)
    ak_loss = ArrayKey(loss_name.upper())
    outputs[net_io_names["loss_bc"]] = ak_loss

    request.add(ak_loss, voxel_size, voxel_size=voxel_size)
    pipeline = (pipeline
                + PreCache(cache_size=cache_size,
                           num_workers=num_workers)
                + Run(net_name+'.meta',
                      inputs=inputs,
                      outputs=outputs,
                      array_specs=array_specs_pred,
                      max_shared_memory=int(1024*1024*1024))
                + PrintProfilingStats(every=100))
    logging.info("Starting evaluation...")
    costs = []
    with build(pipeline) as pp:
        for i in range(iterations):
            start_it = time.time()
            b = pp.request_batch(request)
            costs.append(float(np.squeeze(b.arrays[ak_loss].data)))
            time_it = time.time() - start_it
            logging.info("it{0:}: {1:}".format(i+1, time_it))
    logging.info("Evaluation finished")
    return costs