from gunpowder import *
from gunpowder.tensorflow import *
from gunpowder.contrib import (
    ZeroOutConstSections,
    AddDistance,
    AddPrePostCleftDistance,
    TanhSaturate,
)
import fuse
import tensorflow as tf
import os
import math
import json
import csv
import logging
import numpy as np
import time


def make_cleft_to_prepostsyn_neuron_id_dict(csv_files, filter_comments_pre, filter_comments_post):
    cleft_to_pre = dict()
    cleft_to_post = dict()
    for csv_f in csv_files:
        logging.info("Reading csv file {0:}".format(csv_f))
        f = open(csv_f, "r")
        reader = csv.DictReader(
            f,
            fieldnames=[
                "pre_label",
                "pre_id",
                "pre_x",
                "pre_y",
                "pre_z",
                "pre_comment",
                "post_label",
                "post_id",
                "post_x",
                "post_y",
                "post_z",
                "post_comment",
                "cleft",
            ],
        )
        next(reader)
        for row in reader:
            if int(row["cleft"]) > 0:
                for fc in filter_comments_pre:
                    if fc not in row["pre_comment"] and fc not in row["post_comment"]:
                        if int(row["pre_label"]) > 0:
                            try:
                                cleft_to_pre[int(row["cleft"])].add(int(row["pre_label"]))
                            except KeyError:
                                cleft_to_pre[int(row["cleft"])] = {int(row["pre_label"])}
                for fc in filter_comments_post:
                    if fc not in row["pre_comment"] and fc not in row["post_comment"]:
                        if int(row["post_label"]) > 0:
                            try:
                                cleft_to_post[int(row["cleft"])].add(int(row["post_label"]))
                            except KeyError:
                                cleft_to_post[int(row["cleft"])] = {int(row["post_label"])}

    return cleft_to_pre, cleft_to_post


def train_until(
    max_iteration,
    cremi_dir,
    samples,
    n5_filename_format,
    csv_filename_format,
    filter_comments_pre,
    filter_comments_post,
    labels,
    net_name,
    input_shape,
    output_shape,
    loss_name,
    aug_mode,
    dt_scaling_factor=50,
    cache_size=5,
    num_workers=10,
    min_masked_voxels=17561.0,
    voxel_size=Coordinate((40,4,4))
):
    def label_filter(cond_f):
        return [ll for ll in labels if cond_f(ll)]

    def get_label(name):
        filter = label_filter(lambda l: l.labelname == name)
        if len(filter) > 0:
            return filter[0]
        else:
            return None

    def network_setup():
        # load net_io_names.json
        with open("net_io_names.json", "r") as f:
            net_io_names = json.load(f)

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

        # define network inputs
        inputs = dict()
        inputs[net_io_names["raw"]] = ak_raw
        inputs[net_io_names["mask"]] = ak_training
        for label in labels:
            inputs[net_io_names["mask_" + label.labelname]] = label.mask_key
            inputs[net_io_names["gt_" + label.labelname]] = label.gt_dist_key
            if label.scale_loss or label.scale_key is not None:
                inputs[net_io_names["w_" + label.labelname]] = label.scale_key

        # define network outputs
        outputs = dict()
        for label in labels:
            outputs[net_io_names[label.labelname]] = label.pred_dist_key
        return net_io_names, start_iteration, inputs, outputs

    keep_thr = float(min_masked_voxels) / np.prod(output_shape)
    max_distance = 2.76 * dt_scaling_factor

    ak_raw = ArrayKey("RAW")
    ak_alpha = ArrayKey("ALPHA_MASK")
    ak_neurons = ArrayKey("GT_NEURONS")
    ak_training = ArrayKey("TRAINING_MASK")
    ak_integral = ArrayKey("INTEGRAL_MASK")
    ak_gradient = ArrayKey("LOSS_GRADIENT")
    ak_clefts = ArrayKey("GT_CLEFTS")

    input_size = Coordinate(input_shape) * voxel_size
    output_size = Coordinate(output_shape) * voxel_size
    net_io_names, start_iteration, inputs, outputs = network_setup()

    # specifiy which Arrays should be requested for each batch
    request = BatchRequest()
    request.add(ak_raw, input_size, voxel_size=voxel_size)
    request.add(ak_neurons, output_size, voxel_size=voxel_size)
    request.add(ak_clefts, output_size, voxel_size=voxel_size)
    request.add(ak_training, output_size, voxel_size=voxel_size)
    request.add(ak_integral, output_size, voxel_size=voxel_size)
    for l in labels:
        request.add(l.mask_key, output_size, voxel_size=voxel_size)
        request.add(l.scale_key, output_size, voxel_size=voxel_size)
        request.add(l.gt_dist_key, output_size, voxel_size=voxel_size)

    # specify specs for output
    array_specs_pred = dict()
    for l in labels:
        array_specs_pred[l.pred_dist_key] = ArraySpec(voxel_size=voxel_size, interpolatable=True)

    snapshot_data = {
        ak_raw: "volumes/raw",
        ak_training: "volumes/masks/training",
        ak_clefts: "volumes/masks/gt_clefts",
        ak_gradient: "volumes/loss_gradient"
    }

    # specify snapshot data layout
    for l in labels:
        snapshot_data[l.mask_key] = "volumes/masks/" + l.labelname
        snapshot_data[l.pred_dist_key] = "volumes/labels/pred_dist_" + l.labelname
        snapshot_data[l.gt_dist_key] = "volumes/labels/gt_dist_" + l.labelname

    # specify snapshot request
    snapshot_request_dict = {ak_gradient: request[ak_clefts]}
    for l in labels:
        snapshot_request_dict[l.pred_dist_key] = request[l.gt_dist_key]
    snapshot_request = BatchRequest(snapshot_request_dict)

    csv_files = [
        os.path.join(cremi_dir, csv_filename_format.format(sample))
        for sample in samples
    ]

    cleft_to_pre, cleft_to_post = make_cleft_to_prepostsyn_neuron_id_dict(
        csv_files, filter_comments_pre, filter_comments_post
    )

    data_providers = []

    for sample in samples:
        logging.info("Adding sample {0:}".format(sample))
        datasets = {
            ak_raw: "volumes/raw",
            ak_training: "volumes/masks/training",
            ak_integral: "volumes/masks/groundtruth_integral",
            ak_clefts: "volumes/labels/clefts",
            ak_neurons: "volumes/labels/neuron_ids"
        }
        specs = {
            ak_clefts: ArraySpec(interpolatable=False),
            ak_training: ArraySpec(interpolatable=False),
            ak_integral: ArraySpec(interpolatable=False)
        }
        for l in labels:
            datasets[l.mask_key] = "volumes/masks/groundtruth"
            specs[l.mask_key] = ArraySpec(interpolatable=False)

        n5_source = ZarrSource(
            os.path.join(cremi_dir, n5_filename_format.format(sample)),
            datasets=datasets,
            array_specs=specs,
        )
        data_providers.append(n5_source)

    data_sources = tuple(
        provider +
        Normalize(ak_raw) +  # ensures RAW is in float in [0, 1]
        RandomLocationWithIntegralMask(
            integral_mask=ak_integral, min_masked=keep_thr
        )
        + Reject(ak_training, min_masked=0.999)
        + Reject(ak_clefts, min_masked=0.0, reject_probability=0.95)
        for provider in data_providers
    )

    artifact_source = (
        Hdf5Source(
            os.path.join(cremi_dir, "sample_ABC_padded_20160501.defects.hdf"),
            datasets={
                ArrayKeys.RAW: "defect_sections/raw",
                ArrayKeys.ALPHA_MASK: "defect_sections/mask",
            },
            array_specs={
                ArrayKeys.RAW: ArraySpec(voxel_size=(40, 4, 4)),
                ArrayKeys.ALPHA_MASK: ArraySpec(voxel_size=(40, 4, 4)),
            },
        )
        + RandomLocation(min_masked=0.05, mask=ak_alpha)
        + Normalize(ak_raw)
        +
        # IntensityAugment(ArrayKeys.RAW, 0.9, 1.1, -0.1, 0.1, z_section_wise=True) +
        ElasticAugment((4, 40, 40), (0, 2, 2), (0, math.pi / 2.0), subsample=8)
        + SimpleAugment(transpose_only=[1, 2], mirror_only=[1, 2])
    )

    train_pipeline = data_sources + RandomProvider()
    if aug_mode == "deluxe":
        slip_ignore = [ak_clefts, ak_training, ak_neurons, ak_integral]
        for l in labels:
            slip_ignore.append(l.mask_key)
        train_pipeline = (
            train_pipeline
            + fuse.ElasticAugment(
                (40, 4, 4),
                (4, 40, 40),
                (0.0, 2.0, 2.0),
                (0, math.pi / 2.0),
                spatial_dims=3,
                subsample=8,
        )
            + SimpleAugment(transpose_only=[1, 2], mirror_only=[1, 2])
            + fuse.Misalign(
                40,
                prob_slip=0.05,
                prob_shift=0.05,
                max_misalign=(10, 10),
                ignore_keys_for_slip=tuple(slip_ignore),
        )
            + IntensityAugment(ak_raw, 0.9, 1.1, -0.1, 0.1, z_section_wise=True)
            + DefectAugment(
                ak_raw,
                prob_missing=0.03,
                prob_low_contrast=0.01,
                prob_artifact=0.03,
                artifact_source=artifact_source,
                artifacts=ak_raw,
                artifacts_mask=ak_alpha,
                contrast_scale=0.5,
        )
        )
    elif aug_mode == "classic":
        train_pipeline = (
            train_pipeline
            + fuse.ElasticAugment(
                (40, 4, 4),
                (4, 40, 40),
                (0.0, 0.0, 0.0),
                (0, math.pi / 2.0),
                spatial_dims=3,
                subsample=8,
        )
            + fuse.SimpleAugment(transpose_only=[1, 2], mirror_only=[1, 2])
            + IntensityAugment(ak_raw, 0.9, 1.1, -0.1, 0.1, z_section_wise=True)
            + DefectAugment(
                ak_raw,
                prob_missing=0.03,
                prob_low_contrast=0.01,
                prob_artifact=0.03,
                artifact_source=artifact_source,
                artifacts=ak_raw,
                artifacts_mask=ak_alpha,
                contrast_scale=0.5,
        )
        )
    elif aug_mode == "lite":
        train_pipeline = (
            train_pipeline
            + fuse.ElasticAugment(
                (40, 4, 4),
                (4, 40, 40),
                (0.0, 0.0, 0.0),
                (0, math.pi / 2.0),
                spatial_dims=3,
                subsample=8,
        )
            + fuse.SimpleAugment(transpose_only=[1, 2], mirror_only=[1, 2])
            + IntensityAugment(ak_raw, 0.9, 1.1, -0.1, 0.1, z_section_wise=False)
        )
    else:
        pass
    train_pipeline = (
        train_pipeline
        + IntensityScaleShift(ak_raw, 2, -1)
        + ZeroOutConstSections(ak_raw))
    clefts = get_label("clefts")
    if clefts is not None:
        train_pipeline += AddDistance(
            label_array_key=clefts.gt_key,
            distance_array_key=clefts.gt_dist_key,
            mask_array_key=clefts.mask_key,
            max_distance=2.76 * dt_scaling_factor,
        )
    pre = get_label("pre")
    post = get_label("post")
    if pre is not None or post is not None:
        train_pipeline += AddPrePostCleftDistance(
            ak_clefts,
            ak_neurons,
            pre.gt_dist_key if pre is not None else None,
            post.gt_dist_key if post is not None else None,
            pre.mask_key if pre is not None else None,
            post.mask_key if post is not None else None,
            cleft_to_pre,
            cleft_to_post,
            include_cleft=False,
            max_distance=2.76 * dt_scaling_factor,
        )
    for l in labels:
        train_pipeline += TanhSaturate(l.gt_dist_key, dt_scaling_factor)
    for l in labels:
        train_pipeline += BalanceByThreshold(labels=l.gt_dist_key, scales=l.scale_key,
                                             mask=(l.mask_key, ak_training),
                                             threshold=l.thr)

    train_pipeline = (train_pipeline
                      + PreCache(cache_size=cache_size, num_workers=num_workers)
                      + Train(net_name, optimizer=net_io_names["optimizer"], loss=net_io_names[loss_name],
                              inputs=inputs, summary=net_io_names["summary"], log_dir="log", save_every=500,
                              log_every=5, outputs=outputs,
                              gradients={net_io_names[labels[0].labelname]: ArrayKeys.LOSS_GRADIENT},
                              )
                      + Snapshot(snapshot_data, every=500, output_filename="batch_{iteration}.hdf",
                                 output_dir="snapshots/", additional_request=snapshot_request
                                 )
                      + PrintProfilingStats(every=50)
                      )

    logging.info("Starting training...")
    with build(train_pipeline) as pp:
        for i in range(start_iteration, max_iteration+1):
            start_it = time.time()
            pp.request_batch(request)
            time_it = time.time() - start_it
            logging.info("it{0:}: {1:}".format(i+1, time_it))
    logging.info("Training finished")
