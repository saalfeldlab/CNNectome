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


def make_cleft_to_prepostsyn_neuron_id_dict(csv_files, filter_comments):
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
                if (
                    row["pre_comment"] not in filter_comments
                    and row["post_comment"] not in filter_comments
                ):
                    if int(row["pre_label"]) > 0:
                        try:
                            cleft_to_pre[int(row["cleft"])].add(int(row["pre_label"]))
                        except KeyError:
                            cleft_to_pre[int(row["cleft"])] = {int(row["pre_label"])}
                    if int(row["post_label"]) > 0:
                        try:
                            cleft_to_post[int(row["cleft"])].add(int(row["post_label"]))
                        except KeyError:
                            cleft_to_post[int(row["cleft"])] = {int(row["post_label"])}
    return cleft_to_pre, cleft_to_post


def train_until(
    max_iteration,
    samples,
    aug_mode,
    input_shape,
    output_shape,
    cremi_dir,
    n5_filename_format,
    csv_filename_format,
    filter_comments,
    dt_scaling_factor,
    loss_name,
    net_name="unet_train",
    min_masked_voxels=17561.0,
):
    ArrayKey("RAW")
    ArrayKey("ALPHA_MASK")
    ArrayKey("GT_LABELS")
    ArrayKey("GT_CLEFTS")
    ArrayKey("CLEFT_MASK")
    ArrayKey("PRE_MASK")
    ArrayKey("POST_MASK")
    ArrayKey("TRAINING_MASK")
    ArrayKey("INTEGRAL_MASK")
    ArrayKey("CLEFT_SCALE")
    ArrayKey("PRE_SCALE")
    ArrayKey("POST_SCALE")
    ArrayKey("LOSS_GRADIENT")
    ArrayKey("GT_CLEFT_DIST")
    ArrayKey("PRED_CLEFT_DIST")
    ArrayKey("GT_PRE_DIST")
    ArrayKey("PRED_PRE_DIST")
    ArrayKey("GT_POST_DIST")
    ArrayKey("PRED_POST_DIST")
    ArrayKey("GT_POST_DIST")
    data_providers = []
    if tf.train.latest_checkpoint("."):
        trained_until = int(tf.train.latest_checkpoint(".").split("_")[-1])
        logging.info("Resuming training from", trained_until)
    else:
        trained_until = 0
        logging.info("Starting fresh training")
    for sample in samples:
        logging.info("Adding sample {0:}".format(sample))
        n5_source = N5Source(
            os.path.join(cremi_dir, n5_filename_format.format(sample)),
            datasets={
                ArrayKeys.RAW: "volumes/raw",
                ArrayKeys.GT_CLEFTS: "volumes/labels/clefts",
                ArrayKeys.CLEFT_MASK: "volumes/masks/groundtruth",
                ArrayKeys.PRE_MASK: "volumes/masks/groundtruth",
                ArrayKeys.POST_MASK: "volumes/masks/groundtruth",
                ArrayKeys.TRAINING_MASK: "volumes/masks/training",
                ArrayKeys.INTEGRAL_MASK: "volumes/masks/groundtruth_integral",
                ArrayKeys.GT_LABELS: "volumes/labels/neuron_ids",
            },
            array_specs={
                ArrayKeys.CLEFT_MASK: ArraySpec(interpolatable=False),
                ArrayKeys.PRE_MASK: ArraySpec(interpolatable=False),
                ArrayKeys.POST_MASK: ArraySpec(interpolatable=False),
                ArrayKeys.GT_CLEFTS: ArraySpec(interpolatable=False),
                ArrayKeys.TRAINING_MASK: ArraySpec(interpolatable=False),
                ArrayKeys.INTEGRAL_MASK: ArraySpec(interpolatable=False),
            },
        )
        data_providers.append(n5_source)
    csv_files = [
        os.path.join(cremi_dir, csv_filename_format.format(sample))
        for sample in samples
    ]
    cleft_to_pre, cleft_to_post = make_cleft_to_prepostsyn_neuron_id_dict(
        csv_files, filter_comments
    )
    keep_thr = float(min_masked_voxels) / np.prod(output_shape)

    with open("net_io_names.json", "r") as f:
        net_io_names = json.load(f)

    voxel_size = Coordinate((40, 4, 4))
    input_size = Coordinate(input_shape) * voxel_size
    output_size = Coordinate(output_shape) * voxel_size
    context = input_size - output_size
    # specifiy which Arrays should be requested for each batch
    request = BatchRequest()
    request.add(ArrayKeys.RAW, input_size, voxel_size=voxel_size)
    request.add(ArrayKeys.GT_LABELS, output_size, voxel_size=voxel_size)
    request.add(ArrayKeys.GT_CLEFTS, output_size, voxel_size=voxel_size)
    request.add(ArrayKeys.CLEFT_MASK, output_size, voxel_size=voxel_size)
    request.add(ArrayKeys.PRE_MASK, output_size, voxel_size=voxel_size)
    request.add(ArrayKeys.POST_MASK, output_size, voxel_size=voxel_size)
    request.add(ArrayKeys.TRAINING_MASK, output_size, voxel_size=voxel_size)
    request.add(ArrayKeys.INTEGRAL_MASK, output_size, voxel_size=voxel_size)
    request.add(ArrayKeys.CLEFT_SCALE, output_size, voxel_size=voxel_size)
    request.add(ArrayKeys.PRE_SCALE, output_size, voxel_size=voxel_size)
    request.add(ArrayKeys.POST_SCALE, output_size, voxel_size=voxel_size)
    request.add(ArrayKeys.GT_CLEFT_DIST, output_size, voxel_size=voxel_size)
    request.add(ArrayKeys.GT_PRE_DIST, output_size, voxel_size=voxel_size)
    request.add(ArrayKeys.GT_POST_DIST, output_size, voxel_size=voxel_size)

    # create a tuple of data sources, one for each HDF file
    data_sources = tuple(
        provider + Normalize(ArrayKeys.RAW) +  # ensures RAW is in float in [0, 1]
        # IntensityScaleShift(ArrayKeys.TRAINING_MASK, -1, 1) +
        # zero-pad provided RAW and GT_MASK to be able to draw batches close to
        # the boundary of the available data
        # size more or less irrelevant as followed by Reject Node
        # Pad(ArrayKeys.RAW, None) +
        # Pad(ArrayKeys.GT_MASK, None) +
        # Pad(ArrayKeys.TRAINING_MASK, context) +
        RandomLocationWithIntegralMask(
            integral_mask=ArrayKeys.INTEGRAL_MASK, min_masked=keep_thr
        )
        + Reject(ArrayKeys.TRAINING_MASK, min_masked=0.999)
        + Reject(ArrayKeys.GT_CLEFTS, min_masked=0.0, reject_probability=0.95)
        for provider in data_providers
    )

    snapshot_request = BatchRequest(
        {
            ArrayKeys.LOSS_GRADIENT: request[ArrayKeys.GT_CLEFTS],
            ArrayKeys.PRED_CLEFT_DIST: request[ArrayKeys.GT_CLEFT_DIST],
            ArrayKeys.PRED_PRE_DIST: request[ArrayKeys.GT_PRE_DIST],
            ArrayKeys.PRED_POST_DIST: request[ArrayKeys.GT_POST_DIST],
        }
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
        + RandomLocation(min_masked=0.05, mask=ArrayKeys.ALPHA_MASK)
        + Normalize(ArrayKeys.RAW)
        +
        # IntensityAugment(ArrayKeys.RAW, 0.9, 1.1, -0.1, 0.1, z_section_wise=True) +
        ElasticAugment((4, 40, 40), (0, 2, 2), (0, math.pi / 2.0), subsample=8)
        + SimpleAugment(transpose_only=[1, 2], mirror_only=[1, 2])
    )

    train_pipeline = data_sources + RandomProvider()
    if aug_mode == "deluxe":
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
                ignore_keys_for_slip=(
                    ArrayKeys.GT_CLEFTS,
                    ArrayKeys.CLEFT_MASK,
                    ArrayKeys.PRE_MASK,
                    ArrayKeys.POST_MASK,
                    ArrayKeys.TRAINING_MASK,
                    ArrayKeys.GT_LABELS,
                ),
            )
            + IntensityAugment(ArrayKeys.RAW, 0.9, 1.1, -0.1, 0.1, z_section_wise=True)
            + DefectAugment(
                ArrayKeys.RAW,
                prob_missing=0.03,
                prob_low_contrast=0.01,
                prob_artifact=0.03,
                artifact_source=artifact_source,
                artifacts=ArrayKeys.RAW,
                artifacts_mask=ArrayKeys.ALPHA_MASK,
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
            + IntensityAugment(ArrayKeys.RAW, 0.9, 1.1, -0.1, 0.1, z_section_wise=True)
            + DefectAugment(
                ArrayKeys.RAW,
                prob_missing=0.03,
                prob_low_contrast=0.01,
                prob_artifact=0.03,
                artifact_source=artifact_source,
                artifacts=ArrayKeys.RAW,
                artifacts_mask=ArrayKeys.ALPHA_MASK,
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
            + IntensityAugment(ArrayKeys.RAW, 0.9, 1.1, -0.1, 0.1, z_section_wise=False)
        )
    else:
        pass
    train_pipeline = (
        train_pipeline
        + IntensityScaleShift(ArrayKeys.RAW, 2, -1)
        + ZeroOutConstSections(ArrayKeys.RAW)
        + AddDistance(
            label_array_key=ArrayKeys.GT_CLEFTS,
            distance_array_key=ArrayKeys.GT_CLEFT_DIST,
            mask_array_key=ArrayKeys.CLEFT_MASK,
            max_distance=2.76 * dt_scaling_factor,
        )
        + AddPrePostCleftDistance(
            ArrayKeys.GT_CLEFTS,
            ArrayKeys.GT_LABELS,
            ArrayKeys.GT_PRE_DIST,
            ArrayKeys.GT_POST_DIST,
            ArrayKeys.PRE_MASK,
            ArrayKeys.POST_MASK,
            cleft_to_pre,
            cleft_to_post,
            include_cleft=False,
            max_distance=2.76 * dt_scaling_factor,
        )
        + TanhSaturate(ArrayKeys.GT_PRE_DIST, dt_scaling_factor)
        + TanhSaturate(ArrayKeys.GT_POST_DIST, dt_scaling_factor)
        + TanhSaturate(ArrayKeys.GT_CLEFT_DIST, dt_scaling_factor)
        + BalanceByThreshold(
            labels=ArrayKeys.GT_CLEFT_DIST,
            scales=ArrayKeys.CLEFT_SCALE,
            mask=(ArrayKeys.CLEFT_MASK, ArrayKeys.TRAINING_MASK),
        )
        + BalanceByThreshold(
            labels=ArrayKeys.GT_PRE_DIST,
            scales=ArrayKeys.PRE_SCALE,
            mask=(ArrayKeys.PRE_MASK, ArrayKeys.TRAINING_MASK),
            threshold=-0.5,
        )
        + BalanceByThreshold(
            labels=ArrayKeys.GT_POST_DIST,
            scales=ArrayKeys.POST_SCALE,
            mask=(ArrayKeys.POST_MASK, ArrayKeys.TRAINING_MASK),
            threshold=-0.5,
        )
        + PreCache(cache_size=40, num_workers=10)
        + Train(
            net_name,
            optimizer=net_io_names["optimizer"],
            loss=net_io_names[loss_name],
            inputs={
                net_io_names["raw"]: ArrayKeys.RAW,
                net_io_names["gt_cleft_dist"]: ArrayKeys.GT_CLEFT_DIST,
                net_io_names["gt_pre_dist"]: ArrayKeys.GT_PRE_DIST,
                net_io_names["gt_post_dist"]: ArrayKeys.GT_POST_DIST,
                net_io_names["loss_weights_cleft"]: ArrayKeys.CLEFT_SCALE,
                net_io_names["loss_weights_pre"]: ArrayKeys.PRE_SCALE,
                net_io_names["loss_weights_post"]: ArrayKeys.POST_SCALE,
                net_io_names["cleft_mask"]: ArrayKeys.CLEFT_MASK,
                net_io_names["pre_mask"]: ArrayKeys.PRE_MASK,
                net_io_names["post_mask"]: ArrayKeys.POST_MASK,
            },
            summary=net_io_names["summary"],
            log_dir="log",
            save_every=500,
            log_every=5,
            outputs={
                net_io_names["cleft_dist"]: ArrayKeys.PRED_CLEFT_DIST,
                net_io_names["pre_dist"]: ArrayKeys.PRED_PRE_DIST,
                net_io_names["post_dist"]: ArrayKeys.PRED_POST_DIST,
            },
            gradients={net_io_names["cleft_dist"]: ArrayKeys.LOSS_GRADIENT},
        )
        + Snapshot(
            {
                ArrayKeys.RAW: "volumes/raw",
                ArrayKeys.GT_CLEFTS: "volumes/labels/gt_clefts",
                ArrayKeys.GT_CLEFT_DIST: "volumes/labels/gt_clefts_dist",
                ArrayKeys.PRED_CLEFT_DIST: "volumes/labels/pred_clefts_dist",
                ArrayKeys.CLEFT_MASK: "volumes/masks/cleft",
                ArrayKeys.PRE_MASK: "volumes/masks/pre",
                ArrayKeys.POST_MASK: "volumes/masks/post",
                ArrayKeys.LOSS_GRADIENT: "volumes/loss_gradient",
                ArrayKeys.PRED_PRE_DIST: "volumes/labels/pred_pre_dist",
                ArrayKeys.PRED_POST_DIST: "volumes/labels/pred_post_dist",
                ArrayKeys.GT_PRE_DIST: "volumes/labels/gt_pre_dist",
                ArrayKeys.GT_POST_DIST: "volumes/labels/gt_post_dist",
            },
            every=500,
            output_filename="batch_{iteration}.hdf",
            output_dir="snapshots/",
            additional_request=snapshot_request,
        )
        + PrintProfilingStats(every=50)
    )

    logging.info("Starting training...")
    with build(train_pipeline) as b:
        for i in range(max_iteration):
            b.request_batch(request)

    logging.info("Training finished")
