from gunpowder import *
from gunpowder.tensorflow import *
from gunpowder.contrib import ZeroOutConstSections, AddDistance
import tensorflow as tf
import os
import math
import json
import logging


def train_until(
    max_iteration, data_sources, input_shape, output_shape, dt_scaling_factor, loss_name
):
    ArrayKey("RAW")
    ArrayKey("ALPHA_MASK")
    ArrayKey("GT_LABELS")
    ArrayKey("GT_MASK")
    ArrayKey("TRAINING_MASK")
    ArrayKey("GT_SCALE")
    ArrayKey("LOSS_GRADIENT")
    ArrayKey("GT_DIST")
    ArrayKey("PREDICTED_DIST_LABELS")

    data_providers = []
    if cremi_version == "2016":
        cremi_dir = "/groups/saalfeld/saalfeldlab/larissa/data/cremi-2016/"
        filename = "sample_{0:}_padded_20160501."
    elif cremi_version == "2017":
        cremi_dir = "/groups/saalfeld/saalfeldlab/larissa/data/cremi-2017/"
        filename = "sample_{0:}_padded_20170424."
    if aligned:
        filename += "aligned."
    filename += "0bg.hdf"
    if tf.train.latest_checkpoint("."):
        trained_until = int(tf.train.latest_checkpoint(".").split("_")[-1])
        print("Resuming training from", trained_until)
    else:
        trained_until = 0
        print("Starting fresh training")
    for sample in data_sources:
        print(sample)
        h5_source = Hdf5Source(
            os.path.join(cremi_dir, filename.format(sample)),
            datasets={
                ArrayKeys.RAW: "volumes/raw",
                ArrayKeys.GT_LABELS: "volumes/labels/clefts",
                ArrayKeys.GT_MASK: "volumes/masks/groundtruth",
                ArrayKeys.TRAINING_MASK: "volumes/masks/validation",
            },
            array_specs={ArrayKeys.GT_MASK: ArraySpec(interpolatable=False)},
        )
        data_providers.append(h5_source)

    with open("net_io_names.json", "r") as f:
        net_io_names = json.load(f)

    voxel_size = Coordinate((40, 4, 4))
    input_size = Coordinate(input_shape) * voxel_size
    output_size = Coordinate(output_shape) * voxel_size
    context = input_size - output_size
    # specifiy which Arrays should be requested for each batch
    request = BatchRequest()
    request.add(ArrayKeys.RAW, input_size)
    request.add(ArrayKeys.GT_LABELS, output_size)
    request.add(ArrayKeys.GT_MASK, output_size)
    request.add(ArrayKeys.TRAINING_MASK, output_size)
    request.add(ArrayKeys.GT_SCALE, output_size)
    request.add(ArrayKeys.GT_DIST, output_size)

    # create a tuple of data sources, one for each HDF file
    data_sources = tuple(
        provider
        + Normalize(ArrayKeys.RAW)
        + IntensityScaleShift(  # ensures RAW is in float in [0, 1]
            ArrayKeys.TRAINING_MASK, -1, 1
        )
        +
        # zero-pad provided RAW and GT_MASK to be able to draw batches close to
        # the boundary of the available data
        # size more or less irrelevant as followed by Reject Node
        Pad(ArrayKeys.RAW, None)
        + Pad(ArrayKeys.GT_MASK, None)
        + Pad(ArrayKeys.TRAINING_MASK, context)
        + RandomLocation(min_masked=0.99, mask=ArrayKeys.TRAINING_MASK)
        + Reject(ArrayKeys.GT_MASK)
        + Reject(  # reject batches wich do contain less than 50% labelled data
            ArrayKeys.GT_LABELS, min_masked=0.0, reject_probability=0.95
        )
        for provider in data_providers
    )

    snapshot_request = BatchRequest(
        {
            ArrayKeys.LOSS_GRADIENT: request[ArrayKeys.GT_LABELS],
            ArrayKeys.PREDICTED_DIST_LABELS: request[ArrayKeys.GT_LABELS],
            ArrayKeys.LOSS_GRADIENT: request[ArrayKeys.GT_DIST],
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
        + IntensityAugment(ArrayKeys.RAW, 0.9, 1.1, -0.1, 0.1, z_section_wise=True)
        + ElasticAugment((4, 40, 40), (0, 2, 2), (0, math.pi / 2.0), subsample=8)
        + SimpleAugment(transpose_only=[1, 2])
    )

    train_pipeline = (
        data_sources
        + RandomProvider()
        + ElasticAugment(
            (4, 40, 40),
            (0.0, 2.0, 2.0),
            (0, math.pi / 2.0),
            prob_slip=0.05,
            prob_shift=0.05,
            max_misalign=10,
            subsample=8,
        )
        + SimpleAugment(transpose_only=[1, 2], mirror_only=[1, 2])
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
        + IntensityScaleShift(ArrayKeys.RAW, 2, -1)
        + ZeroOutConstSections(ArrayKeys.RAW)
        + AddDistance(
            label_array_key=ArrayKeys.GT_LABELS,
            distance_array_key=ArrayKeys.GT_DIST,
            normalize="tanh",
            normalize_args=dt_scaling_factor,
        )
        + BalanceByThreshold(
            ArrayKeys.GT_LABELS, ArrayKeys.GT_SCALE, mask=ArrayKeys.GT_MASK
        )
        + PreCache(cache_size=40, num_workers=10)
        + Train(
            "unet",
            optimizer=net_io_names["optimizer"],
            loss=net_io_names[loss_name],
            inputs={
                net_io_names["raw"]: ArrayKeys.RAW,
                net_io_names["gt_dist"]: ArrayKeys.GT_DIST,
                net_io_names["loss_weights"]: ArrayKeys.GT_SCALE,
                net_io_names["mask"]: ArrayKeys.GT_MASK,
            },
            summary=net_io_names["summary"],
            log_dir="log",
            outputs={net_io_names["dist"]: ArrayKeys.PREDICTED_DIST_LABELS},
            gradients={net_io_names["dist"]: ArrayKeys.LOSS_GRADIENT},
        )
        + Snapshot(
            {
                ArrayKeys.RAW: "volumes/raw",
                ArrayKeys.GT_LABELS: "volumes/labels/gt_clefts",
                ArrayKeys.GT_DIST: "volumes/labels/gt_clefts_dist",
                ArrayKeys.PREDICTED_DIST_LABELS: "volumes/labels/pred_clefts_dist",
                ArrayKeys.LOSS_GRADIENT: "volumes/loss_gradient",
            },
            every=500,
            output_filename="batch_{iteration}.hdf",
            output_dir="snapshots/",
            additional_request=snapshot_request,
        )
        + PrintProfilingStats(every=50)
    )

    print("Starting training...")
    with build(train_pipeline) as b:
        for i in range(max_iteration):
            b.request_batch(request)

    print("Training finished")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    data_sources = ["A", "B", "C"]  # , 'B', 'C']
    input_shape = (43, 430, 430)
    output_shape = (23, 218, 218)
    dt_scaling_factor = 50
    max_iteration = 400000
    loss_name = "loss_balanced_syn"
    cremi_version = "2017"
    aligned = True
    train_until(
        max_iteration,
        data_sources,
        input_shape,
        output_shape,
        dt_scaling_factor,
        loss_name,
    )
