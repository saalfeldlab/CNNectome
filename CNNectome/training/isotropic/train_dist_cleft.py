from gunpowder import *
from gunpowder.tensorflow import *
from gunpowder.contrib import ZeroOutConstSections, AddDistance
import tensorflow as tf
import os
import math
import json
import numpy as np
import logging


def train_until(
    max_iteration, data_sources, input_shape, output_shape, dt_scaling_factor, loss_name, cache_size=10, num_workers=10,
):
    raw = ArrayKey("RAW")
    # ArrayKey('ALPHA_MASK')
    clefts = ArrayKey("GT_LABELS")
    mask = ArrayKey("GT_MASK")
    scale = ArrayKey("GT_SCALE")
    # grad = ArrayKey('LOSS_GRADIENT')
    gt_dist = ArrayKey("GT_DIST")
    pred_dist = ArrayKey("PREDICTED_DIST")

    data_providers = []

    if tf.train.latest_checkpoint("."):
        trained_until = int(tf.train.latest_checkpoint(".").split("_")[-1])
        print("Resuming training from", trained_until)
    else:
        trained_until = 0
        print("Starting fresh training")
    if trained_until >= max_iteration:
        return
    data_dir = "/groups/saalfeld/saalfeldlab/larissa/data/fib19/mine/"
    for sample in data_sources:
        print(sample)
        h5_source = Hdf5Source(
            os.path.join(data_dir, "cube{0:}.hdf".format(sample)),
            datasets={
                raw: "volumes/raw",
                clefts: "volumes/labels/clefts",
                mask: "/volumes/masks/groundtruth",
            },
            array_specs={mask: ArraySpec(interpolatable=False)},
        )
        data_providers.append(h5_source)

    with open("net_io_names.json", "r") as f:
        net_io_names = json.load(f)

    voxel_size = Coordinate((8, 8, 8))
    input_size = Coordinate(input_shape) * voxel_size
    output_size = Coordinate(output_shape) * voxel_size
    # input_size = Coordinate((132,)*3) * voxel_size
    # output_size = Coordinate((44,)*3) * voxel_size

    # specifiy which Arrays should be requested for each batch
    request = BatchRequest()
    request.add(raw, input_size)
    request.add(clefts, output_size)
    request.add(mask, output_size)
    # request.add(ArrayKeys.TRAINING_MASK, output_size)
    request.add(scale, output_size)
    request.add(gt_dist, output_size)

    # create a tuple of data sources, one for each HDF file
    data_sources = tuple(
        provider + Normalize(ArrayKeys.RAW) +  # ensures RAW is in float in [0, 1]
        # zero-pad provided RAW and GT_MASK to be able to draw batches close to
        # the boundary of the available data
        # size more or less irrelevant as followed by Reject Node
        Pad(raw, None)
        + RandomLocation()
        +  # chose a random location inside the provided arrays
        # Reject(ArrayKeys.GT_MASK) + # reject batches wich do contain less than 50% labelled data
        # Reject(ArrayKeys.TRAINING_MASK, min_masked=0.99) +
        Reject(mask=mask) + Reject(clefts, min_masked=0.0, reject_probability=0.95)
        for provider in data_providers
    )

    snapshot_request = BatchRequest({pred_dist: request[clefts]})

    train_pipeline = (
        data_sources
        + RandomProvider()
        + ElasticAugment(
            (40, 40, 40),
            (2.0, 2.0, 2.0),
            (0, math.pi / 2.0),
            prob_slip=0.01,
            prob_shift=0.01,
            max_misalign=1,
            subsample=8,
        )
        + SimpleAugment()
        + IntensityAugment(raw, 0.9, 1.1, -0.1, 0.1)
        + IntensityScaleShift(raw, 2, -1)
        + ZeroOutConstSections(raw)
        +
        # GrowBoundary(steps=1) +
        # SplitAndRenumberSegmentationLabels() +
        # AddGtAffinities(malis.mknhood3d()) +
        AddDistance(
            label_array_key=clefts,
            distance_array_key=gt_dist,
            normalize="tanh",
            normalize_args=dt_scaling_factor,
        )
        +
        # BalanceLabels(clefts, scale, mask) +
        BalanceByThreshold(labels=ArrayKeys.GT_DIST, scales=ArrayKeys.GT_SCALE)
        +
        # {
        #     ArrayKeys.GT_AFFINITIES: ArrayKeys.GT_SCALE
        # },
        # {
        #     ArrayKeys.GT_AFFINITIES: ArrayKeys.GT_MASK
        # }) +
        PreCache(cache_size=cache_size, num_workers=num_workers)
        + Train(
            "unet",
            optimizer=net_io_names["optimizer"],
            loss=net_io_names[loss_name],
            inputs={
                net_io_names["raw"]: raw,
                net_io_names["gt_dist"]: gt_dist,
                net_io_names["loss_weights"]: scale,
            },
            summary=net_io_names["summary"],
            log_dir="log",
            outputs={net_io_names["dist"]: pred_dist},
            gradients={},
        )
        + Snapshot(
            {
                raw: "volumes/raw",
                clefts: "volumes/labels/gt_clefts",
                gt_dist: "volumes/labels/gt_clefts_dist",
                pred_dist: "volumes/labels/pred_clefts_dist",
            },
            dataset_dtypes={clefts: np.uint64},
            every=500,
            output_filename="batch_{iteration}.hdf",
            output_dir="snapshots/",
            additional_request=snapshot_request,
        )
        + PrintProfilingStats(every=50)
    )

    print("Starting training...")
    with build(train_pipeline) as b:
        for i in range(max_iteration - trained_until):
            b.request_batch(request)

    print("Training finished")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    data_sources = ["01", "02", "03"]  # , 'B', 'C']
    input_shape = (196, 196, 196)
    output_shape = (92, 92, 92)
    dt_scaling_factor = 50
    max_iteration = 400000
    loss_name = "loss_balanced_syn"
    train_until(
        max_iteration,
        data_sources,
        input_shape,
        output_shape,
        dt_scaling_factor,
        loss_name,
    )
