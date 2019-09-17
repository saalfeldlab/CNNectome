from gunpowder import *
from gunpowder.tensorflow import *
from gunpowder.contrib import ZeroOutConstSections, AddDistance
import tensorflow as tf
import os
import math
import json
import sys
import z5py


def train_until(max_iteration, data_sources, input_shape, output_shape, cache_size=10, num_workers=10,):
    ArrayKey("RAW")
    ArrayKey("PRED_RAW")
    data_providers = []
    data_dir = "/groups/saalfeld/saalfeldlab/larissa/data/cell/superresolution/{0:}.n5"
    voxel_size = Coordinate((4, 4, 4))
    input_size = Coordinate(input_shape) * voxel_size
    output_size = Coordinate(output_shape) * voxel_size

    with open("net_io_names.json", "r") as f:
        net_io_names = json.load(f)

    request = BatchRequest()
    request.add(ArrayKeys.RAW, input_size, voxel_size=voxel_size)

    snapshot_request = BatchRequest()
    snapshot_request.add(ArrayKeys.PRED_RAW, output_size, voxel_size=voxel_size)

    # load latest ckpt for weights if available
    if tf.train.latest_checkpoint("."):
        trained_until = int(tf.train.latest_checkpoint(".").split("_")[-1])
        print("Resuming training from", trained_until)
    else:
        trained_until = 0
        print("Starting fresh training")

    # construct DAG
    for src in data_sources:
        n5_source = N5Source(
            data_dir.format(src), datasets={ArrayKeys.RAW: "volumes/raw"}
        )
        data_providers.append(n5_source)

    data_sources = tuple(
        provider
        + Normalize(ArrayKeys.RAW)
        + Pad(ArrayKeys.RAW, Coordinate((400, 400, 400)))
        + RandomLocation()
        for provider in data_providers
    )

    train_pipeline = (
        data_sources
        + ElasticAugment(
            (100, 100, 100),
            (10.0, 10.0, 10.0),
            (0, math.pi / 2.0),
            prob_slip=0,
            prob_shift=0,
            max_misalign=0,
            subsample=8,
        )
        + SimpleAugment()
        + ElasticAugment((40, 1000, 1000), (10.0, 0.0, 0.0), (0, 0), subsample=8)
        + IntensityAugment(ArrayKeys.RAW, 0.9, 1.1, -0.1, 0.1)
        + IntensityScaleShift(ArrayKeys.RAW, 2, -1)
        + ZeroOutConstSections(ArrayKeys.RAW)
        + PreCache(cache_size=cache_size, num_workers=num_workers)
        + Train(
            "unet",
            optimizer=net_io_names["optimizer"],
            loss=net_io_names["loss"],
            inputs={net_io_names["raw"]: ArrayKeys.RAW},
            summary=net_io_names["summary"],
            log_dir="log",
            outputs={net_io_names["pred_raw"]: ArrayKeys.PRED_RAW},
            gradients={},
        )
        + Snapshot(
            {ArrayKeys.RAW: "volumes/raw", ArrayKeys.PRED_RAW: "volumes/pred_raw"},
            every=500,
            output_filename="batch_{iteration}.hdf",
            output_dir="snapshots/",
            additional_request=snapshot_request,
        )
        + PrintProfilingStats(every=50)
    )
    # no intensity augment cause currently can't apply the same to both in and out

    print("Starting training...")
    with build(train_pipeline) as b:
        for i in range(max_iteration):
            b.request_batch(request)

    print("Training finished")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    data_sources = ["block1_4nm"]
    input_shape = (196, 196, 196)
    output_shape = (92, 92, 92)
    max_iteration = 400000
    train_until(max_iteration, data_sources, input_shape, output_shape)
