from gunpowder import *
from gunpowder.tensorflow import *
from gunpowder.contrib.nodes import *

# from training.gunpowder_wrappers import prepare_h5source
import malis
import os
import math
import json
import tensorflow as tf


def train_until(max_iteration, data_sources, cache_size=10, num_workers=10):

    if tf.train.latest_checkpoint("."):
        trained_until = int(tf.train.latest_checkpoint(".").split("_")[-1])
    else:
        trained_until = 0
    if trained_until >= max_iteration:
        return

    ArrayKey("RAW")
    ArrayKey("GT_LABELS")
    ArrayKey("GT_MASK")
    ArrayKey("GT_AFFINITIES")
    ArrayKey("GT_SCALE")
    ArrayKey("PREDICTED_AFFS_1")
    ArrayKey("PREDICTED_AFFS_2")
    ArrayKey("LOSS_GRADIENT_1")
    ArrayKey("LOSS_GRADIENT_2")

    data_providers = []
    fib25_dir = "/groups/saalfeld/saalfeldlab/larissa/data/gunpowder/fib25/"
    if "fib25h5" in data_sources:

        for volume_name in (
            "tstvol-520-1",
            "tstvol-520-2",
            "trvol-250-1",
            "trvol-250-2",
        ):
            h5_source = Hdf5Source(
                os.path.join(fib25_dir, volume_name + ".hdf"),
                datasets={
                    ArrayKeys.RAW: "volumes/raw",
                    ArrayKeys.GT_LABELS: "volumes/labels/neuron_ids",
                    ArrayKeys.GT_MASK: "volumes/labels/mask",
                },
                array_specs={ArrayKeys.GT_MASK: ArraySpec(interpolatable=False)},
            )
            data_providers.append(h5_source)

    fib19_dir = "/groups/saalfeld/saalfeldlab/larissa/fib19"
    #   if 'fib19h5' in data_sources:
    #       for volume_name in ("trvol-250", "trvol-600"):
    #           h5_source = prepare_h5source(fib19_dir, volume_name)
    #           data_providers.append(h5_source)

    with open("net_io_names.json", "r") as f:
        net_io_names = json.load(f)

    voxel_size = Coordinate((8, 8, 8))
    input_size = Coordinate((220,) * 3) * voxel_size
    output_1_size = Coordinate((132,) * 3) * voxel_size
    output_2_size = Coordinate((44,) * 3) * voxel_size

    # input_size = Coordinate((66, 228, 228))*(40,4,4)
    # output_1_size = Coordinate((38, 140, 140))*(40,4,4)
    # output_2_size = Coordinate((10, 52, 52))*(40,4,4)

    request = BatchRequest()
    request.add(ArrayKeys.RAW, input_size)
    request.add(ArrayKeys.GT_LABELS, output_1_size)
    request.add(ArrayKeys.GT_MASK, output_1_size)
    request.add(ArrayKeys.GT_AFFINITIES, output_1_size)
    request.add(ArrayKeys.GT_SCALE, output_1_size)

    snapshot_request = BatchRequest()
    snapshot_request.add(ArrayKeys.RAW, input_size)  # just to center the rest correctly
    snapshot_request.add(ArrayKeys.PREDICTED_AFFS_1, output_1_size)
    snapshot_request.add(ArrayKeys.PREDICTED_AFFS_2, output_2_size)
    snapshot_request.add(ArrayKeys.LOSS_GRADIENT_1, output_1_size)
    snapshot_request.add(ArrayKeys.LOSS_GRADIENT_2, output_2_size)

    data_sources = tuple(
        provider
        + Normalize(ArrayKeys.RAW)
        + Pad(ArrayKeys.RAW, None)
        + Pad(ArrayKeys.GT_MASK, None)
        + RandomLocation()
        + Reject(ArrayKeys.GT_MASK)
        for provider in data_providers
    )

    train_pipeline = (
        data_sources
        + RandomProvider()
        + ElasticAugment(
            [40, 40, 40],
            [2, 2, 2],
            [0, math.pi / 2.0],
            prob_slip=0.01,
            prob_shift=0.05,
            max_misalign=1,
            subsample=8,
        )
        + SimpleAugment()
        + IntensityAugment(ArrayKeys.RAW, 0.9, 1.1, -0.1, 0.1)
        + IntensityScaleShift(ArrayKeys.RAW, 2, -1)
        + ZeroOutConstSections(ArrayKeys.RAW)
        + GrowBoundary(ArrayKeys.GT_LABELS, mask=ArrayKeys.GT_MASK, steps=2)
        + RenumberConnectedComponents(ArrayKeys.GT_LABELS)
        + AddAffinities(malis.mknhood3d(), ArrayKeys.GT_LABELS, ArrayKeys.GT_AFFINITIES)
        + BalanceLabels(ArrayKeys.GT_AFFINITIES, ArrayKeys.GT_SCALE)
        + PreCache(cache_size=cache_size, num_workers=num_workers)
        + Train(
            "wnet",
            optimizer=net_io_names["optimizer"],
            loss=net_io_names["loss"],
            summary=net_io_names["summary"],
            inputs={
                net_io_names["raw"]: ArrayKeys.RAW,
                net_io_names["gt_affs"]: ArrayKeys.GT_AFFINITIES,
                net_io_names["loss_weights"]: ArrayKeys.GT_SCALE,
            },
            outputs={
                net_io_names["affs_1"]: ArrayKeys.PREDICTED_AFFS_1,
                net_io_names["affs_2"]: ArrayKeys.PREDICTED_AFFS_2,
            },
            gradients={
                net_io_names["affs_1"]: ArrayKeys.LOSS_GRADIENT_1,
                net_io_names["affs_2"]: ArrayKeys.LOSS_GRADIENT_2,
            },
        )
        + IntensityScaleShift(ArrayKeys.RAW, 0.5, 0.5)
        + Snapshot(
            {
                ArrayKeys.RAW: "volumes/raw",
                ArrayKeys.GT_LABELS: "volumes/labels/neuron_ids",
                ArrayKeys.GT_AFFINITIES: "volumes/labels/affinities",
                ArrayKeys.PREDICTED_AFFS_1: "volumes/labels/pred_affinities_1",
                ArrayKeys.PREDICTED_AFFS_2: "volumes/labels/pred_affinities_2",
                ArrayKeys.LOSS_GRADIENT_1: "volumes/loss_gradient_1",
                ArrayKeys.LOSS_GRADIENT_2: "volumes/loss_gradient_2",
            },
            every=500,
            output_filename="batch_{iteration}.hdf",
            output_dir="snapshots/",
            additional_request=snapshot_request,
        )
        + PrintProfilingStats(every=1000)
    )

    print("Starting training...")
    with build(train_pipeline) as b:
        for i in range(max_iteration - trained_until):
            b.request_batch(request)
    print("Training finished")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    data_sources = ["fib25h5"]
    max_iteration = 400000
    train_until(max_iteration, data_sources)
