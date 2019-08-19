from gunpowder import *
from gunpowder.contrib.nodes import *
from gunpowder.tensorflow import *

# from training.gunpowder_wrappers import prepare_h5source
import malis
import os
import math
import json
import logging


def train_until(max_iteration, data_sources):
    ArrayKey("RAW")
    ArrayKey("ALPHA_MASK")
    ArrayKey("GT_LABELS")
    ArrayKey("GT_MASK")
    ArrayKey("GT_SCALE")
    ArrayKey("GT_AFFINITIES")
    ArrayKey("PREDICTED_AFFS")
    ArrayKey("LOSS_GRADIENT")

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
    # if 'fib19h5' in data_sources:
    #    for volume_name in ("trvol-250", "trvol-600"):
    #        h5_source = prepare_h5source(fib19_dir, volume_name)
    #        data_providers.append(h5_source)

    with open("net_io_names.json", "r") as f:
        net_io_names = json.load(f)

    voxel_size = Coordinate((8, 8, 8))
    input_size = Coordinate((132,) * 3) * voxel_size
    output_size = Coordinate((44,) * 3) * voxel_size

    # specifiy which volumes should be requested for each batch
    request = BatchRequest()
    request.add(ArrayKeys.RAW, input_size)
    request.add(ArrayKeys.GT_LABELS, output_size)
    request.add(ArrayKeys.GT_MASK, input_size)
    request.add(ArrayKeys.GT_SCALE, output_size)
    request.add(ArrayKeys.GT_AFFINITIES, output_size)

    # create a tuple of data sources, one for each HDF file
    data_sources = tuple(
        provider + Normalize(ArrayKeys.RAW) +  # ensures RAW is in float in [0, 1]
        # zero-pad provided RAW and GT_MASK to be able to draw batches close to
        # the boundary of the available data
        # size more or less irrelevant as followed by Reject Node
        Pad(ArrayKeys.RAW, None)
        + Pad(ArrayKeys.GT_MASK, None)
        + RandomLocation()
        + Reject(  # chose a random location inside the provided volumes
            ArrayKeys.GT_MASK
        )  # reject batches wich do contain less than 50% labelled data
        for provider in data_providers
    )

    snapshot_request = BatchRequest(
        {
            ArrayKeys.LOSS_GRADIENT: request[ArrayKeys.GT_LABELS],
            ArrayKeys.PREDICTED_AFFS: request[ArrayKeys.GT_LABELS],
            ArrayKeys.LOSS_GRADIENT: request[ArrayKeys.GT_AFFINITIES],
        }
    )

    # artifact_source = (
    #    Hdf5Source(
    #        os.path.join(data_dir, 'sample_ABC_padded_20160501.defects.hdf'),
    #        datasets = {
    #            VolumeTypes.RAW: 'defect_sections/raw',
    #            VolumeTypes.ALPHA_MASK: 'defect_sections/mask',
    #        },
    #        volume_specs = {
    #            VolumeTypes.RAW: VolumeSpec(voxel_size=(40, 4, 4)),
    #            VolumeTypes.ALPHA_MASK: VolumeSpec(voxel_size=(40, 4, 4)),
    #        }
    #    ) +
    #    RandomLocation(min_masked=0.05, mask_volume_type=VolumeTypes.ALPHA_MASK) +
    #    Normalize() +
    #    IntensityAugment(0.9, 1.1, -0.1, 0.1, z_section_wise=True) +
    #    ElasticAugment([4,40,40], [0,2,2], [0,math.pi/2.0], subsample=8) +
    #    SimpleAugment(transpose_only_xy=True)
    # )

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
        + PreCache(cache_size=40, num_workers=10)
        +
        # DefectAugment(
        #    prob_missing=0.03,
        #    prob_low_contrast=0.01,
        #    prob_artifact=0.03,
        #    artifact_source=artifact_source,
        #    contrast_scale=0.5) +
        Train(
            "unet_auto",
            optimizer=net_io_names["optimizer"],
            loss=net_io_names["loss"],
            inputs={
                net_io_names["raw"]: ArrayKeys.RAW,
                net_io_names["pred"]: ArrayKeys.GT_MASK,
                net_io_names["gt_affs"]: ArrayKeys.GT_AFFINITIES,
                net_io_names["loss_weights"]: ArrayKeys.GT_SCALE,
            },
            outputs={net_io_names["affs"]: ArrayKeys.PREDICTED_AFFS},
            gradients={net_io_names["affs"]: ArrayKeys.LOSS_GRADIENT},
        )
        + Snapshot(
            {
                ArrayKeys.RAW: "volumes/raw",
                ArrayKeys.GT_LABELS: "volumes/labels/neuron_ids",
                ArrayKeys.GT_AFFINITIES: "volumes/labels/affinities",
                ArrayKeys.PREDICTED_AFFS: "volumes/labels/pred_affinities",
                ArrayKeys.LOSS_GRADIENT: "volumes/loss_gradient",
            },
            every=1000,
            output_filename="batch_{iteration}.hdf",
            output_dir="snapshots/",
            additional_request=snapshot_request,
        )
        + PrintProfilingStats(every=5000)
    )

    print("Starting training...")
    with build(train_pipeline) as b:
        for i in range(max_iteration):
            b.request_batch(request)
    print("Training finished")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    data_sources = ["fib25h5"]  # , 'fib19h5']
    max_iteration = 400000
    train_until(max_iteration, data_sources)
