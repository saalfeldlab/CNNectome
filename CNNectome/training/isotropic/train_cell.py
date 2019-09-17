from gunpowder import *
from gunpowder.tensorflow import *
from gunpowder.contrib import ZeroOutConstSections, AddDistance
import tensorflow as tf
import os
import math
import json
import sys

print("syspath", sys.path)
import z5py


def train_until(
    max_iteration, data_sources, input_shape, output_shape, dt_scaling_factor, loss_name, cache_size=10, num_workers=10,
):
    ArrayKey("RAW")
    ArrayKey("RAW_UP")
    ArrayKey("ALPHA_MASK")
    ArrayKey("GT_LABELS")
    ArrayKey("MASK")
    ArrayKey("MASK_UP")
    ArrayKey("GT_DIST_CENTROSOME")
    ArrayKey("GT_DIST_GOLGI")
    ArrayKey("GT_DIST_GOLGI_MEM")
    ArrayKey("GT_DIST_ER")
    ArrayKey("GT_DIST_ER_MEM")
    ArrayKey("GT_DIST_MVB")
    ArrayKey("GT_DIST_MVB_MEM")
    ArrayKey("GT_DIST_MITO")
    ArrayKey("GT_DIST_MITO_MEM")
    ArrayKey("GT_DIST_LYSOSOME")
    ArrayKey("GT_DIST_LYSOSOME_MEM")

    ArrayKey("PRED_DIST_CENTROSOME")
    ArrayKey("PRED_DIST_GOLGI")
    ArrayKey("PRED_DIST_GOLGI_MEM")
    ArrayKey("PRED_DIST_ER")
    ArrayKey("PRED_DIST_ER_MEM")
    ArrayKey("PRED_DIST_MVB")
    ArrayKey("PRED_DIST_MVB_MEM")
    ArrayKey("PRED_DIST_MITO")
    ArrayKey("PRED_DIST_MITO_MEM")
    ArrayKey("PRED_DIST_LYSOSOME")
    ArrayKey("PRED_DIST_LYSOSOME_MEM")

    ArrayKey("SCALE_CENTROSOME")
    ArrayKey("SCALE_GOLGI")
    ArrayKey("SCALE_GOLGI_MEM")
    ArrayKey("SCALE_ER")
    ArrayKey("SCALE_ER_MEM")
    ArrayKey("SCALE_MVB")
    ArrayKey("SCALE_MVB_MEM")
    ArrayKey("SCALE_MITO")
    ArrayKey("SCALE_MITO_MEM")
    ArrayKey("SCALE_LYSOSOME")
    ArrayKey("SCALE_LYSOSOME_MEM")

    data_providers = []
    data_dir = "/groups/saalfeld/saalfeldlab/larissa/data/cell/{0:}.n5"
    voxel_size_up = Coordinate((4, 4, 4))
    voxel_size_orig = Coordinate((8, 8, 8))
    input_size = Coordinate(input_shape) * voxel_size_orig
    output_size = Coordinate(output_shape) * voxel_size_orig

    if tf.train.latest_checkpoint("."):
        trained_until = int(tf.train.latest_checkpoint(".").split("_")[-1])
        print("Resuming training from", trained_until)
    else:
        trained_until = 0
        print("Starting fresh training")
    for src in data_sources:
        n5_source = N5Source(
            os.path.join(data_dir.format(src)),
            datasets={
                ArrayKeys.RAW_UP: "volumes/raw",
                ArrayKeys.GT_LABELS: "volumes/labels/all",
                ArrayKeys.MASK_UP: "volumes/mask",
            },
            array_specs={ArrayKeys.MASK_UP: ArraySpec(interpolatable=False)},
        )
        data_providers.append(n5_source)

    with open("net_io_names.json", "r") as f:
        net_io_names = json.load(f)

    # specifiy which Arrays should be requested for each batch
    request = BatchRequest()
    request.add(ArrayKeys.RAW, input_size, voxel_size=voxel_size_orig)
    request.add(ArrayKeys.RAW_UP, input_size, voxel_size=voxel_size_up)
    request.add(ArrayKeys.GT_LABELS, output_size, voxel_size=voxel_size_up)
    request.add(ArrayKeys.MASK_UP, output_size, voxel_size=voxel_size_up)
    request.add(ArrayKeys.MASK, output_size, voxel_size=voxel_size_orig)
    request.add(ArrayKeys.GT_DIST_CENTROSOME, output_size, voxel_size=voxel_size_orig)
    request.add(ArrayKeys.GT_DIST_GOLGI, output_size, voxel_size=voxel_size_orig)
    request.add(ArrayKeys.GT_DIST_GOLGI_MEM, output_size, voxel_size=voxel_size_orig)
    request.add(ArrayKeys.GT_DIST_ER, output_size, voxel_size=voxel_size_orig)
    request.add(ArrayKeys.GT_DIST_ER_MEM, output_size, voxel_size=voxel_size_orig)
    request.add(ArrayKeys.GT_DIST_MVB, output_size, voxel_size=voxel_size_orig)
    request.add(ArrayKeys.GT_DIST_MVB_MEM, output_size, voxel_size=voxel_size_orig)
    request.add(ArrayKeys.GT_DIST_MITO, output_size, voxel_size=voxel_size_orig)
    request.add(ArrayKeys.GT_DIST_MITO_MEM, output_size, voxel_size=voxel_size_orig)
    request.add(ArrayKeys.GT_DIST_LYSOSOME, output_size, voxel_size=voxel_size_orig)
    request.add(ArrayKeys.GT_DIST_LYSOSOME_MEM, output_size, voxel_size=voxel_size_orig)
    request.add(ArrayKeys.SCALE_CENTROSOME, output_size, voxel_size=voxel_size_orig)
    request.add(ArrayKeys.SCALE_GOLGI, output_size, voxel_size=voxel_size_orig)
    # request.add(ArrayKeys.SCALE_GOLGI_MEM, output_size, voxel_size=voxel_size_orig)
    request.add(ArrayKeys.SCALE_ER, output_size, voxel_size=voxel_size_orig)
    # request.add(ArrayKeys.SCALE_ER_MEM, output_size, voxel_size=voxel_size_orig)
    request.add(ArrayKeys.SCALE_MVB, output_size, voxel_size=voxel_size_orig)
    # request.add(ArrayKeys.SCALE_MVB_MEM, output_size, voxel_size=voxel_size_orig)
    request.add(ArrayKeys.SCALE_MITO, output_size, voxel_size=voxel_size_orig)
    # request.add(ArrayKeys.SCALE_MITO_MEM, output_size, voxel_size=voxel_size_orig)
    request.add(ArrayKeys.SCALE_LYSOSOME, output_size, voxel_size=voxel_size_orig)
    # request.add(ArrayKeys.SCALE_LYSOSOME_MEM, output_size, voxel_size=voxel_size_orig)

    # create a tuple of data sources, one for each HDF file
    data_sources = tuple(
        provider + Normalize(ArrayKeys.RAW_UP) +  # ensures RAW is in float in [0, 1]
        # zero-pad provided RAW and MASK to be able to draw batches close to
        # the boundary of the available data
        # size more or less irrelevant as followed by Reject Node
        Pad(ArrayKeys.RAW_UP, None)
        + RandomLocation(
            min_masked=0.5, mask=ArrayKeys.MASK_UP
        )  # chose a random location inside the provided arrays
        # Reject(ArrayKeys.MASK) # reject batches wich do contain less than 50% labelled data
        for provider in data_providers
    )

    snapshot_request = BatchRequest()
    snapshot_request.add(ArrayKeys.PRED_DIST_CENTROSOME, output_size)
    snapshot_request.add(ArrayKeys.PRED_DIST_GOLGI, output_size)
    snapshot_request.add(ArrayKeys.PRED_DIST_GOLGI_MEM, output_size)
    snapshot_request.add(ArrayKeys.PRED_DIST_ER, output_size)
    snapshot_request.add(ArrayKeys.PRED_DIST_ER_MEM, output_size)
    snapshot_request.add(ArrayKeys.PRED_DIST_MVB, output_size)
    snapshot_request.add(ArrayKeys.PRED_DIST_MVB_MEM, output_size)
    snapshot_request.add(ArrayKeys.PRED_DIST_MITO, output_size)
    snapshot_request.add(ArrayKeys.PRED_DIST_MITO_MEM, output_size)
    snapshot_request.add(ArrayKeys.PRED_DIST_LYSOSOME, output_size)
    snapshot_request.add(ArrayKeys.PRED_DIST_LYSOSOME_MEM, output_size)
    train_pipeline = (
        data_sources
        + RandomProvider()
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
        + IntensityAugment(ArrayKeys.RAW_UP, 0.9, 1.1, -0.1, 0.1)
        + IntensityScaleShift(ArrayKeys.RAW_UP, 2, -1)
        + ZeroOutConstSections(ArrayKeys.RAW_UP)
        +
        # GrowBoundary(steps=1) +
        # SplitAndRenumberSegmentationLabels() +
        # AddGtAffinities(malis.mknhood3d()) +
        AddDistance(
            label_array_key=ArrayKeys.GT_LABELS,
            distance_array_key=ArrayKeys.GT_DIST_CENTROSOME,
            normalize="tanh",
            normalize_args=dt_scaling_factor,
            label_id=1,
            factor=2,
        )
        + AddDistance(
            label_array_key=ArrayKeys.GT_LABELS,
            distance_array_key=ArrayKeys.GT_DIST_GOLGI,
            normalize="tanh",
            normalize_args=dt_scaling_factor,
            label_id=(2, 11),
            factor=2,
        )
        + AddDistance(
            label_array_key=ArrayKeys.GT_LABELS,
            distance_array_key=ArrayKeys.GT_DIST_GOLGI_MEM,
            normalize="tanh",
            normalize_args=dt_scaling_factor,
            label_id=11,
            factor=2,
        )
        + AddDistance(
            label_array_key=ArrayKeys.GT_LABELS,
            distance_array_key=ArrayKeys.GT_DIST_ER,
            normalize="tanh",
            normalize_args=dt_scaling_factor,
            label_id=(3, 10),
            factor=2,
        )
        + AddDistance(
            label_array_key=ArrayKeys.GT_LABELS,
            distance_array_key=ArrayKeys.GT_DIST_ER_MEM,
            normalize="tanh",
            normalize_args=dt_scaling_factor,
            label_id=10,
            factor=2,
        )
        + AddDistance(
            label_array_key=ArrayKeys.GT_LABELS,
            distance_array_key=ArrayKeys.GT_DIST_MVB,
            normalize="tanh",
            normalize_args=dt_scaling_factor,
            label_id=(4, 9),
            factor=2,
        )
        + AddDistance(
            label_array_key=ArrayKeys.GT_LABELS,
            distance_array_key=ArrayKeys.GT_DIST_MVB_MEM,
            normalize="tanh",
            normalize_args=dt_scaling_factor,
            label_id=9,
            factor=2,
        )
        + AddDistance(
            label_array_key=ArrayKeys.GT_LABELS,
            distance_array_key=ArrayKeys.GT_DIST_MITO,
            normalize="tanh",
            normalize_args=dt_scaling_factor,
            label_id=(5, 8),
            factor=2,
        )
        + AddDistance(
            label_array_key=ArrayKeys.GT_LABELS,
            distance_array_key=ArrayKeys.GT_DIST_MITO_MEM,
            normalize="tanh",
            normalize_args=dt_scaling_factor,
            label_id=8,
            factor=2,
        )
        + AddDistance(
            label_array_key=ArrayKeys.GT_LABELS,
            distance_array_key=ArrayKeys.GT_DIST_LYSOSOME,
            normalize="tanh",
            normalize_args=dt_scaling_factor,
            label_id=(6, 7),
            factor=2,
        )
        + AddDistance(
            label_array_key=ArrayKeys.GT_LABELS,
            distance_array_key=ArrayKeys.GT_DIST_LYSOSOME_MEM,
            normalize="tanh",
            normalize_args=dt_scaling_factor,
            label_id=7,
            factor=2,
        )
        + DownSample(ArrayKeys.MASK_UP, 2, ArrayKeys.MASK)
        + BalanceByThreshold(
            ArrayKeys.GT_DIST_CENTROSOME,
            ArrayKeys.SCALE_CENTROSOME,
            mask=ArrayKeys.MASK,
        )
        + BalanceByThreshold(
            ArrayKeys.GT_DIST_GOLGI, ArrayKeys.SCALE_GOLGI, mask=ArrayKeys.MASK
        )
        +
        # BalanceByThreshold(ArrayKeys.GT_DIST_GOLGI_MEM, ArrayKeys.SCALE_GOLGI_MEM, mask=ArrayKeys.MASK) +
        BalanceByThreshold(
            ArrayKeys.GT_DIST_ER, ArrayKeys.SCALE_ER, mask=ArrayKeys.MASK
        )
        +
        # BalanceByThreshold(ArrayKeys.GT_DIST_ER_MEM, ArrayKeys.SCALE_ER_MEM, mask=ArrayKeys.MASK) +
        BalanceByThreshold(
            ArrayKeys.GT_DIST_MVB, ArrayKeys.SCALE_MVB, mask=ArrayKeys.MASK
        )
        +
        # BalanceByThreshold(ArrayKeys.GT_DIST_MVB_MEM, ArrayKeys.SCALE_MVB_MEM, mask=ArrayKeys.MASK) +
        BalanceByThreshold(
            ArrayKeys.GT_DIST_MITO, ArrayKeys.SCALE_MITO, mask=ArrayKeys.MASK
        )
        +
        # BalanceByThreshold(ArrayKeys.GT_DIST_MITO_MEM, ArrayKeys.SCALE_MITO_MEM, mask=ArrayKeys.MASK) +
        BalanceByThreshold(
            ArrayKeys.GT_DIST_LYSOSOME, ArrayKeys.SCALE_LYSOSOME, mask=ArrayKeys.MASK
        )
        +
        # BalanceByThreshold(ArrayKeys.GT_DIST_LYSOSOME_MEM, ArrayKeys.SCALE_LYSOSOME_MEM, mask=ArrayKeys.MASK) +
        # BalanceByThreshold(
        #    labels=ArrayKeys.GT_DIST,
        #    scales= ArrayKeys.GT_SCALE) +
        # {
        #     ArrayKeys.GT_AFFINITIES: ArrayKeys.GT_SCALE
        # },
        # {
        #     ArrayKeys.GT_AFFINITIES: ArrayKeys.MASK
        # }) +
        DownSample(ArrayKeys.RAW_UP, 2, ArrayKeys.RAW)
        + PreCache(cache_size=cache_size, num_workers=num_workers)
        + Train(
            "build",
            optimizer=net_io_names["optimizer"],
            loss=net_io_names[loss_name],
            inputs={
                net_io_names["raw"]: ArrayKeys.RAW,
                net_io_names["gt_centrosome"]: ArrayKeys.GT_DIST_CENTROSOME,
                net_io_names["gt_golgi"]: ArrayKeys.GT_DIST_GOLGI,
                net_io_names["gt_golgi_mem"]: ArrayKeys.GT_DIST_GOLGI_MEM,
                net_io_names["gt_er"]: ArrayKeys.GT_DIST_ER,
                net_io_names["gt_er_mem"]: ArrayKeys.GT_DIST_ER_MEM,
                net_io_names["gt_mvb"]: ArrayKeys.GT_DIST_MVB,
                net_io_names["gt_mvb_mem"]: ArrayKeys.GT_DIST_MVB_MEM,
                net_io_names["gt_mito"]: ArrayKeys.GT_DIST_MITO,
                net_io_names["gt_mito_mem"]: ArrayKeys.GT_DIST_MITO_MEM,
                net_io_names["gt_lysosome"]: ArrayKeys.GT_DIST_LYSOSOME,
                net_io_names["gt_lysosome_mem"]: ArrayKeys.GT_DIST_LYSOSOME_MEM,
                net_io_names["w_centrosome"]: ArrayKeys.SCALE_CENTROSOME,
                net_io_names["w_golgi"]: ArrayKeys.SCALE_GOLGI,
                net_io_names["w_golgi_mem"]: ArrayKeys.SCALE_GOLGI,
                net_io_names["w_er"]: ArrayKeys.SCALE_ER,
                net_io_names["w_er_mem"]: ArrayKeys.SCALE_ER,
                net_io_names["w_mvb"]: ArrayKeys.SCALE_MVB,
                net_io_names["w_mvb_mem"]: ArrayKeys.SCALE_MVB,
                net_io_names["w_mito"]: ArrayKeys.SCALE_MITO,
                net_io_names["w_mito_mem"]: ArrayKeys.SCALE_MITO,
                net_io_names["w_lysosome"]: ArrayKeys.SCALE_LYSOSOME,
                net_io_names["w_lysosome_mem"]: ArrayKeys.SCALE_LYSOSOME,
            },
            summary=net_io_names["summary"],
            log_dir="log",
            outputs={
                net_io_names["centrosome"]: ArrayKeys.PRED_DIST_CENTROSOME,
                net_io_names["golgi"]: ArrayKeys.PRED_DIST_GOLGI,
                net_io_names["golgi_mem"]: ArrayKeys.PRED_DIST_GOLGI_MEM,
                net_io_names["er"]: ArrayKeys.PRED_DIST_ER,
                net_io_names["er_mem"]: ArrayKeys.PRED_DIST_ER_MEM,
                net_io_names["mvb"]: ArrayKeys.PRED_DIST_MVB,
                net_io_names["mvb_mem"]: ArrayKeys.PRED_DIST_MVB_MEM,
                net_io_names["mito"]: ArrayKeys.PRED_DIST_MITO,
                net_io_names["mito_mem"]: ArrayKeys.PRED_DIST_MITO_MEM,
                net_io_names["lysosome"]: ArrayKeys.PRED_DIST_LYSOSOME,
                net_io_names["lysosome_mem"]: ArrayKeys.PRED_DIST_LYSOSOME_MEM,
            },
            gradients={},
        )
        + Snapshot(
            {
                ArrayKeys.RAW: "volumes/raw",
                ArrayKeys.GT_LABELS: "volumes/labels/gt_labels",
                ArrayKeys.GT_DIST_CENTROSOME: "volumes/labels/gt_dist_centrosome",
                ArrayKeys.PRED_DIST_CENTROSOME: "volumes/labels/pred_dist_centrosome",
                ArrayKeys.GT_DIST_GOLGI: "volumes/labels/gt_dist_golgi",
                ArrayKeys.PRED_DIST_GOLGI: "volumes/labels/pred_dist_golgi",
                ArrayKeys.GT_DIST_GOLGI_MEM: "volumes/labels/gt_dist_golgi_mem",
                ArrayKeys.PRED_DIST_GOLGI_MEM: "volumes/labels/pred_dist_golgi_mem",
                ArrayKeys.GT_DIST_ER: "volumes/labels/gt_dist_er",
                ArrayKeys.PRED_DIST_ER: "volumes/labels/pred_dist_er",
                ArrayKeys.GT_DIST_ER_MEM: "volumes/labels/gt_dist_er_mem",
                ArrayKeys.PRED_DIST_ER_MEM: "volumes/labels/pred_dist_er_mem",
                ArrayKeys.GT_DIST_MVB: "volumes/labels/gt_dist_mvb",
                ArrayKeys.PRED_DIST_MVB: "volumes/labels/pred_dist_mvb",
                ArrayKeys.GT_DIST_MVB_MEM: "volumes/labels/gt_dist_mvb_mem",
                ArrayKeys.PRED_DIST_MVB_MEM: "volumes/labels/pred_dist_mvb_mem",
                ArrayKeys.GT_DIST_MITO: "volumes/labels/gt_dist_mito",
                ArrayKeys.PRED_DIST_MITO: "volumes/labels/pred_dist_mito",
                ArrayKeys.GT_DIST_MITO_MEM: "volumes/labels/gt_dist_mito_mem",
                ArrayKeys.PRED_DIST_MITO_MEM: "volumes/labels/pred_dist_mito_mem",
                ArrayKeys.GT_DIST_LYSOSOME: "volumes/labels/gt_dist_lysosome",
                ArrayKeys.PRED_DIST_LYSOSOME: "volumes/labels/pred_dist_lysosome",
                ArrayKeys.GT_DIST_LYSOSOME_MEM: "volumes/labels/gt_dist_lysosome_mem",
                ArrayKeys.PRED_DIST_LYSOSOME_MEM: "volumes/labels/pred_dist_lysosome_mem",
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
    set_verbose(False)
    data_sources = ["gt_v1"]
    input_shape = (196, 196, 196)
    output_shape = (92, 92, 92)
    dt_scaling_factor = 100
    max_iteration = 400000
    loss_name = "loss_total_unbalanced"
    train_until(
        max_iteration,
        data_sources,
        input_shape,
        output_shape,
        dt_scaling_factor,
        loss_name,
    )
