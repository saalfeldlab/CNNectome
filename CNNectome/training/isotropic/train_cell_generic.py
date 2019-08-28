from gunpowder import *
from gunpowder.tensorflow import *
from gunpowder.contrib import (
    AddDistance,
    TanhSaturate,
    CombineDistances,
)
import fuse
import tensorflow as tf
import math
import json
import sys
import logging
import time

print("syspath", sys.path)
from CNNectome.utils.label import *
import numpy as np


def train_until(
    max_iteration,
    data_sources,
    ribo_sources,
    nucleolus_sources,
    centrosomes_sources,
    input_shape,
    output_shape,
    dt_scaling_factor,
    loss_name,
    labels,
    net_name,
    min_masked_voxels=17561.0,
    mask_ds_name="volumes/masks/training",
    integral_mask_ds_name="volumes/masks/training_integral",
):
    with open("net_io_names.json", "r") as f:
        net_io_names = json.load(f)

    ArrayKey("RAW")
    ArrayKey("ALPHA_MASK")
    ArrayKey("GT_LABELS")
    ArrayKey("MASK")
    ArrayKey("INTEGRAL_MASK")
    ArrayKey("RIBO_GT")
    ArrayKey("NUCLEOLUS_GT")
    ArrayKey("CENTROSOMES_GT")

    voxel_size_up = Coordinate((2, 2, 2))
    voxel_size_orig = Coordinate((4, 4, 4))
    input_size = Coordinate(input_shape) * voxel_size_orig
    output_size = Coordinate(output_shape) * voxel_size_orig
    # context = input_size-output_size

    keep_thr = float(min_masked_voxels) / np.prod(output_shape)

    data_providers = []
    inputs = dict()
    outputs = dict()
    snapshot = dict()
    request = BatchRequest()
    snapshot_request = BatchRequest()

    # datasets_ribo = {
    #     ArrayKeys.RAW:       'volumes/raw/data/s0',
    #     ArrayKeys.GT_LABELS: 'volumes/labels/all',
    #     ArrayKeys.MASK:      mask_ds_name,
    #     ArrayKeys.RIBO_GT:   'volumes/labels/ribosomes',
    # }
    # for datasets without ribosome annotations volumes/labels/ribosomes doesn't exist, so use volumes/labels/all
    # instead (only one with the right resolution)
    # datasets_no_ribo = {
    #     ArrayKeys.RAW:       'volumes/raw/data/s0',
    #     ArrayKeys.GT_LABELS: 'volumes/labels/all',
    #     ArrayKeys.MASK:      mask_ds_name,
    #     ArrayKeys.RIBO_GT:   'volumes/labels/all',
    # }
    datasets = {
        ArrayKeys.RAW: "volumes/raw",
        ArrayKeys.GT_LABELS: "volumes/labels/all",
        ArrayKeys.MASK: mask_ds_name,
        ArrayKeys.INTEGRAL_MASK: integral_mask_ds_name,
        ArrayKeys.RIBO_GT: "volumes/labels/all",
        ArrayKeys.NUCLEOLUS_GT: "volumes/labels/all",
        ArrayKeys.CENTROSOMES_GT: "volumes/labels/all",
    }

    array_specs = {
        ArrayKeys.MASK: ArraySpec(interpolatable=False),
        ArrayKeys.RAW: ArraySpec(voxel_size=Coordinate(voxel_size_orig)),
        ArrayKeys.INTEGRAL_MASK: ArraySpec(interpolatable=False),
    }
    array_specs_pred = {}

    inputs[net_io_names["raw"]] = ArrayKeys.RAW

    snapshot[ArrayKeys.RAW] = "volumes/raw"
    snapshot[ArrayKeys.GT_LABELS] = "volumes/labels/gt_labels"

    request.add(ArrayKeys.GT_LABELS, output_size, voxel_size=voxel_size_up)
    request.add(ArrayKeys.MASK, output_size, voxel_size=voxel_size_orig)
    request.add(ArrayKeys.RIBO_GT, output_size, voxel_size=voxel_size_up)
    request.add(ArrayKeys.NUCLEOLUS_GT, output_size, voxel_size=voxel_size_up)
    request.add(ArrayKeys.CENTROSOMES_GT, output_size, voxel_size=voxel_size_up)
    request.add(ArrayKeys.RAW, input_size, voxel_size=voxel_size_orig)
    request.add(ArrayKeys.INTEGRAL_MASK, output_size, voxel_size=voxel_size_orig)

    for label in labels:
        datasets[label.mask_key] = "volumes/masks/" + label.labelname

        array_specs[label.mask_key] = ArraySpec(interpolatable=False)
        array_specs_pred[label.pred_dist_key] = ArraySpec(
            voxel_size=voxel_size_orig, interpolatable=True
        )

        inputs[net_io_names["mask_" + label.labelname]] = label.mask_key
        inputs[net_io_names["gt_" + label.labelname]] = label.gt_dist_key
        if label.scale_loss or label.scale_key is not None:
            inputs[net_io_names["w_" + label.labelname]] = label.scale_key

        outputs[net_io_names[label.labelname]] = label.pred_dist_key

        snapshot[label.gt_dist_key] = "volumes/labels/gt_dist_" + label.labelname
        snapshot[label.pred_dist_key] = "volumes/labels/pred_dist_" + label.labelname

        request.add(label.gt_dist_key, output_size, voxel_size=voxel_size_orig)
        request.add(label.pred_dist_key, output_size, voxel_size=voxel_size_orig)
        request.add(label.mask_key, output_size, voxel_size=voxel_size_orig)
        if label.scale_loss:
            request.add(label.scale_key, output_size, voxel_size=voxel_size_orig)

        snapshot_request.add(
            label.pred_dist_key, output_size, voxel_size=voxel_size_orig
        )

    if tf.train.latest_checkpoint("."):
        trained_until = int(tf.train.latest_checkpoint(".").split("_")[-1])
        print("Resuming training from", trained_until)
    else:
        trained_until = 0
        print("Starting fresh training")

    for src in data_sources:

        datasets_i = datasets.copy()
        if src in ribo_sources:
            datasets_i[ArrayKeys.RIBO_GT] = "volumes/labels/ribosomes"
        if src in nucleolus_sources:
            datasets_i[ArrayKeys.NUCLEOLUS_GT] = "volumes/labels/nucleolus"
        if src in centrosomes_sources:
            datasets_i[ArrayKeys.CENTROSOMES_GT] = "volumes/labels/centrosomes"

        n5_source = N5Source(
            src.full_path, datasets=datasets_i, array_specs=array_specs
        )
        # if src not in ribo_sources:
        #     n5_source = N5Source(
        #         src.full_path,
        #         datasets=datasets_no_ribo,
        #         array_specs=array_specs
        #     )
        # else:
        #     n5_source = N5Source(
        #         src.full_path,
        #         datasets=datasets_ribo,
        #         array_specs=array_specs
        #     )

        data_providers.append(n5_source)

    # create a tuple of data sources, one for each HDF file
    data_stream = tuple(
        provider + Normalize(ArrayKeys.RAW) +  # ensures RAW is in float in [0, 1]
        # zero-pad provided RAW and MASK to be able to draw batches close to
        # the boundary of the available data
        # size more or less irrelevant as followed by Reject Node
        # Pad(ArrayKeys.RAW, context) +
        RandomLocationWithIntegralMask(
            keep_thr, integral_mask=ArrayKeys.INTEGRAL_MASK
        )  # chose a random location inside the
        # RandomLocation() +
        # provided arrays
        # Reject(ArrayKeys.MASK, min_masked=keep_thr)
        # Reject(ArrayKeys.MASK) # reject batches wich do contain less than 50% labelled data
        for provider in data_providers
    )

    train_pipeline = (
        data_stream
        + RandomProvider(tuple([ds.labeled_voxels for ds in data_sources]))
        + fuse.SimpleAugment()
        + fuse.ElasticAugment(
            voxel_size_orig,
            (100, 100, 100),
            (10.0, 10.0, 10.0),
            (0, math.pi / 2.0),
            spatial_dims=3,
            subsample=8,
        )
        +
        # ElasticAugment((40, 1000, 1000), (10., 0., 0.), (0, 0), subsample=8) +
        fuse.IntensityAugment(ArrayKeys.RAW, 0.25, 1.75, -0.5, 0.35)
        + GammaAugment(ArrayKeys.RAW, 0.5, 2.0)
        + IntensityScaleShift(ArrayKeys.RAW, 2, -1)
    )
    # ZeroOutConstSections(ArrayKeys.RAW))

    for label in labels:
        if label.labelname == "ribosomes":
            train_pipeline += AddDistance(
                label_array_key=ArrayKeys.RIBO_GT,
                distance_array_key=label.gt_dist_key,
                mask_array_key=label.mask_key,
                add_constant=8,
                label_id=label.labelid,
                factor=2,
                max_distance=2.76 * dt_scaling_factor,
            )
        elif label.labelname == "nucleolus":
            train_pipeline += AddDistance(
                label_array_key=ArrayKeys.NUCLEOLUS_GT,
                distance_array_key=label.gt_dist_key,
                mask_array_key=label.mask_key,
                label_id=label.labelid,
                factor=2,
                max_distance=2.76 * dt_scaling_factor,
            )
        elif label.labelname == "centrosomes":
            train_pipeline += AddDistance(
                label_array_key=ArrayKeys.CENTROSOMES_GT,
                distance_array_key=label.gt_dist_key,
                mask_array_key=label.mask_key,
                add_constant=2,
                label_id=label.labelid,
                factor=2,
                max_distance=2.76 * dt_scaling_factor,
            )
        else:
            train_pipeline += AddDistance(
                label_array_key=ArrayKeys.GT_LABELS,
                distance_array_key=label.gt_dist_key,
                mask_array_key=label.mask_key,
                label_id=label.labelid,
                factor=2,
                max_distance=2.76 * dt_scaling_factor,
            )
    for label in labels:
        if label.labelname == "microtubules_out":
            microtubules = label
        elif label.labelname == "centrosomes":
            centrosomes = label
        elif label.labelname == "subdistal_app":
            subdistal_app = label
        elif label.labelname == "distal_app":
            distal_app = label
    train_pipeline += CombineDistances(
        (microtubules.gt_dist_key, centrosomes.gt_dist_key),
        microtubules.gt_dist_key,
        (microtubules.mask_key, centrosomes.mask_key),
        microtubules.mask_key,
    )

    train_pipeline += CombineDistances(
        (distal_app.gt_dist_key, subdistal_app.gt_dist_key, centrosomes.gt_dist_key),
        centrosomes.gt_dist_key,
        (distal_app.mask_key, subdistal_app.mask_key, centrosomes.mask_key),
        centrosomes.mask_key,
    )

    for label in labels:
        train_pipeline += TanhSaturate(label.gt_dist_key, dt_scaling_factor)
    for label in labels:
        if label.scale_loss:
            train_pipeline += BalanceByThreshold(
                label.gt_dist_key, label.scale_key, mask=label.mask_key
            )

    train_pipeline = (
        train_pipeline
        + PreCache(cache_size=30, num_workers=30)
        + Train(
            net_name,
            optimizer=net_io_names["optimizer"],
            loss=net_io_names[loss_name],
            inputs=inputs,
            summary=net_io_names["summary"],
            log_dir="log",
            outputs=outputs,
            gradients={},
            log_every=5,
            save_every=500,
            array_specs=array_specs_pred,
        )
        + Snapshot(
            snapshot,
            every=500,
            output_filename="batch_{iteration}.hdf",
            output_dir="snapshots/",
            additional_request=snapshot_request,
        )
        + PrintProfilingStats(every=500)
    )

    print("Starting training...")
    with build(train_pipeline) as b:
        for i in range(max_iteration):
            start_it = time.time()
            b.request_batch(request)
            time_it = time.time() - start_it
            logging.info("it {0:}: {1:}".format(i + 1, time_it))

    print("Training finished")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    data_dir = (
        "/groups/saalfeld/saalfeldlab/larissa/data/cell/multires/v061719_o750x750x750_m1800x1800x1800_8nm/{"
        "0:}.n5"
    )
    data_sources = list()
    data_sources.append(N5Dataset("crop1", 500 * 500 * 100, data_dir=data_dir))
    data_sources.append(N5Dataset("crop3", 400 * 400 * 250, data_dir=data_dir))
    data_sources.append(
        N5Dataset(
            "crop4",
            300 * 300 * 238,
            special_categories=("centrosomes",),
            data_dir=data_dir,
        )
    )
    data_sources.append(N5Dataset("crop6", 250 * 250 * 250, data_dir=data_dir))
    data_sources.append(N5Dataset("crop7", 300 * 300 * 80, data_dir=data_dir))
    data_sources.append(N5Dataset("crop8", 200 * 200 * 100, data_dir=data_dir))
    data_sources.append(N5Dataset("crop9", 100 * 100 * 53, data_dir=data_dir))
    data_sources.append(N5Dataset("crop13", 160 * 160 * 110, data_dir=data_dir))
    data_sources.append(N5Dataset("crop14", 150 * 150 * 65, data_dir=data_dir))
    data_sources.append(N5Dataset("crop15", 150 * 150 * 64, data_dir=data_dir))
    data_sources.append(
        N5Dataset(
            "crop16",
            200 * 200 * 200,
            special_categories=("ribosomes", "nucleolus"),
            data_dir=data_dir,
        )
    )
    data_sources.append(N5Dataset("crop18", 200 * 200 * 110, data_dir=data_dir))
    data_sources.append(N5Dataset("crop19", 150 * 150 * 55, data_dir=data_dir))
    data_sources.append(N5Dataset("crop20", 200 * 200 * 85, data_dir=data_dir))
    data_sources.append(N5Dataset("crop21", 160 * 160 * 55, data_dir=data_dir))
    data_sources.append(N5Dataset("crop22", 170 * 170 * 100, data_dir=data_dir))
    data_sources.append(N5Dataset("crop31", 150 * 150 * 150, data_dir=data_dir))
    data_sources.append(N5Dataset("crop33", 200 * 200 * 200, data_dir=data_dir))
    data_sources.append(N5Dataset("crop34", 200 * 200 * 200, data_dir=data_dir))

    ribo_sources = filter_by_category(data_sources, "ribosomes")
    nucleolus_sources = filter_by_category(data_sources, "nucleolus")
    centrosomes_sources = filter_by_category(data_sources, "centrosomes")
    input_shape = (196, 196, 196)
    # output_shape = (92, 92, 92)
    dt_scaling_factor = 50
    max_iteration = 500000
    loss_name = "loss_total"

    labels = list()
    labels.append(Label("ecs", 1, data_sources=data_sources, data_dir=data_dir))
    labels.append(
        Label("plasma_membrane", 2, data_sources=data_sources, data_dir=data_dir)
    )
    labels.append(
        Label("mito", (3, 4, 5), data_sources=data_sources, data_dir=data_dir)
    )
    labels.append(
        Label(
            "mito_membrane",
            3,
            scale_loss=False,
            scale_key=labels[-1].scale_key,
            data_sources=data_sources,
            data_dir=data_dir,
        )
    )
    labels.append(
        Label(
            "mito_DNA",
            5,
            scale_loss=False,
            scale_key=labels[-2].scale_key,
            data_sources=data_sources,
            data_dir=data_dir,
        )
    )
    labels.append(Label("golgi", (6, 7), data_sources=data_sources, data_dir=data_dir))
    labels.append(
        Label("golgi_membrane", 6, data_sources=data_sources, data_dir=data_dir)
    )
    labels.append(
        Label("vesicle", (8, 9), data_sources=data_sources, data_dir=data_dir)
    )
    labels.append(
        Label(
            "vesicle_membrane",
            8,
            scale_loss=False,
            scale_key=labels[-1].scale_key,
            data_sources=data_sources,
            data_dir=data_dir,
        )
    )
    labels.append(Label("MVB", (10, 11), data_sources=data_sources, data_dir=data_dir))
    labels.append(
        Label(
            "MVB_membrane",
            10,
            scale_loss=False,
            scale_key=labels[-1].scale_key,
            data_sources=data_sources,
            data_dir=data_dir,
        )
    )
    labels.append(
        Label("lysosome", (12, 13), data_sources=data_sources, data_dir=data_dir)
    )
    labels.append(
        Label(
            "lysosome_membrane",
            12,
            scale_loss=False,
            scale_key=labels[-1].scale_key,
            data_sources=data_sources,
            data_dir=data_dir,
        )
    )
    labels.append(Label("LD", (14, 15), data_sources=data_sources, data_dir=data_dir))
    labels.append(
        Label(
            "LD_membrane",
            14,
            scale_loss=False,
            scale_key=labels[-1].scale_key,
            data_sources=data_sources,
            data_dir=data_dir,
        )
    )
    labels.append(
        Label(
            "er",
            (16, 17, 18, 19, 20, 21, 22, 23),
            data_sources=data_sources,
            data_dir=data_dir,
        )
    )
    labels.append(
        Label(
            "er_membrane",
            (16, 18, 20),
            scale_loss=False,
            scale_key=labels[-1].scale_key,
            data_sources=data_sources,
            data_dir=data_dir,
        )
    )
    labels.append(Label("ERES", (18, 19), data_sources=data_sources, data_dir=data_dir))
    # labels.append(Label('ERES_membrane', 18, scale_loss=False, scale_key=labels[-1].scale_key,
    #                    data_sources=data_sources, data_dir=data_dir))
    labels.append(
        Label(
            "nucleus",
            (20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 36),
            data_sources=data_sources,
            data_dir=data_dir,
        )
    )
    labels.append(
        Label("nucleolus", 29, data_sources=nucleolus_sources, data_dir=data_dir)
    )
    labels.append(
        Label("NE", (20, 21, 22, 23), data_sources=data_sources, data_dir=data_dir)
    )
    labels.append(
        Label(
            "NE_membrane",
            (20, 22, 23),
            scale_loss=False,
            scale_key=labels[-1].scale_key,
            data_sources=data_sources,
            data_dir=data_dir,
        )
    )
    labels.append(
        Label("nuclear_pore", (22, 23), data_sources=data_sources, data_dir=data_dir)
    )
    labels.append(
        Label("nuclear_pore_out", 22, scale_loss=False, scale_key=labels[-1].scale_key)
    )
    labels.append(
        Label(
            "chromatin", (24, 25, 26, 27), data_sources=data_sources, data_dir=data_dir
        )
    )
    # labels.append(Label('NHChrom', 25, scale_loss=False, scale_key=labels[-1].scale_key))
    # labels.append(Label('EChrom', 26, scale_loss=False, scale_key=labels[-2].scale_key))
    # labels.append(Label('NEChrom', 27, scale_loss=False, scale_key=labels[-3].scale_key))
    labels.append(Label("NHChrom", 25, data_sources=data_sources, data_dir=data_dir))
    labels.append(Label("EChrom", 26, data_sources=data_sources, data_dir=data_dir))
    labels.append(Label("NEChrom", 27, data_sources=data_sources, data_dir=data_dir))
    labels.append(
        Label("microtubules", (30, 36), data_sources=data_sources, data_dir=data_dir)
    )
    labels.append(
        Label(
            "microtubules_out",
            (30,),
            scale_loss=False,
            scale_key=labels[-1].scale_key,
            data_sources=data_sources,
            data_dir=data_dir,
        )
    )
    labels.append(
        Label("centrosomes", 255, data_sources=centrosomes_sources, data_dir=data_dir)
    )
    labels.append(Label("distal_app", 32, data_sources=data_sources, data_dir=data_dir))
    labels.append(
        Label("subdistal_app", 33, data_sources=data_sources, data_dir=data_dir)
    )
    labels.append(Label("ribosomes", 1, data_sources=ribo_sources, data_dir=data_dir))
    # labels.append(Label('HChrom_points', 1, data_sources=hchrom_sources, data_dir=data_dir))
    # labels.append(Label('EChrom_points', 1, data_sources=echrom_sources, data_dir=data_dir))
    make_net(labels, (340, 340, 340), mode="inference")
    tf.reset_default_graph()
    net_name, output_shape = make_net(
        labels, input_shape, mode="train", loss_name=loss_name
    )
    train_until(
        max_iteration,
        data_sources,
        ribo_sources,
        input_shape,
        output_shape,
        dt_scaling_factor,
        loss_name,
        labels,
        net_name,
    )
