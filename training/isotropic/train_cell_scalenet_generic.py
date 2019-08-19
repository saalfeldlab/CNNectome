from gunpowder import *
from gunpowder.tensorflow import *
from gunpowder.contrib import (
    ZeroOutConstSections,
    AddDistance,
    TanhSaturate,
    CombineDistances,
)
import gpn
import tensorflow as tf
import numpy as np
import os
import math
import json
import sys
import logging
import time
import collections

print("syspath", sys.path)
import z5py
from networks import scale_net
from networks.isotropic.mk_scale_net_cell_generic import *
from utils.label import *


def train_until(
    max_iteration,
    data_sources,
    ribo_sources,
    nucleolus_sources,
    centrosomes_sources,
    dt_scaling_factor,
    loss_name,
    labels,
    scnet,
    raw_names,
    min_masked_voxels=17561.0,
    mask_ds_name="volumes/masks/training",
    integral_mask_ds_name="volumes/masks/training_integral",
):
    with open("net_io_names.json", "r") as f:
        net_io_names = json.load(f)

    ArrayKey("ALPHA_MASK")
    ArrayKey("GT_LABELS")
    ArrayKey("MASK")
    ArrayKey("INTEGRAL_MASK")
    ArrayKey("RIBO_GT")
    ArrayKey("NUCLEOLUS_GT")
    ArrayKey("CENTROSOMES_GT")

    raw_array_keys = []
    contexts = []
    # input and output sizes in world coordinates
    input_sizes_wc = [
        Coordinate(inp_sh) * Coordinate(vs)
        for inp_sh, vs in zip(scnet.input_shapes, scnet.voxel_sizes)
    ]
    output_size_wc = Coordinate(scnet.output_shapes[0]) * Coordinate(
        scnet.voxel_sizes[0]
    )
    keep_thr = float(min_masked_voxels) / np.prod(scnet.output_shapes[0])

    voxel_size_up = Coordinate((2, 2, 2))
    voxel_size_orig = Coordinate((4, 4, 4))
    assert voxel_size_orig == Coordinate(
        scnet.voxel_sizes[0]
    )  # make sure that scnet has the same base voxel siz

    data_providers = []
    inputs = dict()
    outputs = dict()
    snapshot = dict()
    request = BatchRequest()
    snapshot_request = BatchRequest()
    datasets = {
        ArrayKeys.GT_LABELS: "volumes/labels/all",
        ArrayKeys.MASK: mask_ds_name,
        ArrayKeys.INTEGRAL_MASK: integral_mask_ds_name,
        ArrayKeys.RIBO_GT: "volumes/labels/all",
        ArrayKeys.NUCLEOLUS_GT: "volumes/labels/all",
        ArrayKeys.CENTROSOMES_GT: "volumes/labels/all",
    }

    array_specs = {ArrayKeys.MASK: ArraySpec(interpolatable=False)}
    array_specs_pred = {}

    # inputs = {net_io_names['mask']: ArrayKeys.MASK}
    snapshot[ArrayKeys.GT_LABELS] = "volumes/labels/gt_labels"

    for k, (inp_sh_wc, vs, raw_name) in enumerate(
        zip(input_sizes_wc, scnet.voxel_sizes, raw_names)
    ):
        ak = ArrayKey("RAW_S{0:}".format(k))
        raw_array_keys.append(ak)
        datasets[ak] = raw_name
        inputs[net_io_names["raw_{0:}".format(vs[0])]] = ak
        snapshot[ak] = "volumes/raw_s{0:}".format(k)
        array_specs[ak] = ArraySpec(voxel_size=Coordinate(vs))
        request.add(ak, inp_sh_wc, voxel_size=Coordinate(vs))
        contexts.append(inp_sh_wc - output_size_wc)

    request.add(ArrayKeys.GT_LABELS, output_size_wc, voxel_size=voxel_size_up)
    request.add(ArrayKeys.MASK, output_size_wc, voxel_size=voxel_size_orig)
    request.add(ArrayKeys.RIBO_GT, output_size_wc, voxel_size=voxel_size_up)
    request.add(ArrayKeys.NUCLEOLUS_GT, output_size_wc, voxel_size=voxel_size_up)
    request.add(ArrayKeys.CENTROSOMES_GT, output_size_wc, voxel_size=voxel_size_up)
    request.add(ArrayKeys.INTEGRAL_MASK, output_size_wc, voxel_size=voxel_size_orig)

    # individual mask per label
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

        request.add(label.gt_dist_key, output_size_wc, voxel_size=voxel_size_orig)
        request.add(label.pred_dist_key, output_size_wc, voxel_size=voxel_size_orig)
        request.add(label.mask_key, output_size_wc, voxel_size=voxel_size_orig)
        if label.scale_loss:
            request.add(label.scale_key, output_size_wc, voxel_size=voxel_size_orig)

        snapshot_request.add(
            label.pred_dist_key, output_size_wc, voxel_size=voxel_size_orig
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
        data_providers.append(n5_source)

    data_stream = []
    for provider in data_providers:
        data_stream.append(provider)
        for ak, context in zip(raw_array_keys, contexts):
            data_stream[-1] += Normalize(ak)
            # data_stream[-1] += Pad(ak, context) # this shouldn't be necessary as I cropped the input data to have
            # sufficient padding
        data_stream[-1] += RandomLocationWithIntegralMask(
            keep_thr, integral_mask=ArrayKeys.INTEGRAL_MASK
        )
    data_stream = tuple(data_stream)

    train_pipeline = (
        data_stream
        + RandomProvider(tuple([ds.labeled_voxels for ds in data_sources]))
        + gpn.SimpleAugment()
        + gpn.ElasticAugment(
            tuple(scnet.voxel_sizes[0]),
            (100, 100, 100),
            (10.0, 10.0, 10.0),
            (0, math.pi / 2.0),
            spatial_dims=3,
            subsample=8,
        )
        + gpn.IntensityAugment(raw_array_keys, 0.25, 1.75, -0.5, 0.35)
        + GammaAugment(raw_array_keys, 0.5, 2.0)
    )
    for ak in raw_array_keys:
        train_pipeline += IntensityScaleShift(ak, 2, -1)

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
        if label.scale_loss:
            train_pipeline += BalanceByThreshold(
                label.gt_dist_key, label.scale_key, mask=label.mask_key
            )

    train_pipeline = (
        train_pipeline
        + PreCache(cache_size=30, num_workers=30)
        + Train(
            scnet.name,
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
        + PrintProfilingStats(every=50)
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
    data_sources.append(N5Dataset("crop4", 300 * 300 * 238, data_dir=data_dir))
    data_sources.append(
        N5Dataset(
            "crop6",
            250 * 250 * 250,
            special_categories=("ribosomes",),
            data_dir=data_dir,
        )
    )
    data_sources.append(
        N5Dataset(
            "crop7",
            300 * 300 * 80,
            special_categories=("ribosomes",),
            data_dir=data_dir,
        )
    )
    data_sources.append(N5Dataset("crop8", 200 * 200 * 100, data_dir=data_dir))
    data_sources.append(N5Dataset("crop9", 100 * 100 * 53, data_dir=data_dir))
    data_sources.append(
        N5Dataset(
            "crop13",
            160 * 160 * 110,
            special_categories=("ribosomes",),
            data_dir=data_dir,
        )
    )
    data_sources.append(N5Dataset("crop14", 150 * 150 * 65, data_dir=data_dir))
    data_sources.append(N5Dataset("crop15", 150 * 150 * 64, data_dir=data_dir))
    data_sources.append(N5Dataset("crop18", 200 * 200 * 110, data_dir=data_dir))
    data_sources.append(N5Dataset("crop19", 150 * 150 * 55, data_dir=data_dir))
    data_sources.append(N5Dataset("crop20", 200 * 200 * 85, data_dir=data_dir))
    data_sources.append(N5Dataset("crop21", 160 * 160 * 55, data_dir=data_dir))
    data_sources.append(N5Dataset("crop22", 170 * 170 * 100, data_dir=data_dir))

    ribo_sources = filter_by_category(data_sources, "ribosomes")

    # input_shape = (196, 196, 196)
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
    labels.append(Label("nucleolus", 29, data_sources=data_sources, data_dir=data_dir))
    labels.append(
        Label(
            "NE",
            (20, 21, 22, 23),
            scale_loss=False,
            scale_key=labels[-1].scale_key,
            data_sources=data_sources,
            data_dir=data_dir,
        )
    )
    # labels.append(Label('NE_membrane', (20, 22, 23), scale_loss=False, scale_key=labels[-1].scale_key,
    # data_sources=data_sources, data_dir=data_dir))
    labels.append(
        Label("nuclear_pore", (22, 23), data_sources=data_sources, data_dir=data_dir)
    )
    labels.append(
        Label(
            "nuclear_pore_out",
            22,
            scale_loss=False,
            scale_key=labels[-1].scale_key,
            data_sources=data_sources,
            data_dir=data_dir,
        )
    )
    labels.append(
        Label(
            "chromatin",
            (24, 25, 26, 27, 36),
            data_sources=data_sources,
            data_dir=data_dir,
        )
    )
    # labels.append(Label('NHChrom', 25, scale_loss=False, scale_key=labels[-1].scale_key, data_sources=data_sources,
    # data_dir=data_dir))
    # labels.append(Label('EChrom', 26, scale_loss=False, scale_key=labels[-2].scale_key, data_sources=data_sources,
    # data_dir=data_dir))
    # labels.append(Label('NEChrom', 27, scale_loss=False, scale_key=labels[-3].scale_key, data_sources=data_sources,
    # data_dir=data_dir))
    labels.append(Label("NHChrom", 25, data_sources=data_sources, data_dir=data_dir))
    labels.append(Label("EChrom", 26, data_sources=data_sources, data_dir=data_dir))
    labels.append(Label("NEChrom", 27, data_sources=data_sources, data_dir=data_dir))
    labels.append(
        Label("microtubules", (30, 31), data_sources=data_sources, data_dir=data_dir)
    )
    labels.append(
        Label("centrosome", (31, 32, 33), data_sources=data_sources, data_dir=data_dir)
    )
    labels.append(Label("distal_app", 32, data_sources=data_sources, data_dir=data_dir))
    labels.append(
        Label("subdistal_app", 33, data_sources=data_sources, data_dir=data_dir)
    )
    labels.append(Label("ribosomes", 1, data_sources=ribo_sources, data_dir=data_dir))

    unet0 = scale_net.SerialUNet(
        [12, 12 * 6, 12 * 6 ** 2],
        [48, 12 * 6, 12 * 6 ** 2],
        [(3, 3, 3), (3, 3, 3)],
        [[(3, 3, 3), (3, 3, 3)], [(3, 3, 3), (3, 3, 3)], [(3, 3, 3), (3, 3, 3)]],
        [[(3, 3, 3), (3, 3, 3)], [(3, 3, 3), (3, 3, 3)]],
        input_voxel_size=(4, 4, 4),
    )
    unet1 = scale_net.SerialUNet(
        [12, 12 * 6, 12 * 6 ** 2],
        [12 * 6 ** 2, 12 * 6 ** 2, 12 * 6 ** 2],
        [(3, 3, 3), (3, 3, 3)],
        [[(3, 3, 3), (3, 3, 3)], [(3, 3, 3), (3, 3, 3)], [(3, 3, 3), (3, 3, 3)]],
        [[(3, 3, 3), (3, 3, 3)], [(3, 3, 3), (3, 3, 3)]],
        input_voxel_size=(36, 36, 36),
    )
    make_any_scale_net([unet0, unet1], labels, 4, mode="inference")
    tf.reset_default_graph()
    train_sc_net = make_any_scale_net(
        [unet0, unet1], labels, 5, mode="train", loss_name=loss_name
    )
    train_until(
        max_iteration,
        data_sources,
        ribo_sources,
        dt_scaling_factor,
        loss_name,
        labels,
        train_sc_net,
    )
    # train_until(max_iteration, data_sources, labeled_voxels, ribo_sources, input_shape, output_shape,
    #            dt_scaling_factor, loss_name,
    #            labels)
