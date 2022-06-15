from gunpowder import *
from gunpowder.tensorflow import *
from gunpowder.contrib import IntensityCrop, ImpulseNoiseAugment
from gunpowder.ext import zarr
from gunpowder.compat import ensure_str
from CNNectome.validation.denoising.metrics import compute_metric
import fuse
import corditea
import tensorflow as tf
import math
import time
import itertools
import json
import os
import numpy as np
import logging
import typing


def network_setup(net_name, load_checkpoint=True, mode="training"):
    # load graph's io names
    with open("{0:}_io_names.json".format(net_name), "r") as f:
        net_io_names = json.load(f)
    if load_checkpoint:
        # find checkpoint from previous training, start a new one if not found
        if tf.train.latest_checkpoint("."):
            start_iteration = int(tf.train.latest_checkpoint(".").split("_")[-1])
            if start_iteration >= max_iteration:
                logging.info(
                    "Network has already been trained for {0:} iterations".format(
                        start_iteration
                    )
                )
            else:
                logging.info("Resuming training from {0:}".format(start_iteration))
        else:
            start_iteration = 0
            logging.info("Starting fresh training")
    else:
        start_iteration = None
    # define network inputs
    inputs = dict()
    inputs[net_io_names["raw_input"]] = ArrayKey("RAW_INPUT")
    if mode == "training" or mode == "forward":
        inputs[net_io_names["raw_target"]] = ArrayKey("RAW_TARGET")
    # define network outputs
    outputs = dict()
    outputs[net_io_names["raw_predicted"]] = ArrayKey("RAW_PREDICTED")
    return net_io_names, start_iteration, inputs, outputs


def batch_generator(
    data_path,
    raw_dataset,
    input_shape,
    output_shape,
    augmentations,
    intensity_scale_range=(0.75, 1.25),
    intensity_shift_range=(-0.2, 0.2),
    gamma_range=(0.75, 4 / 3.0),
    impulse_noise_prob=0.05,
    prob_missing=0.05,
    prob_low_contrast=0.05,
    contrast_scale=0.1,
    voxel_size=Coordinate((4, 4, 4)),
):

    input_size = Coordinate(input_shape) * voxel_size
    output_size = Coordinate(output_shape) * voxel_size

    # construct batch request
    request = BatchRequest()
    source_keys = dict()
    request.add(ArrayKey("RAW_INPUT"), input_size, voxel_size=voxel_size)
    source_keys[ArrayKey("RAW_INPUT")] = raw_dataset
    request.add(ArrayKey("RAW_TARGET"), output_size, voxel_size=voxel_size)
    source_keys[ArrayKey("RAW_TARGET")] = raw_dataset
    request.add(ArrayKey("RAW_PREDICTED"), output_size, voxel_size=voxel_size)
    request.add(ArrayKey("DATA_MASK"), input_size, voxel_size=voxel_size)

    # specify specs for output
    array_specs_pred = {
        ArrayKey("RAW_PREDICTED"): ArraySpec(voxel_size=voxel_size, interpolatable=True)
    }

    source_specs = {}
    for ak in source_keys.keys():
        source_specs[ak] = ArraySpec(voxel_size=voxel_size)
    src = ZarrSource(data_path, source_keys, source_specs)

    pipeline = src + RandomLocation()
    mask_lambda = lambda val: ((val > 0) * 1).astype(np.bool)
    pipeline += LambdaFilter(
        mask_lambda,
        source_key=ArrayKey("RAW_INPUT"),
        target_key=ArrayKey("DATA_MASK"),
        target_spec=ArraySpec(dtype=np.bool, interpolatable=False),
    )
    pipeline += Reject(mask=ArrayKey("DATA_MASK"))
    pipeline += Normalize(ArrayKey("RAW_INPUT"))
    pipeline += Normalize(ArrayKey("RAW_TARGET"))
    # pipeline += IntensityCrop(ArrayKey("RAW_INPUT"), 0., 1.)
    # pipeline += IntensityCrop(ArrayKey("RAW_TARGET"), 0., 1.)

    # augmentations
    for aug in augmentations:
        if aug == "simple":
            pipeline += fuse.SimpleAugment()
        elif aug == "elastic":
            pipeline += fuse.ElasticAugment(
                voxel_size,
                (100, 100, 100),
                (10.0, 10.0, 10.0),
                (0, math.pi / 2.0),
                spatial_dims=3,
                subsample=8,
            )
        # elif aug == "intensity_narrow" or aug == "intensity":
        #     pipeline += fuse.IntensityAugment([ArrayKey("RAW_INPUT"), ArrayKey("RAW_TARGET")], 0.75, 1.25, -0.2, 0.2)
        #     pipeline += GammaAugment([ArrayKey("RAW_INPUT"), ArrayKey("RAW_TARGET")], 0.75, 4/3.)
        # elif aug == "intensity_wide":
        #     pipeline += fuse.IntensityAugment([ArrayKey("RAW_INPUT"), ArrayKey("RAW_TARGET")], 0.25, 1.75, -0.5, 0.35)
        #     pipeline += GammaAugment([ArrayKey("RAW_INPUT"), ArrayKey("RAW_TARGET")], 0.5, 2.)
        elif aug == "intensity":
            pipeline += fuse.IntensityAugment(
                [ArrayKey("RAW_INPUT"), ArrayKey("RAW_TARGET")],
                intensity_scale_range[0],
                intensity_scale_range[1],
                intensity_shift_range[0],
                intensity_shift_range[1],
            )
        elif aug == "gamma":
            pipeline += GammaAugment(
                [ArrayKey("RAW_INPUT"), ArrayKey("RAW_TARGET")],
                gamma_range[0],
                gamma_range[1],
            )
        elif aug == "poisson":
            pipeline += NoiseAugment(ArrayKey("RAW_INPUT"), "poisson", clip=True)
        elif aug == "impulse_noise":
            pipeline += ImpulseNoiseAugment(ArrayKey("RAW_INPUT"), impulse_noise_prob)
        elif aug == "defect":
            pipeline += DefectAugment(
                ArrayKey("RAW_INPUT"),
                prob_missing=prob_missing,
                prob_low_contrast=prob_low_contrast,
                contrast_scale=contrast_scale,
            )
        else:
            raise ValueError("")
    pipeline += corditea.Multiply(
        (ArrayKey("RAW_INPUT"), ArrayKey("DATA_MASK")),
        ArrayKey("RAW_INPUT"),
        target_spec=ArraySpec(dtype=np.float32, interpolatable=True),
    )
    pipeline += IntensityScaleShift(ArrayKey("RAW_INPUT"), 2, -1)
    pipeline += IntensityScaleShift(ArrayKey("RAW_TARGET"), 2, -1)
    return pipeline, request, array_specs_pred


def train_until(
    max_iteration: int,
    net_name: str,
    data_path: str,
    raw_dataset: str,
    input_shape: np.ndarray,
    output_shape: np.ndarray,
    augmentations: typing.List[str],
    cache_size: int = 5,
    num_workers: int = 10,
    intensity_scale_range: typing.Tuple[float, float] = (0.75, 1.25),
    intensity_shift_range: typing.Tuple[float, float] = (-0.2, 0.2),
    gamma_range=(0.75, 4 / 3.0),
    impulse_noise_prob=0.05,
    prob_missing=0.05,
    prob_low_contrast=0.05,
    contrast_scale=0.1,
    voxel_size=Coordinate((4, 4, 4)),
):

    pipeline, request, array_specs_pred = batch_generator(
        data_path,
        raw_dataset,
        input_shape,
        output_shape,
        augmentations,
        intensity_scale_range=intensity_scale_range,
        intensity_shift_range=intensity_shift_range,
        gamma_range=gamma_range,
        impulse_noise_prob=impulse_noise_prob,
        prob_missing=prob_missing,
        prob_low_contrast=prob_low_contrast,
        contrast_scale=contrast_scale,
        voxel_size=voxel_size,
    )
    net_io_names, start_iteration, inputs, outputs = network_setup(net_name)
    # specify snapshot data layout
    snapshot_data = dict()
    snapshot_data[ArrayKey("RAW_INPUT")] = "volumes/raw_input"
    snapshot_data[ArrayKey("RAW_TARGET")] = "volumes/raw_target"
    snapshot_data[ArrayKey("RAW_PREDICTED")] = "volumes/raw_predicted"

    # specify snapshot request
    snapshot_request = BatchRequest()
    pipeline = (
        pipeline
        + PreCache(cache_size=cache_size, num_workers=num_workers)
        + Train(
            net_name + "_training",
            optimizer=net_io_names["optimizer"],
            loss=net_io_names["loss"],
            inputs=inputs,
            summary=net_io_names["summary"],
            log_dir="log",
            outputs=outputs,
            gradients={},
            log_every=10,
            save_every=500,
            array_specs=array_specs_pred,
        )
        + Snapshot(
            snapshot_data,
            every=500,
            output_filename="batch_{iteration}.hdf",
            output_dir="snapshots/",
            additional_request=snapshot_request,
        )
        + PrintProfilingStats(every=50)
    )

    logging.info("Starting training...")
    with build(pipeline) as pp:
        for i in range(start_iteration, max_iteration + 1):
            start_it = time.time()
            pp.request_batch(request)
            time_it = time.time() - start_it
            logging.info("it{0:}: {1:}".format(i + 1, time_it))
        pp.internal_teardown()
    logging.info("Training finished")


def metric_in_graph(metric: str, net_name: str) -> bool:
    with open("{0:}_io_names.json".format(net_name), "r") as f:
        net_io_names = json.load(f)
    return metric in net_io_names.keys()


def evaluate_metric(
    eval_iterations: int,
    metric: str,
    net_name: typing.Optional[str],
    iteration: typing.Optional[int],
    data_path: str,
    raw_dataset: str,
    input_shape: np.ndarray,
    output_shape: np.ndarray,
    augmentations: typing.List[str],
    cache_size: int = 5,
    num_workers: int = 10,
    intensity_scale_range: typing.Tuple[float, float] = (0.75, 1.25),
    intensity_shift_range: typing.Tuple[float, float] = (0.75, 1.25),
    gamma_range: typing.Tuple[float, float] = (0.75, 4 / 3.0),
    impulse_noise_prob: float = 0.05,
    prob_missing: float = 0.05,
    prob_low_contrast: float = 0.05,
    contrast_scale: float = 0.1,
    voxel_size: typing.Union[Coordinate, typing.Tuple[int, int, int]] = Coordinate(
        (4, 4, 4)
    ),
) -> typing.List[float]:
    pipeline, request, array_specs_pred = batch_generator(
        data_path,
        raw_dataset,
        input_shape,
        output_shape,
        augmentations,
        intensity_scale_range=intensity_scale_range,
        intensity_shift_range=intensity_shift_range,
        gamma_range=gamma_range,
        impulse_noise_prob=impulse_noise_prob,
        prob_missing=prob_missing,
        prob_low_contrast=prob_low_contrast,
        contrast_scale=contrast_scale,
        voxel_size=voxel_size,
    )

    if net_name is not None:
        pipeline + PreCache(cache_size=cache_size, num_workers=num_workers)
        if metric_in_graph(metric, net_name):
            net_io_names, _, inputs, outputs = network_setup(
                net_name, load_checkpoint=False, mode="forward"
            )
            graph = "{net_name:}_forward.meta".format(net_name=net_name)
            ak_metric = ArrayKey(metric.upper())
            outputs[net_io_names[metric]] = ak_metric
            request.add(
                ak_metric, voxel_size, voxel_size=voxel_size
            )  # todo: potential issue with shape here
        else:
            net_io_names, _, inputs, outputs = network_setup(
                net_name, load_checkpoint=False, mode="inference"
            )
            graph = "{net_name:}_inference.meta".format(net_name=net_name)
        if iteration is not None:
            pipeline += Predict(
                "{net_name:}_training_checkpoint_{it:}".format(
                    net_name=net_name, it=iteration
                ),
                inputs=inputs,
                outputs=outputs,
                array_specs=array_specs_pred,
                graph=graph,
            )
        else:
            pipeline += Run(
                graph,
                inputs=inputs,
                outputs=outputs,
                array_specs=array_specs_pred,
            )
        ak_out = "RAW_PREDICTED"
        ak_ref = "RAW_TARGET"

    else:
        assert (
            len(augmentations) > 0
        ), "Evaluation without denoising model or augmentation is nonsensical"
        del request[ArrayKey("RAW_PREDICTED")]
        ak_out = "RAW_INPUT"
        ak_ref = "RAW_TARGET"
    results = []
    with build(pipeline) as pp:
        for i in range(eval_iterations):
            start_it = time.time()
            b = pp.request_batch(request)
            if net_name is not None and metric_in_graph(metric, net_name):
                res = float(np.squeeze(b.arrays[ak_metric].data))
            else:

                res = compute_metric(
                    b.arrays[ArrayKey(ak_out)].data,
                    b.arrays[ArrayKey(ak_ref)].data,
                    metric,
                )
            results.append(res)
            time_it = time.time() - start_it
            logging.info("it{0:}: {1:}".format(i + 1, time_it))
    return results


def evaluate_blur(
    iterations,
    net_name,
    data_path,
    raw_dataset,
    input_shape,
    output_shape,
    loss_name,
    augmentations,
    cache_size=5,
    num_workers=10,
    intensity_scale_range=(0.75, 1.25),
    intensity_shift_range=(-0.2, 0.2),
    gamma_range=(0.75, 4 / 3.0),
    impulse_noise_prob=0.05,
    prob_missing=0.05,
    prob_low_contrast=0.05,
    contrast_scale=0.1,
    voxel_size=Coordinate((8, 8, 8)),
):
    pipeline, request, array_specs_pred = batch_generator(
        data_path,
        raw_dataset,
        input_shape,
        output_shape,
        augmentations,
        intensity_scale_range=intensity_scale_range,
        intensity_shift_range=intensity_shift_range,
        gamma_range=gamma_range,
        impulse_noise_prob=impulse_noise_prob,
        prob_missing=prob_missing,
        prob_low_contrast=prob_low_contrast,
        contrast_scale=contrast_scale,
        voxel_size=voxel_size,
    )
    net_io_names, _, inputs, outputs = network_setup(net_name, load_checkpoint=False)
    ak_loss = ArrayKey(loss_name.upper())
    outputs[net_io_names["loss_bc"]] = ak_loss

    request.add(ak_loss, voxel_size, voxel_size=voxel_size)
    pipeline = (
        pipeline
        + PreCache(cache_size=cache_size, num_workers=num_workers)
        + Run(
            net_name + "_training.meta",
            inputs=inputs,
            outputs=outputs,
            array_specs=array_specs_pred,
            max_shared_memory=int(1024 * 1024 * 1024),
        )
        + PrintProfilingStats(every=100)
    )
    logging.info("Starting evaluation...")
    costs = []
    with build(pipeline) as pp:
        for i in range(iterations):
            start_it = time.time()
            b = pp.request_batch(request)
            costs.append(float(np.squeeze(b.arrays[ak_loss].data)))
            time_it = time.time() - start_it
            logging.info("it{0:}: {1:}".format(i + 1, time_it))
    logging.info("Evaluation finished")
    return costs
