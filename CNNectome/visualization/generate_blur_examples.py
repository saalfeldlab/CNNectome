from CNNectome.networks import ops3d
from CNNectome.utils import config_loader
import tensorflow as tf
import json
import logging
import numpy as np
from simpleference.inference.inference import run_inference_zarr_multi_crop
from simpleference.inference.util import *
from simpleference.backends.gunpowder.tensorflow.backend import TensorflowPredict
from simpleference.backends.gunpowder.preprocess import normalize, clip, scale_shift
from simpleference.postprocessing import clip_float_to_uint8
import argparse
import os
import json
import time
import zarr
import functools
import logging
import numcodecs
from gunpowder import Coordinate

size = Coordinate((200, 200, 200))
voxel_size = Coordinate((8, 8, 8))


def blur_tf(input_size, sigma):
    names = dict()
    input = tf.placeholder(tf.float32, shape=tuple(input_size))
    names["raw_input"] = input.name
    input_bc = tf.reshape(input, (1, 1) + tuple(input_size))
    output_bc = ops3d.gaussian_blur(input_bc, sigma)
    output_shape_bc = output_bc.get_shape().as_list()
    output_shape_c = output_shape_bc[1:]
    output_shape = output_shape_c[1:]
    output_c = tf.reshape(output_bc, output_shape_c)
    names["output"] = output_c.name
    with open("net_io_names.json", "w") as f:
        json.dump(names, f)

    tf.train.export_meta_graph(filename="blur_{0:}.meta".format(float(sigma)))
    return names


def get_output_paths(raw_data_path, setup_path, output_path):
    if output_path is None:
        basename, n5_filename = os.path.split(raw_data_path)
        assert n5_filename.endswith(".n5")

        # output directory, e.g. "(...)/setup01/HeLa_Cell2_4x4x4nm/"
        all_data_dir, cell_identifier = os.path.split(basename)
        output_dir = os.path.join(setup_path, cell_identifier)

        # output file, e.g. "(...)/setup01/HeLa_Cell2_4x4x4nm/HeLa_Cell2_4x4x4nm_it10000.n5"
        base_n5_filename, n5 = os.path.splitext(n5_filename)
        output_filename = base_n5_filename + "_sigma{0:}".format(float(sigma)) + n5
        out_file = os.path.join(output_dir, output_filename)
    else:
        assert output_path.endswith(".n5") or output_path.endswith(".n5/")
        output_dir = os.path.abspath(os.path.dirname(output_path))
        out_file = os.path.abspath(output_path)
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    if not os.path.exists(out_file):
        os.mkdir(out_file)
    return output_dir, out_file


def get_contrast_adjustment(rf, raw_ds, factor, min_sc, max_sc):
    if factor is None:
        if rf[raw_ds].dtype == np.uint8:
            factor = 255.0
        elif rf[raw_ds].dtype == np.uint16:
            factor = 256.0 * 256.0 - 1
        elif rf[raw_ds].dtype == np.float32:
            assert (
                rf[raw_ds].min() >= 0 and rf[raw_ds].max() <= 1
            ), "Raw values are float but not in [0,1], I don't know how to normalize. Please provide a factor."
            factor = 1.0
        else:
            raise ValueError(
                "don't know which factor to assume for data of type {0:}".format(
                    rf[raw_ds].dtype
                )
            )

    if min_sc is None or max_sc is None:
        try:
            if min_sc is None:
                min_sc = rf[raw_ds].attrs["contrastAdjustment"]["min"]
            if max_sc is None:
                max_sc = rf[raw_ds].attrs["contrastAdjustment"]["max"]
        except KeyError:
            min_sc = 0.0
            max_sc = factor
            logging.warning(
                "min_sc and max_sc not specified and contrastAdjustment not found in attributes of {0:}, will continue "
                "with default contrast (min {1:}, max{2:}".format(
                    os.path.join(rf, raw_ds), min_sc, max_sc
                )
            )

    scale = (factor / (float(max_sc) - float(min_sc))) * 2.0
    shift = -scale * (float(min_sc) / factor) - 1

    return factor, scale, shift


def prepare_cell_inference(
    n_jobs,
    raw_data_path,
    dataset_id,
    sigma,
    raw_ds,
    setup_path,
    output_path,
    factor,
    min_sc,
    max_sc,
    float_range,
    safe_scale,
    n_cpus,
    finish_interrupted,
):
    # assert os.path.exists(setup_path), "Path to experiment directory does not exist"
    # sys.path.append(setup_path)
    # import unet_template
    if raw_data_path.endswith("/"):
        raw_data_path = raw_data_path[:-1]
    assert os.path.exists(
        raw_data_path
    ), "Path to N5 dataset with raw data and mask does not exist"
    # assert os.path.exists(os.path.join(setup_path, "blur.meta"))
    rf = zarr.open(raw_data_path, mode="r")
    assert raw_ds in rf, "Raw data not present in N5 dataset"
    shape_vc = rf[raw_ds].shape

    output_dir, out_file = get_output_paths(raw_data_path, setup_path, output_path)

    if not finish_interrupted:
        names = blur_tf(size, sigma)
        input_shape_vc = Coordinate(size)

        output_shape_wc = Coordinate(size) * voxel_size
        output_shape_vc = Coordinate(size)
        chunk_shape_vc = Coordinate(size)
        chunk_shape_wc = output_shape_wc

        full_shape_wc = Coordinate(shape_vc) * voxel_size
        full_shape_vc_output = Coordinate(shape_vc)

        # offset file, e.g. "(...)/setup01/HeLa_Cell2_4x4x4nm/offsets_volumes_masks_foreground_shape180x180x180.json"
        offset_filename = "offsets_{0:}_shape{1:}x{2:}x{3:}.json".format(
            mask_ds.replace("/", "_"), *output_shape_wc
        )
        offset_file = os.path.join(output_dir, offset_filename)

        # prepare datasets
        factor, scale, shift = get_contrast_adjustment(
            rf, raw_ds, factor, min_sc, max_sc
        )

        f = zarr.open(out_file)
        dataset_target_keys = ["raw_blurred"]
        for dstk in dataset_target_keys:
            if dstk not in f:
                ds = f.empty(
                    name=dstk,
                    shape=full_shape_vc_output,
                    compressor=numcodecs.GZip(6),
                    dtype="uint8",
                    chunks=chunk_shape_vc,
                )
            else:
                ds = f[dstk]
            ds.attrs["resolution"] = tuple(voxel_size)[::-1]
            ds.attrs["offset"] = (0, 0, 0)
            ds.attrs["raw_data_path"] = raw_data_path
            ds.attrs["raw_ds"] = raw_ds
            ds.attrs["parent_dataset_id"] = dataset_id
            ds.attrs["sigma"] = sigma
            ds.attrs["raw_scale"] = scale
            ds.attrs["raw_shift"] = shift
            ds.attrs["raw_normalize_factor"] = factor
            ds.attrs["float_range"] = float_range
            ds.attrs["safe_scale"] = safe_scale

        if not os.path.exists(offset_file):
            generate_full_list(offset_file, output_shape_wc, raw_data_path, raw_ds)
        shapes_file = os.path.join(
            setup_path, "shapes_steps_{0:}x{1:}x{2:}.json".format(*size)
        )
        if not os.path.exists(shapes_file):
            shapes = {
                "input_shape_vc": tuple(int(isv) for isv in input_shape_vc),
                "output_shape_vc": tuple(int(osv) for osv in output_shape_vc),
                "chunk_shape_vc": tuple(int(csv) for csv in chunk_shape_vc),
            }
            with open(shapes_file, "w") as f:
                json.dump(shapes, f)

    p_proc = re.compile("list_gpu_\d+_\S+_processed.txt")
    print(any([p_proc.match(f) is not None for f in os.listdir(out_file)]))
    if any([p_proc.match(f) is not None for f in os.listdir(out_file)]):
        print("Redistributing offset lists over {0:} jobs".format(n_jobs))
        redistribute_offset_lists(list(range(n_jobs)), out_file)
    else:
        with open(offset_file, "r") as f:
            offset_list = json.load(f)
            offset_list_from_precomputed(offset_list, list(range(n_jobs)), out_file)
    return input_shape_vc, output_shape_vc, chunk_shape_vc


def preprocess(data, scale=2, shift=-1.0, factor=None):
    return clip(scale_shift(normalize(data, factor=factor), scale, shift))


def single_job_inference(
    job_no,
    raw_data_path,
    sigma,
    raw_ds,
    setup_path,
    output_path=None,
    factor=None,
    min_sc=None,
    max_sc=None,
    float_range=(-1, 1),
    safe_scale=False,
    n_cpus=5,
):

    output_dir, out_file = get_output_paths(raw_data_path, setup_path, output_path)
    offset_file = os.path.join(out_file, "list_gpu_{0:}.json".format(job_no))
    if not os.path.exists(offset_file):
        return

    with open(offset_file, "r") as f:
        offset_list = json.load(f)

    rf = zarr.open(raw_data_path, mode="r")
    shape_vc = rf[raw_ds].shape
    weight_meta_graph = os.path.join(setup_path, "blur_{0:}".format(float(sigma)))
    inference_meta_graph = os.path.join(setup_path, "blur_{0:}".format(float(sigma)))

    net_io_json = os.path.join(setup_path, "net_io_names.json")
    with open(net_io_json, "r") as f:
        net_io_names = json.load(f)

    shapes_file = os.path.join(
        setup_path, "shapes_steps_{0:}x{1:}x{2:}.json".format(*size)
    )
    with open(shapes_file, "r") as f:
        shapes = json.load(f)
    input_shape_vc, output_shape_vc, chunk_shape_vc = (
        shapes["input_shape_vc"],
        shapes["output_shape_vc"],
        shapes["chunk_shape_vc"],
    )

    input_key = net_io_names["raw_input"]
    network_output_keys = net_io_names["output"]
    dataset_target_keys = ["raw_blurred"]

    input_shape_wc = Coordinate(input_shape_vc) * voxel_size
    output_shape_wc = Coordinate(output_shape_vc) * voxel_size
    chunk_shape_wc = Coordinate(chunk_shape_vc) * voxel_size

    prediction = TensorflowPredict(
        weight_meta_graph,
        inference_meta_graph,
        input_keys=input_key,
        output_keys=network_output_keys,
        has_trained_variables=False,
    )

    t_predict = time.time()

    factor, scale, shift = get_contrast_adjustment(rf, raw_ds, factor, min_sc, max_sc)

    run_inference_zarr_multi_crop(
        prediction,
        functools.partial(preprocess, factor=1.0 / factor, scale=scale, shift=shift),
        functools.partial(
            clip_float_to_uint8, float_range=float_range, safe_scale=safe_scale
        ),
        raw_data_path,
        out_file,
        offset_list,
        network_input_shapes_wc=[
            input_shape_wc,
        ],
        network_output_shape_wc=output_shape_wc,
        chunk_shape_wc=chunk_shape_wc,
        input_keys=[
            raw_ds,
        ],
        target_keys=dataset_target_keys,
        input_resolutions=[
            tuple(voxel_size),
        ],
        target_resolutions=[
            tuple(voxel_size),
        ]
        * len(dataset_target_keys),
        log_processed=os.path.join(
            os.path.dirname(offset_file),
            "list_gpu_{0:}_{1:}_processed.txt".format(job_no, sigma),
        ),
        pad_value=int(round(-factor * (shift / scale))),
        num_cpus=n_cpus,
    )

    t_predict = time.time() - t_predict


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", type=str, choices=("prepare", "inference"))
    parser.add_argument("n_job", type=int)
    parser.add_argument("n_cpus", type=int)
    parser.add_argument("dataset_id", type=str)
    parser.add_argument("sigma", type=float)
    parser.add_argument("--raw_data_path", type=str, default="None")
    parser.add_argument("--raw_ds", type=str, default="volumes/raw/s0")
    parser.add_argument("--mask_ds", type=str, default="volumes/masks/foreground")
    parser.add_argument("--setup_path", type=str, default=".")
    parser.add_argument("--output_path", type=str, default=None)
    parser.add_argument("--finish_interrupted", type=bool, default=False)
    parser.add_argument("--factor", type=int, default=None)
    parser.add_argument("--min_sc", type=float, default=None)
    parser.add_argument("--max_sc", type=float, default=None)
    parser.add_argument("--float_range", type=int, nargs="+", default=(-1, 1))
    parser.add_argument("--safe_scale", type=bool, default=False)
    args = parser.parse_args()
    print(args)
    action = args.action
    dataset_id = args.dataset_id
    if args.raw_data_path == "None":
        raw_data_path = os.path.join(
            config_loader.get_config()["organelles"]["data_path"],
            dataset_id,
            dataset_id + ".n5",
        )
    else:
        raw_data_path = args.raw_data_path
    assert os.path.exists(raw_data_path), "Path {raw_data:} does not exist".format(
        raw_data=raw_data_path
    )
    output_path = args.output_path
    sigma = args.sigma
    n_job = args.n_job
    n_cpus = args.n_cpus
    raw_ds = args.raw_ds
    if args.mask_ds == "None":
        mask_ds = None
    else:
        mask_ds = args.mask_ds
    setup_path = args.setup_path
    factor = args.factor
    min_sc = args.min_sc
    max_sc = args.max_sc
    float_range = tuple(args.float_range)
    assert len(float_range) == 2
    safe_scale = args.safe_scale
    finish_interrupted = args.finish_interrupted
    if action == "prepare":
        prepare_cell_inference(
            n_job,
            raw_data_path,
            dataset_id,
            sigma,
            raw_ds,
            setup_path,
            output_path,
            factor,
            min_sc,
            max_sc,
            float_range,
            safe_scale,
            n_cpus,
            finish_interrupted,
        )
    # elif action == "run":
    #     input_shape_vc, output_shape_vc, chunk_shape_vc = prepare_cell_inference(n_job, raw_data_path, iteration,
    #                                                                              raw_ds, mask_ds, setup_path, factor,
    #                                                                              min_sc, max_sc, finish_interrupted)
    #     submit_jobs(n_job, input_shape_vc, output_shape_vc, chunk_shape_vc, raw_data_path, iteration, raw_ds,
    #                 setup_path, factor=factor, min_sc=min_sc, max_sc=max_sc)

    elif action == "inference":
        single_job_inference(
            n_job,
            raw_data_path,
            sigma,
            raw_ds,
            setup_path,
            output_path=output_path,
            factor=factor,
            min_sc=min_sc,
            max_sc=max_sc,
            float_range=float_range,
            safe_scale=safe_scale,
            n_cpus=n_cpus,
        )
