import sys
sys.path.append("/groups/saalfeld/home/heinrichl/dev/simpleference")
from simpleference.inference.inference import run_inference_n5_multi_crop
from simpleference.inference.util import *
from simpleference.backends.gunpowder.tensorflow.backend import TensorflowPredict
from simpleference.backends.gunpowder.preprocess import normalize, clip, scale_shift
from simpleference.postprocessing import clip_float_to_uint8
import argparse
import os
import json
import time
import z5py
import functools
from gunpowder import Coordinate


def get_output_paths(raw_data_path, setup_path):
    basename, n5_filename = os.path.split(raw_data_path)
    assert n5_filename.endswith('.n5')

    # output directory, e.g. "(...)/setup01/HeLa_Cell2_4x4x4nm/"
    all_data_dir, cell_identifier = os.path.split(basename)
    output_dir = os.path.join(setup_path, cell_identifier)
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    # output file, e.g. "(...)/setup01/HeLa_Cell2_4x4x4nm/HeLa_Cell2_4x4x4nm_it10000.n5"
    base_n5_filename, n5 = os.path.splitext(n5_filename)
    output_filename = base_n5_filename + '_it{0:}'.format(iteration) + n5
    out_file = os.path.join(output_dir, output_filename)
    if not os.path.exists(out_file):
        os.mkdir(out_file)
    return output_dir, out_file


def get_contrast_adjustment(rf, raw_ds, factor, min_sc, max_sc):
    if factor is None:
        if rf[raw_ds].dtype == np.uint8:
            factor = 255.
        elif rf[raw_ds].dtype == np.uint16:
            factor = 256. * 256. - 1
        elif rf[raw_ds].dtype == np.float32:
            assert rf[raw_ds].min() >= 0 and rf[raw_ds].max() <= 1,\
            "Raw values are float but not in [0,1], I don't know how to normalize. Please provide a factor."
            factor = 1.
        else:
            raise ValueError("don't know which factor to assume for data of type {0:}".format(rf[raw_ds].dtype))

    if min_sc is None or max_sc is None:
        try:
            if min_sc is None:
                min_sc = rf[raw_ds].attrs["contrastAdjustment"]["min"]
            if max_sc is None:
                max_sc = rf[raw_ds].attrs["contrastAdjustment"]["max"]
        except KeyError:
            min_sc = 0.
            max_sc = factor
            logging.warning(
                "min_sc and max_sc not specified and contrastAdjustment not found in attributes of {0:}, will continue "
                "with default contrast (min {1:}, max{2:}".format(
                    os.path.join(rf, raw_ds),min_sc, max_sc
                )
            )

    scale = (factor / (float(max_sc) - float(min_sc))) * 2.
    shift = (- scale * (float(min_sc) / factor)) * 2. - 1

    return factor, scale, shift

def prepare_cell_inference(raw_data_path, iteration, n_jobs, raw_ds, mask_ds, setup_path, factor,
                           min_sc, max_sc, finish_interrupted):
    assert os.path.exists(setup_path), "Path to experiment directory does not exist"
    sys.path.append(setup_path)
    import unet_template

    if raw_data_path.endswith('/'):
        raw_data_path = raw_data_path[:-1]
    assert os.path.exists(raw_data_path), "Path to N5 dataset with raw data and mask does not exist"
    assert os.path.exists(os.path.join(setup_path, "unet_train_checkpoint_{0:}".format(iteration)))
    assert os.path.exists(os.path.join(setup_path, "unet_inference"))
    assert os.path.exists(os.path.join(setup_path, "net_io_names.json"))


    rf = z5py.File(raw_data_path, use_zarr_format=False)
    assert raw_ds in rf, "Raw data not present in N5 dataset"
    assert mask_ds in rf, "Mask data not present in N5 dataset"
    shape_vc = rf[raw_ds].shape


    net_name, input_shape_vc, output_shape_vc = unet_template.build_net(steps=unet_template.steps_inference,
                                                                        mode="inference")
    voxel_size_input = unet_template.voxel_size_input
    voxel_size_output = unet_template.voxel_size

    output_shape_wc = Coordinate(output_shape_vc) * voxel_size_output
    chunk_shape_vc = output_shape_vc
    chunk_shape_wc = Coordinate(output_shape_vc) * voxel_size_output

    full_shape_wc = Coordinate(shape_vc) * voxel_size_input
    full_shape_vc_output = full_shape_wc / voxel_size_output

    output_dir, out_file = get_output_paths(raw_data_path, setup_path)

    # offset file, e.g. "(...)/setup01/HeLa_Cell2_4x4x4nm/offsets_volumes_masks_foreground_shape180x180x180.json"

    offset_filename = "offsets_{0:}_shape{1:}x{2:}x{3:}.json".format(mask_ds.replace("/", "_"), *output_shape_wc)
    offset_file = os.path.join(output_dir, offset_filename)

    # prepare datasets
    factor, scale, shift = get_contrast_adjustment(rf, raw_ds, factor, min_sc, max_sc)

    f = z5py.File(out_file, use_zarr_format=False)
    for label in unet_template.labels:
        f.require_dataset(label.labelname, shape=full_shape_vc_output, compression="gzip", dtype="uint8",
                          chunks=chunk_shape_vc)
        f[label.labelname].attrs["resolution"] = tuple(voxel_size_output)
        f[label.labelname].attrs["offset"] = (0, 0, 0)
        f[label.labelname].attrs["raw_data_path"] = raw_data_path
        f[label.labelname].attrs["raw_ds"] = raw_ds
        f[label.labelname].attrs["iteration"] = iteration
        f[label.labelname].attrs["raw_scale"] = scale
        f[label.labelname].attrs["raw_shift"] = shift
        f[label.labelname].attrs["raw_normalize_factor"] = factor

    if not os.path.exists(offset_file) and not finish_interrupted:
        generate_list_for_mask(offset_file, output_shape_wc, raw_data_path, mask_ds)

    if not finish_interrupted:
        with open(offset_file, 'r') as f:
            offset_list = json.load(f)
        offset_list_from_precomputed(offset_list, list(range(n_jobs)), out_file)
    else:
        redistribute_offset_lists(list(range(n_jobs)), out_file)
    return input_shape_vc, output_shape_vc, chunk_shape_vc


def preprocess(data, scale=2, shift=-1., factor=None):
    return clip(scale_shift(normalize(data, factor=factor), scale, shift))


def single_job_inference(job_no, input_shape_vc, output_shape_vc, chunk_shape_vc, raw_data_path, iteration, raw_ds,
                         setup_path, factor=None, min_sc=None, max_sc=None):
    sys.path.append(setup_path)
    import unet_template

    output_dir, out_file = get_output_paths(raw_data_path, setup_path)

    rf = z5py.File(raw_data_path, use_zarr_format=False)
    shape = rf[raw_ds].shape
    weight_meta_graph = os.path.join(setup_path, "unet_train_checkpoint_{0:}".format(iteration))
    inference_meta_graph = os.path.join(setup_path, "unet_inference")
    net_io_json = os.path.join(setup_path, "net_io_names.json")
    with open(net_io_json, "r") as f:
        net_io_names = json.load(f)

    offset_file = os.path.join(out_file, "list_gpu_{0:}.json".format(job_no))
    with open(offset_file, 'r') as f:
        offset_list = json.load(f)

    input_key = net_io_names["raw"]
    network_output_keys = []
    dataset_target_keys = []

    for label in unet_template.labels:
        network_output_keys.append(net_io_names[label.labelname])
        dataset_target_keys.append(net_io_names[label.labelname])

    voxel_size_input = unet_template.voxel_size_input
    voxel_size_output = unet_template.voxel_size

    input_shape_wc = Coordinate(input_shape_vc) * voxel_size_input
    output_shape_wc = Coordinate(output_shape_vc) * voxel_size_output
    chunk_shape_wc = Coordinate(chunk_shape_vc) * voxel_size_output

    prediction = TensorflowPredict(
        weight_meta_graph,
        inference_meta_graph,
        input_keys=input_key,
        output_keys=network_output_keys
    )

    t_predict = time.time()

    factor, scale, shift = get_contrast_adjustment(rf, raw_ds, factor, min_sc, max_sc)

    run_inference_n5_multi_crop(
        prediction,
        functools.partial(preprocess, factor=1./factor, scale=scale, shift=shift),
        functools.partial(clip_float_to_uint8, float_range=(-1, 1), safe_scale=False),
        raw_data_path,
        out_file,
        offset_list,
        network_input_shapes_wc=[input_shape_wc, ],
        network_output_shape_wc=output_shape_wc,
        chunk_shape_wc=chunk_shape_wc,
        input_keys=[raw_ds, ],
        target_keys=dataset_target_keys,
        input_resolutions=[tuple(voxel_size_input), ],
        target_resolutions=[tuple(voxel_size_output), ] * len(dataset_target_keys),
        log_processed=os.path.join(os.path.dirname(offset_file), "list_gpu_{0:}_{1:}_processed.txt".format(job_no,
                                                                                                           iteration)),
        pad_value=int(round(-factor*(shift/scale)))
    )

    t_predict = time.time() - t_predict


def submit_jobs(n_job, input_shape_vc, output_shape_vc, chunk_shape_vc, raw_data_path, iteration, raw_ds, setup_path,
                factor=None, min_sc=None, max_sc=None):
    subprocess.call(['./submit_inference_job.sh', ])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", type=str, choices=("prepare", "inference"))
    parser.add_argument("raw_data_path", type=str)
    parser.add_argument("iteration", type=int)
    parser.add_argument("n_job", type=int)
    parser.add_argument("--raw_ds", type=str, default="volumes/raw")
    parser.add_argument("--mask_ds", type=str, default="volumes/masks/foreground")
    parser.add_argument("--setup_path", type=str, default='.')
    parser.add_argument("--finish_interrupted", type=bool, default=False)
    parser.add_argument("--factor", type=int, default=None)
    parser.add_argument("--min_sc", type=float, default=None)
    parser.add_argument("--max_sc", type=float, default=None)

    args = parser.parse_args()
    action = args.action
    raw_data_path = args.raw_data_path
    iteration = args.iteration
    n_job = args.n_job
    raw_ds = args.raw_ds
    mask_ds = args.mask_ds
    setup_path = args.setup_path
    factor = args.factor
    min_sc = args.min_sc
    max_sc = args.max_sc
    finish_interrupted = args.finish_interrupted
    if action == "prepare":
        prepare_cell_inference(n_job, raw_data_path, iteration, raw_ds, mask_ds, setup_path, factor,
                               min_sc, max_sc, finish_interrupted)
    elif action == "run":
        input_shape_vc, output_shape_vc, chunk_shape_vc = prepare_cell_inference(n_job, raw_data_path, iteration,
                                                                                 raw_ds, mask_ds, setup_path, factor,
                                                                                 min_sc, max_sc, finish_interrupted)
        submit_jobs(n_job, input_shape_vc, output_shape_vc, chunk_shape_vc, raw_data_path, iteration, raw_ds,
                    setup_path, factor=factor, min_sc=min_sc, max_sc=max_sc)

    elif action == "inference":
        single_job_inference(n_job, input_shape_vc, output_shape_vc, chunk_shape_vc, raw_data_path, iteration, raw_ds,
                             setup_path, factor=factor, min_sc=min_sc, max_sc=max_sc)