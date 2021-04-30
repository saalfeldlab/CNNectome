import logging
from CNNectome.networks.mk_denoise_unet import make_net
from CNNectome.networks.mk_blurgraph import make_graph
from CNNectome.networks import unet_class
from CNNectome.training.isotropic.train_denoise_flyem import train_until, evaluate_metric
from CNNectome.inference.single_block_inference import single_block_inference
import tensorflow.compat.v1 as tf
from gunpowder import Coordinate
import json
import numpy as np
import argparse
import typing
import functools
import os

logging.basicConfig(level=logging.INFO)

# running parameters
max_iteration = 300000
eval_iterations = 30
cache_size = 5
num_workers = 10

# voxel size parameters
voxel_size = Coordinate((4,) * 3)

# network parameters
network_name = "unet"

loss_name = "L2"
sigma = 0.5

constant_upsample = True
trans_equivariant = True
feature_widths_down = [12, 12 * 3, 12 * 3 ** 2, 12 * 3 ** 3]
feature_widths_up = [12, 12 * 3, 12 * 3 ** 2, 12 * 3 ** 3]
downsampling_factors = [(2,) * 3, (2,) * 3, (2,) * 3]
kernel_sizes_down = [
    [(3,) * 3, (3,) * 3],
    [(3,) * 3, (3,) * 3],
    [(3,) * 3, (3,) * 3],
    [(3,) * 3, (3,) * 3]
]
kernel_sizes_up = [
    [(3,) * 3, (3,) * 3],
    [(3,) * 3, (3,) * 3],
    [(3,) * 3, (3,) * 3]
]
skip_connections = [True, True, True]
enforce_even_context = True

add_context_training = 15 * np.prod(downsampling_factors, axis=0)
padding_train = "valid"
add_context_inference = 35 * np.prod(downsampling_factors, axis=0)
padding_inference = "valid"

n_out = 1
input_name = "raw_input"
output_names = ["raw",]

# additional network parameters for upsampling network
final_kernel_size = [(3,) * 3, (3,) * 3]
final_feature_width = 12 * 6

# groundtruth source parameters
data_path = "/groups/cosem/cosem/data/jrc_mb-1a/jrc_mb-1a.n5"
raw_dataset = "volumes/raw/s0"

# augmentations
augmentations = ["simple", "intensity", "gamma", "impulse_noise"]
exclude_for_baseline = ["simple", "intensity", "gamma", "elastic", "defect"]
intensity_scale_range = (0.75, 1.25)
intensity_shift_range = (-0.2, 0.2)
gamma_range = (0.75, 4/3.)
impulse_noise_prob = 0.1
prob_missing = 0.05
prob_low_contrast = 0.05
contrast_scale = 0.1


def build_net(mode="inference"):
    if mode != "training" and not os.path.exists("{0:}_io_names.json".format(network_name)):
        logging.info("Building mode training first to generate io names")
        build_net(mode="training")
        tf.reset_default_graph()
    if mode == "inference" or mode == "forward":
        padding = padding_inference
        add_context = add_context_inference
    elif mode == "training":
        padding = padding_train
        add_context = add_context_training
    else:
        raise ValueError("Unkown mode: {0:}".format(mode))
    unet = unet_class.UNet(
        feature_widths_down,
        feature_widths_up,
        downsampling_factors,
        kernel_sizes_down,
        kernel_sizes_up,
        skip_connections=skip_connections,
        padding=padding,
        constant_upsample=constant_upsample,
        trans_equivariant=trans_equivariant,
        enforce_even_context=enforce_even_context,
        input_voxel_size=voxel_size,
        input_fov=voxel_size,
    )
    net, input_shape, output_shape = make_net(network_name, unet, n_out, add_context,  input_name=input_name,
                                              output_names=output_names, loss_name=loss_name, mode=mode)

    logging.info(
        "Built {0:} with input shape {1:} and output_shape {2:}".format(
            net, input_shape, output_shape
        )
    )

    return net, input_shape, output_shape


def build_blur_graph(sigma: float = 0.5, mode="forward"):
    _, input_shape, output_shape = build_net(mode="forward")
    tf.reset_default_graph()
    blur_graph, input_shape, output_shape = make_graph(input_shape, output_shape, sigma,
               input_name=input_name, output_names=output_names, loss_name=loss_name, mode=mode)
    logging.info("Built {0:} with sigma {1:}, input_shape{2:} and output_shape{3:}".format(blur_graph, sigma,
                                                                                           input_shape, output_shape))
    return blur_graph, input_shape, output_shape


def test_memory_consumption(mode="training"):
    from CNNectome.utils.test_memory_consumption import Test

    net, input_shape, output_shape = build_net(mode=mode)
    with open("{0:}_io_names.json".format(net), "r") as f:
        net_io_names = json.load(f)
    input_arrays = dict()
    requested_outputs = dict()
    input_arrays[net_io_names[input_name]] = np.random.random(
        input_shape.astype(np.int)
    ).astype(np.float32)
    for n, out_name in zip(range(n_out), output_names):
        if mode.lower() == "training":
            input_arrays[net_io_names[out_name+"_target"]] = np.random.random(output_shape).astype(np.float32)
        requested_outputs[out_name + "_predicted"] = net_io_names[out_name + "_predicted"]

    t = Test(
        net,
        requested_outputs,
        net_io_names["optimizer"],
        net_io_names["loss"],
        mode=mode,
    )
    t.setup()
    for it in range(100):
        t.train_step(input_arrays, iteration=it + 1)


def train():
    net_name, input_shape, output_shape = build_net(mode="training")
    train_until(
        max_iteration,
        net_name,
        data_path,
        raw_dataset,
        input_shape,
        output_shape,
        augmentations,
        cache_size=cache_size,
        num_workers=num_workers,
        intensity_scale_range=intensity_scale_range,
        intensity_shift_range=intensity_shift_range,
        gamma_range=gamma_range,
        impulse_noise_prob=impulse_noise_prob,
        prob_missing=prob_missing,
        prob_low_contrast=prob_low_contrast,
        contrast_scale=contrast_scale,
        voxel_size=voxel_size
    )


def inference(ckpt, input_file, input_ds, coordinate, output_file):
    net_name, input_shape, output_shape = build_net(mode="inference")
    outputs = [out_name + "_predicted" for out_name in output_names]
    single_block_inference(net_name, input_shape, output_shape, ckpt, outputs, input_file,
                           input_ds_name=input_ds, coordinate=coordinate, output_file=output_file,
                           voxel_size_input=voxel_size, voxel_size_output=voxel_size, input="raw_input")


def evaluate(model: typing.Optional[str] = "unet",
             iteration: typing.Optional[int] = None,
             add_input_noise: bool = False,
             metric: str = "structural_similarity",
             sigma: float = 0.5):
    filename = metric
    if model == "unet" or model is None:
        # easiest to just have both types built
        net_name, input_shape, output_shape = build_net("forward")
        tf.reset_default_graph()
        net_name, input_shape, output_shape = build_net("inference")
        if model is None:
            filename += "_None"
            net_name = None
        else:
            filename += "_" + net_name + "_it{0:}".format(iteration)

    elif model == "blur":
        net_name, input_shape, output_shape = build_blur_graph(sigma, mode="forward")
        tf.reset_default_graph()
        net_name, input_shape, output_shape = build_blur_graph(sigma, mode="inference")
        filename += "_" + net_name
    else:
        raise ValueError("Unknown denoising model {0:}".format(model))

    if add_input_noise:
        eval_augmentations = [aug for aug in augmentations if aug not in exclude_for_baseline]
        filename += "_plusnoise"
    else:
        eval_augmentations = []

    results = evaluate_metric(
        eval_iterations,
        metric,
        net_name,
        iteration,
        data_path,
        raw_dataset,
        input_shape,
        output_shape,
        augmentations=eval_augmentations,
        cache_size=cache_size,
        num_workers=num_workers,
        intensity_scale_range=intensity_scale_range,
        intensity_shift_range=intensity_shift_range,
        gamma_range=gamma_range,
        impulse_noise_prob=impulse_noise_prob,
        prob_missing=prob_missing,
        prob_low_contrast=prob_low_contrast,
        contrast_scale=contrast_scale,
        voxel_size=voxel_size
    )
    with open(filename + ".json", "w") as f:
        json.dump(results, f)


def main():
    def type_or_none(arg: str, type: type = str):
        if arg == "None":
            return None
        else:
            return type(arg)
    parser = argparse.ArgumentParser("Build, train, test memory consumption or evaluate a U-Net for denoising")
    subparser = parser.add_subparsers(dest="script", help="Pick a subcommand to run.")

    train_parser = subparser.add_parser("train")

    build_parser = subparser.add_parser("build")
    build_parser.add_argument("--mode", type=str, help="mode for network construction",
                              choices=["training", "inference", "forward"])

    test_mem_parser = subparser.add_parser("test_mem")
    test_mem_parser.add_argument("--mode", type=str, help="mode for network construction",
                                 choices=["training", "inference", "forward"])

    inference_parser = subparser.add_parser("inference")
    inference_parser.add_argument("--ckpt", type=str, help="checkpoint file")
    inference_parser.add_argument("--input_file", type=str, help="n5 file for input data to predict from")
    inference_parser.add_argument("--input_ds", type=str, help="n5 dataset to predict from", default="volumes/raw/s0")
    inference_parser.add_argument("--output_file", type=str, help="n5 file to write inference output to",
                                  default="prediction.n5")
    inference_parser.add_argument("--coordinate", type=int, help="upper left coordinate of input block to predict from",
                                  default=(0, 0, 0), nargs="+")

    evaluation_parser = subparser.add_parser("evaluation", aliases=["eval"])
    evaluation_parser.add_argument("metric", type=str, help="metric to evaluate")
    evaluation_parser.add_argument("--model", type=functools.partial(type_or_none, type=str),
                                   help="denoising model to use", choices=["unet", "blur", None],
                                   default="unet")
    evaluation_parser.add_argument("--iteration", type=functools.partial(type_or_none, type=int),
                                   help="iteration of network to use, set to None if the model is not trainable",
                                   default=None)
    evaluation_parser.add_argument("--sigma", type=float, help="sigma for gaussian when using blur as denoising model",
                                   default=sigma)
    evaluation_parser.add_argument("--add_input_noise", help="compare model output to raw data with artificially added "
                                   "noise as for training", action="store_true")

    args = parser.parse_args()

    if args.script == "train":
        train()
    elif args.script == "inference":
        ckpt = args.ckpt
        input_file = args.input_file
        input_ds = args.input_ds
        output_file = args.output_file
        coordinate = tuple(args.coordinate)
        inference(ckpt, input_file, input_ds, coordinate, output_file)
    elif args.script == "test_mem":
        test_memory_consumption(args.mode)
    elif args.script == "build":
        build_net(args.mode)
    elif args.script == "eval" or args.script == "evaluation":
        if args.iteration is not None:
            assert args.model in ["unet", ], "model {0:} does not have iterations".format(str(args.model))
        if args.model in ["unet", ]:
            assert args.iteration is not None, "need to specify iteration for model {0:}".format(str(args.model))
        evaluate(args.model, args.iteration, add_input_noise=args.add_input_noise, metric=args.metric, sigma=args.sigma)


if __name__ == "__main__":
    main()