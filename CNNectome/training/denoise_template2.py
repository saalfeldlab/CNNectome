import logging
from CNNectome.networks.mk_unet import make_net
from CNNectome.networks.mk_blurgraph import make_graph_trainable, make_graph
from CNNectome.networks import unet_class
from CNNectome.training.isotropic.train_denoise import train_until, evaluate_blur
from CNNectome.validation.single_block_inference import single_block_inference
from gunpowder import Coordinate
import tensorflow.compat.v1 as tf
import json
import numpy as np
import argparse
import itertools

logging.basicConfig(level=logging.INFO)

# running parameters
max_iteration = 500000
baseline_iterations = 50000
cache_size=5
num_workers=10

# voxel size parameters
voxel_size = Coordinate((8,) * 3)

# network parameters
steps_train = 4
steps_inference = 11
loss_name = "L2"
sigma = 0.5
padding = "valid"
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
n_out = 1
input_name = "raw_input"
output_names = ["raw",]

# additional network parameters for upsampling network
final_kernel_size = [(3,) * 3, (3,) * 3]
final_feature_width = 12 * 6

# groundtruth source parameters
data_path="/nrs/cosem/heinrichl/scale-pyramids/"
gt_version = "v0003"
db_name = 'crops'
completion_min = 5


# augmentations
augmentations = ["simple", "elastic", "intensity", "gamma", "poisson", "impulse_noise", "defect"]
intensity_scale_range = (0.75, 1.25)
intensity_shift_range = (-0.2, 0.2)
gamma_range = (0.75, 4/3.)
impulse_noise_prob = 0.05
prob_missing = 0.05
prob_low_contrast = 0.05
contrast_scale = 0.1

# groundtruth construction parameters
min_masked_voxels = 17561.
no_of_variants = 8
target_matches = []

# Noise2Noise
for subsample_variant in itertools.permutations(range(no_of_variants), 2):
    target_matches.append(("volumes/subsampled/raw/{0:}".format(subsample_variant[0]),
                           "volumes/subsampled/raw/{0:}".format(subsample_variant[1])))
# Noise2S1
# for subsample_variant in range(no_of_variants):
#     target_matches.append(("volumes/subsampled/raw/{0:}".format(subsample_variant), "volumes/raw/s1"))


def build_net(steps=steps_inference, mode="inference"):
    unet = unet_class.UNet(
        feature_widths_down,
        feature_widths_up,
        downsampling_factors,
        kernel_sizes_down,
        kernel_sizes_up,
        padding=padding,
        constant_upsample=constant_upsample,
        trans_equivariant=trans_equivariant,
        input_voxel_size=voxel_size,
        input_fov=voxel_size,
    )
    net, input_shape, output_shape = make_net(unet, n_out, steps,  input_name=input_name,
                                              output_names=output_names, loss_name=loss_name, mode=mode)

    logging.info(
        "Built {0:} with input shape {1:} and output_shape {2:}".format(
            net, input_shape, output_shape
        )
    )

    return net, input_shape, output_shape


def build_blur_graph(sigma):
    _, input_shape, output_shape = build_net(steps=steps_train, mode="train")
    blur_graph, input_shape, output_shape = make_graph(input_shape, output_shape, sigma,
               input_name=input_name, output_names=output_names, loss_name=loss_name, mode="train")
    logging.info("Built {0:} with sigma {1:}, input_shape{2:} and output_shape{3:}".format(blur_graph, sigma,
                                                                                           input_shape, output_shape))
    return blur_graph, input_shape, output_shape


def build_blur_graph_trainable(sigma):
    # input_shape = Coordinate((132, 132, 132))
    # output_shape = Coordinate((40, 40, 40))
    _, input_shape, output_shape = build_net(steps=steps_train, mode="train")
    tf.reset_default_graph()
    blur_graph, input_shape, output_shape = make_graph_trainable(input_shape, output_shape, sigma,
                                                       input_name=input_name, output_names=output_names,
                                                       loss_name=loss_name, mode="train")
    logging.info("Built {0:} with sigma {1:}, input_shape{2:} and output_shape{3:}".format(blur_graph, sigma,
                                                                                           input_shape, output_shape))
    return blur_graph, input_shape, output_shape


def baseline_eval(sigma):
    blur_graph, input_shape, output_shape = build_blur_graph(sigma)
    costs = evaluate_blur(
        baseline_iterations,
        gt_version,
        blur_graph,
        input_shape,
        output_shape,
        loss_name,
        target_matches,
        db_username,
        db_password,
        augmentations,
        db_name=db_name,
        completion_min=completion_min,
        data_path=data_path,
        cache_size=cache_size,
        num_workers=num_workers,
        intensity_scale_range=intensity_scale_range,
        intensity_shift_range=intensity_shift_range,
        gamma_range=gamma_range,
        impulse_noise_prob=impulse_noise_prob,
        prob_missing=prob_missing,
        prob_low_contrast=prob_low_contrast,
        contrast_scale=contrast_scale,
        min_masked_voxels=min_masked_voxels,
        voxel_size=voxel_size
    )
    with open("costs_sigma{0:}.json".format(sigma), "w") as f:
        json.dump(costs, f)
    return costs


def baseline_optimize(sigma):
    blur_graph, input_shape, output_shape = build_blur_graph_trainable(sigma)
    train_until(
        baseline_iterations,
        gt_version,
        blur_graph,
        input_shape,
        output_shape,
        loss_name,
        target_matches,
        db_username,
        db_password,
        augmentations,
        db_name=db_name,
        completion_min=completion_min,
        data_path=data_path,
        cache_size=cache_size,
        num_workers=num_workers,
        intensity_scale_range=intensity_scale_range,
        intensity_shift_range=intensity_shift_range,
        gamma_range=gamma_range,
        impulse_noise_prob=impulse_noise_prob,
        prob_missing=prob_missing,
        prob_low_contrast=prob_low_contrast,
        contrast_scale=contrast_scale,
        min_masked_voxels=min_masked_voxels,
        voxel_size=voxel_size
    )


def test_memory_consumption(steps=steps_train, mode="train"):
    from CNNectome.utils.test_memory_consumption import Test

    net, input_shape, output_shape = build_net(steps, mode=mode)
    with open("{0:}_io_names.json".format(net), "r") as f:
        net_io_names = json.load(f)
    input_arrays = dict()
    requested_outputs = dict()
    input_arrays[net_io_names[input_name]] = np.random.random(
        input_shape.astype(np.int)
    ).astype(np.float32)
    for n, out_name in zip(range(n_out), output_names):
        if mode.lower() == "train" or mode.lower() == "training":
            input_arrays[net_io_names[out_name+"_target"]] = np.random.random(output_shape).astype(np.float32)
        requested_outputs[out_name + "_predicted"] = net_io_names[out_name + "_predicted"]

    t = Test(
        net,
        requested_outputs,
        net_io_names["optimizer"],
        net_io_names[loss_name],
        mode=mode,
    )
    t.setup()
    for it in range(100):
        t.train_step(input_arrays, iteration=it + 1)


def train(steps=steps_train):
    net_name, input_shape, output_shape = build_net(steps=steps, mode="train")
    train_until(
        max_iteration,
        gt_version,
        net_name,
        input_shape,
        output_shape,
        loss_name,
        target_matches,
        db_username,
        db_password,
        augmentations,
        db_name=db_name,
        completion_min=completion_min,
        data_path=data_path,
        cache_size=cache_size,
        num_workers=num_workers,
        intensity_scale_range=intensity_scale_range,
        intensity_shift_range=intensity_shift_range,
        gamma_range=gamma_range,
        impulse_noise_prob=impulse_noise_prob,
        prob_missing=prob_missing,
        prob_low_contrast=prob_low_contrast,
        contrast_scale=contrast_scale,
        min_masked_voxels=min_masked_voxels,
        voxel_size=voxel_size
    )


def inference(steps=steps_inference):
    net_name, input_shape, output_shape = build_net(steps=steps, mode="inference")
    outputs = [out_name + "_predicted" for out_name in output_names]
    single_block_inference(net_name, input_shape, output_shape, ckpt, outputs, input_file, coordinate=coordinate,
                           output_file=output_file, voxel_size_input=voxel_size, voxel_size_output=voxel_size)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Build, train or test memory consumption for a U-Net")
    parser.add_argument("script", type=str, help="Pick script that should be run",
                        choices=["train", "build", "test_mem", "inference", "baseline"], default="train")
    parser.add_argument("--mode", type=str, help="for build and test_mem specify whether to run for inference or "
                                               "training network", choices=["training", "inference"],
                        )
    parser.add_argument("--db_username", type=str, help="username for the database")
    parser.add_argument("--db_password", type=str, help="password for the database")
    parser.add_argument("--ckpt", type=str, help="checkpoint file to use for inference")
    parser.add_argument("--input_file", type=str, help="n5 file for input data to predict from")
    parser.add_argument("--output_file", type=str, help="n5 file to write inference output to", default="prediction.n5")
    parser.add_argument("--coordinate", type=int, help="upper left coordinate of block to predict from (input)",
                        default=(0, 0, 0), nargs='+')
    parser.add_argument("--sigma", type=float, help="sigma for gaussian blurring", default=sigma)
    args = parser.parse_args()
    mode = args.mode
    db_username = args.db_username
    db_password = args.db_password
    ckpt = args.ckpt
    input_file = args.input_file
    output_file = args.output_file
    coordinate = tuple(args.coordinate)
    sigma = float(args.sigma)

    if args.script == "train":
        if mode == "inference":
            raise ValueError("script train should not be run with mode inference")
        else:
            mode = "training"
        assert db_username is not None and db_password is not None, \
            "db_username and db_password need to be given to run training"

    elif args.script == "inference":
        if mode == "training":
            raise ValueError("script inference should not be run with mode training")
        else:
            mode = "inference"
        assert ckpt is not None and input_file is not None, \
            "ckpt and input_file need to be given to run inference"

    elif args.script == "baseline" and mode is None:
        mode = "training"

    if mode == "inference":
        steps = steps_inference
    elif mode == "training":
        steps = steps_train
    else:
        raise ValueError("mode needs to be given to run script {0:}".format(args.script))

    if args.script == "train":
        train(steps)
    elif args.script == "build":
        build_net(steps, mode)
    elif args.script == "test_mem":
        test_memory_consumption(steps, mode)
    elif args.script == "inference":
        inference(steps)
    elif args.script == "baseline":
        baseline_optimize(sigma)
