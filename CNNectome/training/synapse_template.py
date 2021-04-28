import logging
from CNNectome.utils.label import *
from CNNectome.networks.mk_dist_unet_with_labels import make_net
from CNNectome.networks import unet_class
from CNNectome.training.anisotropic.train_dist_syn import train_until
from CNNectome.inference.single_block_inference import single_block_inference
from gunpowder import Coordinate
import json
import numpy as np
import argparse

logging.basicConfig(level=logging.INFO)

# running parameters
max_iteration = 500000
cache_size = 5
num_workers = 10

# voxel size parameters
voxel_size = Coordinate((40,4,4))

# network parameters
steps_train = 4
steps_inference = 11
padding = "valid"
constant_upsample = True
trans_equivariant = True
feature_widths_down = [12, 12 * 6, 12 * 6 ** 2, 12 * 6 ** 3]
feature_widths_up = [12 , 12 * 6, 12 * 6 ** 2, 12 * 6 ** 3]
downsampling_factors = [(1, 3, 3), (1, 3, 3) , (3, 3, 3)]
kernel_sizes_down = [
    [(1, 3, 3), (1, 3, 3)],
    [(1, 3, 3), (1, 3, 3)],
    [(3, 3, 3), (3, 3, 3)],
    [(3, 3, 3), (3, 3, 3)]
]
kernel_sizes_up = [
    [(1, 3, 3), (1, 3, 3)],
    [(1, 3, 3), (1, 3, 3)],
    [(3, 3, 3), (3, 3, 3)]
]

# additional network parameters for upsampling network
#upsample_factor = tuple(voxel_size_input/voxel_size)
#final_kernel_size = [(3,) * 3, (3,) * 3]
#final_feature_width = 12 * 6

# groundtruth source parameters
cremi_dir = "/groups/saalfeld/saalfeldlab/larissa/data/cremi-2019/"
samples = ["A", "B", "C"]
n5_filename_format = "sample_{0:}.n5"
csv_filename_format = "sample_{0:}_clefts_to_seg.csv"
filter_comments_pre = ["ER", "autapse"]
filter_comments_post = ["apposition", "ER", "autapse"]
include_cleft = False

# training parameters
loss_name = "loss_total"
aug_mode = "deluxe"

# groundtruth construction parameters
min_masked_voxels = 17561.
dt_scaling_factor = 50
labels = list()
labels.append(Label("clefts", None, thr=0., scale_loss=True, separate_labelset=True))
labels.append(Label("pre", None, thr=-0.5, scale_loss=True, separate_labelset=True))
labels.append(Label("post", None, thr=-0.5, scale_loss=True, separate_labelset=True))


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
        input_fov=voxel_size
    )
    net, input_shape, output_shape = make_net(unet, labels, steps, loss_name=loss_name, mode=mode)
    logging.info(
        "Built {0:} with input shape {1:} and output_shape {2:}".format(
            net, input_shape, output_shape
        )
    )

    return net, input_shape, output_shape


def test_memory_consumption(steps=steps_train, mode="train"):
    from CNNectome.utils.test_memory_consumption import Test

    net, input_shape, output_shape = build_net(steps, mode=mode)
    with open("net_io_names.json", "r") as f:
        net_io_names = json.load(f)
    input_arrays = dict()
    requested_outputs = dict()
    input_arrays[net_io_names["raw"]] = np.random.random(
        input_shape.astype(np.int)
    ).astype(np.float32)
    for l in labels:
        if mode.lower() == "train" or mode.lower() == "training":
            input_arrays[net_io_names["mask"]] = np.random.randint(0, 1, output_shape).astype(
                np.float32)
            input_arrays[net_io_names["gt_" + l.labelname]] = np.random.random(
                output_shape
            ).astype(np.float32)
            if l.scale_loss or l.scale_key is not None:
                input_arrays[net_io_names["w_" + l.labelname]] = np.random.random(
                    output_shape
                ).astype(np.float32)
            input_arrays[net_io_names["mask_" + l.labelname]] = np.random.random(
                output_shape
            ).astype(np.float32)
        requested_outputs[l.labelname] = net_io_names[l.labelname]

    t = Test(
        net,
        requested_outputs,
        net_io_names["optimizer"],
        net_io_names["loss_total"],
        mode=mode,
    )
    t.setup()
    for it in range(100):
        t.train_step(input_arrays, iteration=it + 1)


def train(steps=steps_train):
    net_name, input_shape, output_shape = build_net(steps=steps, mode="train")
    train_until(
        max_iteration,
        cremi_dir,
        samples,
        n5_filename_format,
        csv_filename_format,
        filter_comments_pre,
        filter_comments_post,
        labels,
        net_name,
        input_shape,
        output_shape,
        loss_name,
        aug_mode,
        dt_scaling_factor=dt_scaling_factor,
        cache_size=cache_size,
        num_workers=num_workers,
        min_masked_voxels=min_masked_voxels,
        voxel_size=voxel_size
    )


def inference(steps=steps_inference):
    net_name, input_shape, output_shape = build_net(steps=steps, mode="inference")
    outputs = [l.labelname for l in labels]
    single_block_inference(net_name, input_shape, output_shape, ckpt, outputs, input_file, coordinate=coordinate,
                           output_file=output_file, voxel_size_input=voxel_size, voxel_size_output=voxel_size)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Build, train or test memory consumption for a U-Net")
    parser.add_argument("script", type=str, help="Pick script that should be run",
                        choices=["train", "build", "test_mem", "inference"], default="train")
    parser.add_argument("--mode", type=str, help="for build and test_mem specify whether to run for inference or "
                                               "training network", choices=["training", "inference"])
    parser.add_argument("--ckpt", type=str, help="checkpoint file to use for inference")
    parser.add_argument("--input_file", type=str, help="n5 file for input data to predict from")
    parser.add_argument("--output_file", type=str, help="n5 file to write inference output to", default="prediction.n5")
    parser.add_argument("--coordinate", type=int, help="upper left coordinate of block to predict from (input)",
                        default=(0, 0, 0), nargs='+')
    args = parser.parse_args()
    mode = args.mode
    ckpt = args.ckpt
    input_file = args.input_file
    output_file = args.output_file
    coordinate = tuple(args.coordinate)

    if args.script == "train":
        if mode == "inference":
            raise ValueError("script train should not be run with mode inference")
        else:
            mode = "training"

    elif args.script == "inference":
        if mode == "training":
            raise ValueError("script inference should not be run with mode training")
        else:
            mode = "inference"
        assert ckpt is not None and input_file is not None, \
            "ckpt and input_file need to be given to run inference"

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
