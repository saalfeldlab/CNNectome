import logging
from CNNectome.utils.label import *
from CNNectome.networks.isotropic.mk_cell_unet_generic import make_net, make_net_upsample
from CNNectome.networks import unet_class
from CNNectome.training.isotropic.train_cell_generic import train_until
from gunpowder import Coordinate
import json
import numpy as np
import argparse

logging.basicConfig(level=logging.INFO)

# running parameters
max_iteration = 500000
cache_size=5
num_workers=10

# voxel size parameters
voxel_size_labels = Coordinate((2,) * 3)
voxel_size = Coordinate((4,) * 3)
voxel_size_input = Coordinate((4,) * 3)

# network parameters
steps_train = 4
steps_inference = 11
loss_name = "loss_total"
constant_upsample = True
trans_equivariant = True
feature_widths_down = [12, 12 * 6, 12 * 6 ** 2, 12 * 6 ** 3]
feature_widths_up = [12 * 6, 12 * 6, 12 * 6 ** 2, 12 * 6 ** 3]
downsampling_factors = [(2,) * 3, (3,) * 3, (3,) * 3]
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

# additional network parameters for upsampling network
upsample_factor = tuple(voxel_size_input/voxel_size)
final_kernel_size = [(3,) * 3, (3,) * 3]
final_feature_width = 12 * 6

# groundtruth source parameters
gt_version = "v0003"
db_name = 'crops'
completion_min = 5


# groundtruth construction parameters
min_masked_voxels = 17561.
dt_scaling_factor = 50
labels = list()
labels.append(Label("ecs", 1))
labels.append(Label("plasma_membrane", 2))
labels.append(Label("mito", (3, 4, 5)))
labels.append(Label("mito_membrane", 3, scale_loss=False, scale_key=labels[-1].scale_key))
labels.append(Label("mito_DNA", 5, scale_loss=False, scale_key=labels[-2].scale_key))
labels.append(Label("golgi", (6, 7)))
labels.append(Label("golgi_membrane", 6))
labels.append(Label("vesicle", (8, 9)))
labels.append(Label("vesicle_membrane", 8, scale_loss=False, scale_key=labels[-1].scale_key))
labels.append(Label("MVB", (10, 11), ))
labels.append(Label("MVB_membrane", 10, scale_loss=False, scale_key=labels[-1].scale_key))
labels.append(Label("lysosome", (12, 13)))
labels.append(Label("lysosome_membrane", 12, scale_loss=False, scale_key=labels[-1].scale_key))
labels.append(Label("LD", (14, 15)))
labels.append(Label("LD_membrane", 14, scale_loss=False, scale_key=labels[-1].scale_key))
labels.append(Label("er", (16, 17, 18, 19, 20, 21, 22, 23)))
labels.append(Label("er_membrane", (16, 18, 20), scale_loss=False, scale_key=labels[-1].scale_key))
labels.append(Label("ERES", (18, 19)))
labels.append(Label("nucleus", (20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 37)))
labels.append(Label("nucleolus", 29, separate_labelset=True))
labels.append(Label("NE", (20, 21, 22, 23)))
labels.append(Label("NE_membrane", (20, 22, 23), scale_loss=False, scale_key=labels[-1].scale_key))
labels.append(Label("nuclear_pore", (22, 23)))
labels.append(Label("nuclear_pore_out", 22, scale_loss=False, scale_key=labels[-1].scale_key))
labels.append(Label("chromatin", (24, 25, 26, 27)))
labels.append(Label("NHChrom", 25))
labels.append(Label("EChrom", 26))
labels.append(Label("NEChrom", 27))
labels.append(Label("microtubules", (30, 36)))
labels.append(Label("microtubules_out", (30,),
                    scale_loss=False,
                    scale_key=labels[-1].scale_key))
labels.append(Label("centrosome", 31, separate_labelset=True))
labels.append(Label("distal_app", 32))
labels.append(Label("subdistal_app", 33))
labels.append(Label("ribosomes", 34, add_constant=8, separate_labelset=True))


def build_net(steps=steps_inference, mode="inference"):
    unet = unet_class.UNet(
        feature_widths_down,
        feature_widths_up,
        downsampling_factors,
        kernel_sizes_down,
        kernel_sizes_up,
        input_voxel_size=voxel_size,
        input_fov=voxel_size,
    )
    if voxel_size == voxel_size_input:
        net, input_shape, output_shape = make_net(unet, labels, steps, loss_name=loss_name, mode=mode)
    else:
        net, input_shape, output_shape = make_net_upsample(unet, labels, steps, upsample_factor,
                                                           final_kernel_size, final_feature_width,
                                                           loss_name=loss_name, mode=mode)
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
        gt_version,
        labels,
        net_name,
        input_shape,
        output_shape,
        loss_name,
        db_username,
        db_password,
        db_name=db_name,
        completion_min=completion_min,
        dt_scaling_factor=dt_scaling_factor,
        cache_size=cache_size,
        num_workers=num_workers,
        min_masked_voxels=min_masked_voxels,
        voxel_size_labels=voxel_size_labels,
        voxel_size=voxel_size,
        voxel_size_input=voxel_size_input
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Build, train or test memory consumption for a U-Net")
    parser.add_argument("script", type=str, help="Pick script that should be run",
                        choices=["train", "build", "test_mem"], default="train")
    parser.add_argument("--mode", type=str, help="for build and test_mem specify whether to run for inference or "
                                               "training network", choices=["training", "inference"],
                        default="training")
    parser.add_argument("--db_username", type=str, help="username for the database")
    parser.add_argument("--db_password", type=str, help="password for the database")
    args = parser.parse_args()
    mode = args.mode
    db_username = args.db_username
    db_password = args.db_password

    if mode == "inference":
        steps = steps_inference
    elif mode == "train":
        steps = steps_train

    if args.script == "train":
        assert mode != "inference"
        assert db_username is not None and db_password is not None, \
            "db_username and db_password need to be given to run training"
        train()
    elif args.script == "build":
        build_net(steps, mode)
    elif args.script == "test_mem":
        test_memory_consumption(steps, mode)
