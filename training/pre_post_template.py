import logging
from networks.mknet_cleftprepost import make_net
from networks import unet_class
from training.anisotropic.train_dist_cleftprepost import train_until
import json
import numpy as np

logging.basicConfig(level=logging.INFO)
dt_scaling_factor = 50
max_iteration = 500000
loss_name = "loss_total"
samples = ["A", "B", "C"]
cremi_dir = "/groups/saalfeld/saalfeldlab/projects/fafb-synapses/cremi2019/"
n5_filename_format = "sample_{0:}.n5"
csv_filename_format = "sample_{0:}_clefts_to_seg.csv"
steps_inference = 20
steps_train = 8
aug_mode = "deluxe"
filter_comments = []


def build_net(steps=steps_inference, mode="inference"):
    unet = unet_class.UNet(
        [12, 12 * 6, 12 * 6 ** 2, 12 * 6 ** 3],
        [12, 12 * 6, 12 * 6 ** 2, 12 * 6 ** 3],
        [(1, 3, 3), (1, 3, 3), (3, 3, 3)],
        [
            [(1, 3, 3), (1, 3, 3), (1, 3, 3)],
            [(1, 3, 3), (1, 3, 3), (3, 3, 3)],
            [(1, 3, 3), (3, 3, 3), (3, 3, 3)],
            [(3, 3, 3), (3, 3, 3), (3, 3, 3)],
        ],
        [
            [(1, 3, 3), (1, 3, 3), (1, 3, 3)],
            [(1, 3, 3), (1, 3, 3), (3, 3, 3)],
            [(1, 3, 3), (3, 3, 3), (3, 3, 3)],
        ],
        constant_upsample=True,
        trans_equivariant=False,
        input_voxel_size=(40, 4, 4),
        input_fov=(40, 4, 4),
    )
    net, input_shape, output_shape = make_net(unet, steps, mode=mode)
    logging.info(
        "Built {0:} with input shape {1:} and output_shape {2:}".format(
            net, input_shape, output_shape
        )
    )
    return net, input_shape, output_shape


def test_memory_consumption(steps=steps_train, mode="train"):
    from utils.test_memory_consumption import Test

    net, input_shape, output_shape = build_net(steps, mode=mode)
    with open("net_io_names.json", "r") as f:
        net_io_names = json.load(f)
    input_arrays = dict()
    requested_outputs = dict()
    input_arrays[net_io_names["raw"]] = np.random.random(
        input_shape.astype(np.int)
    ).astype(np.float32)
    for l in ["cleft", "pre", "post"]:
        if mode.lower() == "train" or mode.lower() == "training":
            input_arrays[net_io_names["gt_" + l + "_dist"]] = np.random.random(
                output_shape
            ).astype(np.float32)

            input_arrays[net_io_names["loss_weights_" + l]] = np.random.random(
                output_shape
            ).astype(np.float32)
            input_arrays[net_io_names[l + "_mask"]] = np.random.random(
                output_shape
            ).astype(np.float32)
        requested_outputs[l] = net_io_names[l + "_dist"]

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
        samples,
        aug_mode,
        input_shape,
        output_shape,
        cremi_dir,
        n5_filename_format,
        csv_filename_format,
        filter_comments,
        dt_scaling_factor,
        loss_name,
        net_name,
    )


if __name__ == "__main__":
    train()
