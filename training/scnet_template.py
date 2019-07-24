import logging
from utils.label import *
from networks.isotropic.mk_scale_net_cell_generic import make_any_scale_net
from networks import scale_net
from training.isotropic.train_cell_scalenet_generic import train_until
import json
import numpy as np

logging.basicConfig(level=logging.INFO)
steps_train = 4
steps_inference = 20
data_dir = "/groups/saalfeld/saalfeldlab/larissa/data/cell/multires/v061719_o750x750x750_m1800x1800x1800_8nm/{0:}.n5"
data_sources = list()
data_sources.append(N5Dataset("crop1", 500 * 500 * 100, data_dir=data_dir))
data_sources.append(N5Dataset("crop3", 400 * 400 * 250, data_dir=data_dir))
data_sources.append(
    N5Dataset(
        "crop4", 300 * 300 * 238, special_categories=("centrosomes",), data_dir=data_dir
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

# input_shape = (196, 196, 196)
# output_shape = (92, 92, 92)
dt_scaling_factor = 50
max_iteration = 500000
loss_name = "loss_total"
raw_name = "raw"
labels = list()
labels.append(Label("ecs", 1, data_sources=data_sources, data_dir=data_dir))
labels.append(Label("plasma_membrane", 2, data_sources=data_sources, data_dir=data_dir))
labels.append(Label("mito", (3, 4, 5), data_sources=data_sources, data_dir=data_dir))
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
labels.append(Label("golgi_membrane", 6, data_sources=data_sources, data_dir=data_dir))
labels.append(Label("vesicle", (8, 9), data_sources=data_sources, data_dir=data_dir))
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
labels.append(Label("lysosome", (12, 13), data_sources=data_sources, data_dir=data_dir))
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
        (20, 21, 22, 23, 24, 25, 26, 27, 28, 29),
        data_sources=data_sources,
        data_dir=data_dir,
    )
)
labels.append(Label("nucleolus", 29, data_sources=nucleolus_sources, data_dir=data_dir))
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
    Label("chromatin", (24, 25, 26, 27), data_sources=data_sources, data_dir=data_dir)
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
labels.append(Label("subdistal_app", 33, data_sources=data_sources, data_dir=data_dir))
labels.append(Label("ribosomes", 1, data_sources=ribo_sources, data_dir=data_dir))

raw_names = ("volumes/raw_2-3-3-3/data/s0", "volumes/raw_2-3-3-3/data/s2")


def build_net(steps=steps_inference, mode="inference"):
    unet0 = scale_net.SerialUNet(
        [12, 12 * 6, 12 * 6 ** 2],
        [12 * 6, 12 * 6, 12 * 6 ** 2],
        [(2, 2, 2), (3, 3, 3)],
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
    net = make_any_scale_net([unet0, unet1], labels, steps, mode=mode)
    logging.info(
        "Built {0:} with input shapes {1:} and output_shapes {2:}".format(
            net.name, net.input_shapes, net.output_shapes
        )
    )

    return net


def test_memory_consumption(steps=steps_train, mode="train"):
    from utils.test_memory_consumption import Test

    scnet = build_net(steps, mode=mode)

    print("INPUT:", scnet.input_shapes)
    print("OUTPUT:", scnet.output_shapes)
    print("BOTTOM:", scnet.bottom_shapes)
    with open("net_io_names.json", "r") as f:
        net_io_names = json.load(f)

    input_arrays = dict()
    requested_outputs = dict()
    for k, (inp, vs) in enumerate(zip(scnet.input_shapes, scnet.voxel_sizes)):
        input_arrays[net_io_names["raw_{0}".format(vs[0])]] = np.random.random(
            inp.astype(np.int)
        ).astype(np.float32)
    for l in labels:
        if mode.lower() == "train" or mode.lower() == "training":
            input_arrays[net_io_names["gt_" + l.labelname]] = np.random.random(
                scnet.output_shapes[0].astype(np.int)
            ).astype(np.float32)
            if l.scale_loss or l.scale_key is not None:
                input_arrays[net_io_names["w_" + l.labelname]] = np.random.random(
                    scnet.output_shapes[0].astype(np.int)
                ).astype(np.float32)
            input_arrays[net_io_names["mask_" + l.labelname]] = np.random.random(
                scnet.output_shapes[0].astype(np.int)
            ).astype(np.float32)
        requested_outputs[l.labelname] = net_io_names[l.labelname]

    t = Test(
        scnet.name,
        requested_outputs,
        net_io_names["optimizer"],
        net_io_names["loss_total"],
        mode=mode,
    )
    t.setup()
    for it in range(100):
        t.train_step(input_arrays, iteration=it + 1)


def train(steps=steps_train):
    scnet = build_net(steps, mode="train")
    train_until(
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
    )


if __name__ == "__main__":
    train()
