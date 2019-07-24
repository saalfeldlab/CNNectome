import tensorflow as tf
import logging
import numpy as np
import json
from training.isotropic.train_cell_scalenet_generic import make_net
from utils.label import *


class Test:
    def __init__(
        self, meta_graph_filename, requested_outputs, optimizer, loss, mode="train"
    ):
        self.meta_graph_filename = meta_graph_filename
        self.graph = None
        self.session = None
        self.requested_outputs = requested_outputs
        self.iteration = None
        self.iteration_increment = None
        self.optimizer = None
        self.optimizer_func = None
        self.optimizer_loss_names = None
        self.loss = None
        if mode.lower() == "training" or mode.lower() == "train":
            self.mode = 1
        elif (
            mode.lower() == "inference"
            or mode.lower() == "prediction"
            or mode.lower == "pred"
        ):
            self.mode = 0
        if self.mode:
            if isinstance(optimizer, ("".__class__, u"".__class__)):
                self.optimizer_loss_names = (optimizer, loss)
            else:
                self.optimizer_func = optimizer

    def read_meta_graph(self):
        logging.info("Reading meta-graph...")
        tf.train.import_meta_graph(
            self.meta_graph_filename + ".meta", clear_devices=True
        )

        with tf.variable_scope("iterator"):
            self.iteration = tf.get_variable(
                "iteration", shape=1, initializer=tf.zeros_initializer, trainable=False
            )
            self.iteration_increment = tf.assign(self.iteration, self.iteration + 1)
        if self.optimizer_func is not None:
            loss, optimizer = self.optimizer_func(self.graph)
            self.loss = loss
            self.optimizer = optimizer

        self.session.run(tf.global_variables_initializer())

    def setup(self):
        self.graph = tf.Graph()
        self.session = tf.Session(graph=self.graph)
        with self.graph.as_default():
            self.read_meta_graph()
        if self.optimizer_func is None and self.mode:
            self.optimizer = self.graph.get_operation_by_name(
                self.optimizer_loss_names[0]
            )
            self.loss = self.graph.get_tensor_by_name(self.optimizer_loss_names[1])

    def train_step(self, inputs, iteration=None):
        if self.mode:
            to_compute = {
                "optimizer": self.optimizer,
                "loss": self.loss,
                "iteration": self.iteration_increment,
            }
        else:
            to_compute = {}  #'iteration': self.iteration_increment}
        to_compute.update(self.requested_outputs)
        outputs = self.session.run(to_compute, feed_dict=inputs)
        if iteration is None:
            logging.info("SUCCESS")
        else:
            logging.info("it {0:} SUCCESS".format(iteration))


def test(labels):
    scnet = make_net(labels, 14, mode="train")
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
        )
    for l in labels:
        input_arrays[net_io_names["gt_" + l.labelname]] = np.random.random(
            scnet.output_shapes[0].astype(np.int)
        )
        if l.scale_loss or l.scale_key is not None:
            input_arrays[net_io_names["w_" + l.labelname]] = np.random.random(
                scnet.output_shapes[0].astype(np.int)
            )
            requested_outputs[l.labelname] = net_io_names[l.labelname]
        input_arrays[net_io_names["mask_" + l.labelname]] = np.random.random(
            scnet.output_shapes[0].astype(np.int)
        )

    t = Test(
        scnet.name,
        requested_outputs,
        net_io_names["optimizer"],
        net_io_names["loss_total"],
    )
    t.setup()
    for it in range(100):
        t.train_step(input_arrays, iteration=it + 1)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    data_dir = "/groups/saalfeld/saalfeldlab/larissa/data/cell/{0:}.n5"
    data_sources = [
        "hela_cell2_crop1_122018",
        "hela_cell2_crop3_122018",
        "hela_cell2_crop6_122018",
        "hela_cell2_crop7_122018",
        "hela_cell2_crop8_122018",
        "hela_cell2_crop9_122018",
        "hela_cell2_crop13_122018",
        "hela_cell2_crop14_122018",
        "hela_cell2_crop15_122018",
        "hela_cell2_crop18_122018",
        "hela_cell2_crop19_122018",
        "hela_cell2_crop20_122018",
        "hela_cell2_crop21_122018",
        "hela_cell2_crop22_122018",
    ]
    labeled_voxels = (
        500 * 500 * 100,
        400 * 400 * 250,
        250 * 250 * 250,
        300 * 300 * 80,
        200 * 200 * 100,
        100 * 100 * 53,
        160 * 160 * 110,
        150 * 150 * 65,
        150 * 150 * 64,
        200 * 200 * 110,
        150 * 150 * 55,
        200 * 200 * 85,
        160 * 160 * 55,
        170 * 170 * 100,
    )
    ribo_sources = [
        "hela_cell2_crop6_122018",
        "hela_cell2_crop7_122018",
        "hela_cell2_crop13_122018",
    ]
    loss_name = "loss_total"
    labels = []
    labels.append(Label("ecs", 1, data_sources=data_sources))
    labels.append(Label("plasma_membrane", 2, data_sources=data_sources))
    labels.append(Label("mito", (3, 4, 5), data_sources=data_sources))
    labels.append(
        Label(
            "mito_membrane",
            3,
            scale_loss=False,
            scale_key=labels[-1].scale_key,
            data_sources=data_sources,
        )
    )
    labels.append(
        Label(
            "mito_DNA",
            5,
            scale_loss=False,
            scale_key=labels[-2].scale_key,
            data_sources=data_sources,
        )
    )
    labels.append(Label("vesicle", (8, 9), data_sources=data_sources))
    labels.append(
        Label(
            "vesicle_membrane",
            8,
            scale_loss=False,
            scale_key=labels[-1].scale_key,
            data_sources=data_sources,
        )
    )
    labels.append(Label("MVB", (10, 11), data_sources=data_sources))
    labels.append(
        Label(
            "MVB_membrane",
            10,
            scale_loss=False,
            scale_key=labels[-1].scale_key,
            data_sources=data_sources,
        )
    )
    labels.append(Label("lysosome", (12, 13), data_sources=data_sources))
    labels.append(
        Label(
            "lysosome_membrane",
            12,
            scale_loss=False,
            scale_key=labels[-1].scale_key,
            data_sources=data_sources,
        )
    )
    labels.append(Label("LD", (14, 15), data_sources=data_sources))
    labels.append(
        Label(
            "LD_membrane",
            14,
            scale_loss=False,
            scale_key=labels[-1].scale_key,
            data_sources=data_sources,
        )
    )
    labels.append(Label("er", (16, 17, 18, 19, 20, 21), data_sources=data_sources))
    labels.append(
        Label(
            "er_membrane",
            (16, 18, 20),
            scale_loss=False,
            scale_key=labels[-1].scale_key,
            data_sources=data_sources,
        )
    )
    labels.append(Label("ERES", (18, 19), data_sources=data_sources))
    labels.append(
        Label(
            "ERES_membrane",
            18,
            scale_loss=False,
            scale_key=labels[-1].scale_key,
            data_sources=data_sources,
        )
    )
    labels.append(Label("nucleus", (20, 21, 24, 25), data_sources=data_sources))
    labels.append(
        Label(
            "NE",
            (20, 21),
            scale_loss=False,
            scale_key=labels[-1].scale_key,
            data_sources=data_sources,
        )
    )
    labels.append(
        Label(
            "NE_membrane",
            20,
            scale_loss=False,
            scale_key=labels[-1].scale_key,
            data_sources=data_sources,
        )
    )
    labels.append(
        Label(
            "chromatin",
            24,
            scale_loss=False,
            scale_key=labels[-1].scale_key,
            data_sources=data_sources,
        )
    )
    # labels.append(Label('nucleoplasm', 25, scale_loss=False, scale_key=labels[-1].scale_key,
    #                    data_sources=data_sources)) #stop using this
    labels.append(Label("microtubules", 26, data_sources=data_sources))
    labels.append(Label("ribosomes", 1, data_sources=ribo_sources))

    test(labels)
