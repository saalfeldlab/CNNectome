import tensorflow as tf
import logging
import numpy as np
import json
from CNNectome.networks.isotropic.mk_scale_net_cell_generic import make_net
from CNNectome.utils.label import *


class Test:
    def __init__(
        self,
        meta_graph_filename,
        requested_outputs,
        optimizer=None,
        loss=None,
        mode="training",
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
        self.mode = mode
        if self.mode == "training":
            if isinstance(optimizer, "".__class__):
                self.optimizer_loss_names = (optimizer, loss)
            else:
                self.optimizer_func = optimizer

    def read_meta_graph(self):
        logging.info("Reading meta-graph...")
        tf.train.import_meta_graph(
            self.meta_graph_filename + "_" + self.mode + ".meta", clear_devices=True
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
        if self.optimizer_func is None and self.mode == "training":
            self.optimizer = self.graph.get_operation_by_name(
                self.optimizer_loss_names[0]
            )
            self.loss = self.graph.get_tensor_by_name(self.optimizer_loss_names[1])

    def train_step(self, inputs, iteration=None):
        if self.mode == "training":
            to_compute = {
                "optimizer": self.optimizer,
                "loss": self.loss,
                "iteration": self.iteration_increment,
            }
        else:
            to_compute = {}  #'iteration': self.iteration_increment}
        to_compute.update(self.requested_outputs)
        outputs = self.session.run(
            to_compute,
            feed_dict=inputs,
            options=tf.RunOptions(report_tensor_allocations_upon_oom=True),
        )
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
