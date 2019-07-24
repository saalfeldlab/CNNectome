from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from gunpowder.ext import tensorflow as tf
import json
import os


from tensorflow.python.framework import ops
from tensorflow.python.ops import variables
from tensorflow.python.saved_model import tag_constants


def simple_save(graph, export_dir, inputs, outputs, legacy_init_op=None):
    # signature_def_map = {
    #     signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY:
    #         signature_def_utils.predict_signature_def(inputs, outputs)
    # }
    with tf.Session(graph=graph) as session:
        saver = tf.train.import_meta_graph(
            self.graph_name + ".meta", clear_devices=True
        )
        saver.restore(self.session, self.graph_name)
        b = tf.saved_model.builder.SavedModelBuilder(export_dir)
        pred_sig = tf.saved_model.signature_def_utils.build_signature_def(
            inputs=inputs,
            outputs=outputs,
            method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME,
        )
        legacy_init_op = tf.group(tf.tables_initializer(), name="legacy_init_op")
        b.add_meta_graph_and_variables(
            session,
            tags=[tag_constants.SERVING],
            signature_def_map={
                tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY: pred_sig
            },
            assets_collection=ops.get_collection(ops.GraphKeys.ASSET_FILEPATHS),
            legacy_init_op=legacy_init_op,
        )
        b.save()


class Converter(object):
    def __init__(self, graph_name):
        self.graph_name = graph_name
        self.graph = tf.Graph()
        self.session = tf.Session(graph=self.graph)
        with self.graph.as_default():
            self.read_meta_graph()

    def read_meta_graph(self):

        with open(
            os.path.join(os.path.dirname(self.graph_name), "net_io_names.json"), "r"
        ) as f:
            self.net_io_names = json.load(f)

        print(self.net_io_names)
        # print(self.graph.as_graph_element(self.net_io_names['raw']))

    def convert(self, target):
        builder = simple_save(
            self.graph_name, export_dir=target, inputs=inputs, outputs=outputs
        )
        tensor_info_raw = tf.saved_model.utils.build_tensor_info(
            self.graph.as_graph_element(str(self.net_io_names["raw"]))
        )
        tensor_info_dist = tf.saved_model.utils.build_tensor_info(
            self.graph.as_graph_element(str(self.net_io_names["dist"]))
        )
        inputs = {"raw": tensor_info_raw}
        outputs = {"dist": tensor_info_dist}


def store(graph_name, target):
    saver = tf.train.import_meta_graph(graph_name + ".meta", clear_devices=True)
    saver._var_list = variables._all_saveable_objects()
    # print(variables._all_saveable_objects())
    with open(os.path.join(os.path.dirname(graph_name), "net_io_names.json"), "r") as f:
        net_io_names = json.load(f)
    with tf.Session() as sess:
        saver.restore(sess, graph_name)
        # print(variables._all_saveable_objects())
        # print(saver._var_list)
        builder = tf.saved_model.builder.SavedModelBuilder(target)
        input = sess.graph.as_graph_element(net_io_names["raw"])
        output = sess.graph.as_graph_element(net_io_names["dist"])

        input_info = tf.saved_model.utils.build_tensor_info(input)
        output_info = tf.saved_model.utils.build_tensor_info(output)

        signature = tf.saved_model.signature_def_utils.build_signature_def(
            inputs={"raw": input_info},
            outputs={"dist": output_info},
            method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME,
        )
        builder.add_meta_graph_and_variables(
            sess,
            [tf.saved_model.tag_constants.SERVING],
            signature_def_map={"predict_images": signature},
        )
        builder.save(as_text=True)


def load():
    sess = tf.Session()
    signature_key = tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY
    input_key = "raw"
    output_key = "dist"
    export_path = "fib25_unet"
    meta_graph_def = tf.saved_model.loader.load(
        sess, [tf.saved_model.tag_constants.SERVING], export_path
    )
    signature = meta_graph_def.signature_def
    raw_tensor_name = signature[signature_key].inputs[input_key].name
    print(raw_tensor_name)
    dist_tensor_name = signature[signature_key].inputs[output_key].name
    print(dist_tensor_name)

    raw = sess.graph.get_tensor_by_name(raw_tensor_name)
    dist = sess.graph.get_tensor_by_name(dist_tensor_name)


if __name__ == "__main__":
    graph_name = "/nrs/saalfeld/heinrichl/segmentation/distance_jansnet_100/unet_checkpoint_294000"
    target = "fib25_unet"
    store(graph_name, target)
    # converter = Converter(graph_name)
    # converter.convert('fib25_unet')
    # load()
