import zarr
from numcodecs import GZip
import tensorflow.compat.v1 as tf
import json
import logging
import os
import numpy as np


def single_block_inference(net_name, input_shape, output_shape, ckpt, outputs, input_file, input_ds_name="volumes/raw",
                           coordinate=(0, 0, 0), output_file='prediction.n5', voxel_size_input=(4, 4, 4),
                           voxel_size_output=(4, 4, 4), input="raw", factor=None):

    logging.info("Preparing output file {0:}".format(output_file))
    store = zarr.N5Store(output_file)
    root = zarr.group(store=store)
    compr = GZip(level=6)
    for output_key in outputs:
        root.require_dataset(output_key, shape=output_shape, chunks=output_shape, dtype='float32', compressor=compr)
    root.require_dataset(input, shape=input_shape, chunks=input_shape, dtype='float32', compressor=compr)

    logging.info("Reading input data from {0:}".format(os.path.join(input_file, input_ds_name)))
    sf = zarr.open(input_file, mode="r")
    input_ds = sf[input_ds_name]
    input_data = input_ds[tuple(slice(c, c+w) for c, w in zip(coordinate, input_shape))]

    logging.info("Preprocessing input data")
    if factor is None:
        if input_ds.dtype == np.uint8:
            factor = 255.
        elif input_ds.dtype == np.uint16:
            factor = 256.*256.-1
        else:
            raise ValueError("don't know which factor to assume for data of type {0:}".format(input_ds.dtype))
        logging.debug("Normalization factor set to {0:} based on dtype {1:}".format(factor, input_ds.dtype))

    try:
        contr_adj = input_ds.attrs["contrastAdjustment"]
        scale = float(factor) / (float(contr_adj["max"]) - float(contr_adj["min"]))
        shift = - scale * float(contr_adj["min"])
        input_data = input_data * scale + shift
    except KeyError:
        logging.debug("Attribute `contrastAdjustment` not found in {0:}, keeping contrast as is".format(os.path.join(
            input_file, input_ds_name)))

    logging.debug("Normalizing input data to range -1, 1")
    input_data /= factor
    input_data = input_data * 2 - 1


    # prepare input and output definition for model
    with open('{0:}_io_names.json'.format(net_name))as f:
        net_io_names = json.load(f)
    network_input_key = net_io_names[input]
    network_output_keys = []
    for output_key in outputs:
        network_output_keys.append(net_io_names[output_key])

    logging.info("Running inference using ckpt {0:} with network {1:}".format(ckpt, net_name+'.meta'))
    graph = tf.Graph()
    sess = tf.Session(graph=graph)
    with graph.as_default():
        saver = tf.train.import_meta_graph(net_name + '.meta', clear_devices=True)
        saver.restore(sess, ckpt)
    output_data = sess.run(network_output_keys, feed_dict={network_input_key: input_data})

    logging.info("Writing data to file {0:}".format(output_file))
    # write input data to file
    root[input][...] = input_data
    root[input].attrs["offset"] = (0, 0, 0)
    root[input].attrs["resolution"] = voxel_size_input

    # write output data to file
    print((np.array(input_shape) * np.array(voxel_size_input)) - (np.array(output_shape) * np.array(
        voxel_size_output)))
    offset = tuple(((np.array(input_shape) * np.array(voxel_size_input)) - (np.array(output_shape) * np.array(
        voxel_size_output)) )/ 2.)
    for output_key, data in zip(outputs, output_data):
        root[output_key][...] = data
        root[output_key].attrs["offset"] = offset
        root[output_key].attrs["resolution"] = voxel_size_output



