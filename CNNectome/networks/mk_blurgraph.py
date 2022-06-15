from CNNectome.networks import ops3d
import tensorflow as tf
import json
import logging
import numpy as np
import warnings


def make_graph(
    input_shape,
    output_shape,
    sigma,
    input_name="raw",
    output_names=None,
    loss_name="L2+L2gauss",
    mode="forward",
):
    def compute_loss(loss_func, loss_name, opt=True):
        loss_gauss = []
        for output_it, tgt_it, out_name in zip(network_outputs, target, output_names):
            names[out_name + "_predicted"] = output_it.name
            names[out_name + "_target"] = tgt_it.name

            l_gauss = loss_func(tgt_it, output_it)
            loss_gauss.append(l_gauss)
            tf.summary.scalar(
                "{0:}_gauss_{1:}".format(loss_choice.lower(), out_name), l_gauss
            )
            names[out_name + "_l1_gauss"] = l_gauss.name

        l_gauss_total = tf.add_n(loss_gauss)
        loss_gp_readout = tf.reshape(l_gauss_total, (1,) * 3)
        if opt:
            names["loss"] = l_gauss_total.name
            tf.summary.scalar("loss", l_gauss_total)
        names[loss_name] = loss_gp_readout.name
        tf.summary.scalar(loss_name, l_gauss_total)
        return l_gauss_total

    net_name = "blur_sigma{0:}".format(float(sigma))
    n_out = 1
    names = dict()
    input_shape_bc = (1, 1) + tuple(input_shape)
    output_shape_bc = (1, 1) + tuple(output_shape)
    output_shape_c = (1,) + tuple(output_shape)
    input = tf.placeholder(tf.float32, shape=tuple(input_shape))
    names[input_name] = input.name
    input_bc = tf.reshape(input, input_shape_bc)
    output_full = ops3d.gaussian_blur(input_bc, sigma)
    output_bc = ops3d.crop_zyx(output_full, output_shape_bc)

    output_c = tf.reshape(output_bc, output_shape_c)
    names["output"] = output_c.name
    network_outputs = [tf.reshape(output_c, output_shape)]
    if output_names is None:
        output_names = ["output_{0:}".format(n) for n in range(len(network_outputs))]
    assert len(output_names) == len(network_outputs)
    if "L2gauss" in loss_name:
        loss_choice = "L2"
    elif "L1gauss" in loss_name:
        loss_choice = "L1"
    elif loss_name == "L2" or loss_name == "L1":
        warnings.warn(
            "No loss function specified for gaussian blurring term, default to the otherwise specified {0:}".format(
                loss_name
            )
        )
        loss_choice = loss_name
    else:
        raise ValueError("Cannot interpret loss_name {0:}".format(loss_name))

    if mode.lower() == "forward":
        target = []
        for tgt in range(n_out):
            target.append(tf.placeholder(tf.float32, shape=output_shape))

        compute_loss(tf.losses.absolute_difference, "L1", opt=loss_choice == "L1")
        compute_loss(tf.losses.mean_squared_error, "L2", opt=loss_choice == "L2")

        merged = tf.summary.merge_all()
        names["summary"] = merged.name

        with open("{0:}_io_names.json".format(net_name), "w") as f:
            json.dump(names, f)
    elif mode.lower() == "inference":
        pass
    else:
        raise ValueError("unknown mode for network construction {0:}".format(mode))

    tf.train.export_meta_graph(filename=net_name + "_" + mode + ".meta")
    return net_name, input_shape, output_shape
