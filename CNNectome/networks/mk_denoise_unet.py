from CNNectome.networks import ops3d
import tensorflow as tf
import json
import logging
import numpy as np


def make_net(
    unet,
    n_out,
    added_context,
    sigma=1.0,
    lamb=1.0,
    input_name="raw",
    output_names=None,
    loss_name="loss_total",
    mode="train",
):
    net_name = "unet_" + mode
    names = dict()
    input_size = unet.min_input_shape
    if unet.padding == "valid":
        assert np.all(np.array(added_context) % np.array(unet.step_valid_shape) == 0), "input shape not suitable for " \
                                                                                       "valid padding"
    else:
        if not np.all(np.array(added_context) > 0):
            logging.warning("Small input shape does not generate any output elements free of influence from padding")

    input_size_actual = (np.array(input_size) + np.array(added_context)).astype(np.int)

    input = tf.placeholder(tf.float32, shape=tuple(input_size_actual))
    names[input_name] = input.name
    input_bc = tf.reshape(input, (1, 1) + tuple(input_size_actual))
    last_fmap, fov, anisotropy = unet.build(input_bc)
    output_bc, fov = ops3d.conv_pass(
        last_fmap,
        kernel_size=[[1, 1, 1]],
        num_fmaps=n_out,
        activation=None,
        padding=unet.padding,
        fov=fov,
        voxel_size=anisotropy,
    )
    output_shape_bc = output_bc.get_shape().as_list()
    output_shape_c = output_shape_bc[1:]  # strip the batch dimension
    output_shape = output_shape_c[1:]

    output_c = tf.reshape(output_bc, output_shape_c)
    names["output"] = output_c.name
    network_outputs = tf.unstack(output_c, n_out, axis=0)

    blurred_full = ops3d.gaussian_blur(input_bc, sigma)
    blurred_bc = ops3d.crop_zyx(blurred_full, output_shape_bc)
    blurred_c = tf.reshape(blurred_bc, output_shape_c)
    blurred = tf.reshape(blurred_c, output_shape)
    names["blurred"] = blurred_c.name

    if output_names is None:
        output_names = ["output_{0:}".format(n) for n in range(n_out)]
    assert len(output_names) == n_out
    if mode.lower() == "train" or mode.lower() == "training":
        target = []
        for tgt in range(n_out):
            target.append(tf.placeholder(tf.float32, shape=output_shape))

        loss_l2 = []
        loss_l1 = []
        loss_l2_gauss = []
        loss_l1_gauss = []

        for output_it, tgt_it, out_name in zip(network_outputs, target, output_names):
            names[out_name + "_predicted"] = output_it.name
            names[out_name + "_target"] = tgt_it.name

            l2 = tf.losses.mean_squared_error(tgt_it, output_it)
            loss_l2.append(l2)
            tf.summary.scalar("l2_" + out_name, l2)
            names[out_name + "_l2"] = l2.name

            l1 = tf.losses.absolute_difference(tgt_it, output_it)
            loss_l1.append(l1)
            tf.summary.scalar("l1_" + out_name, l1)
            names[out_name + "_l1"] = l1.name

            l2_gauss = tf.losses.mean_squared_error(blurred, output_it)
            loss_l2_gauss.append(l2_gauss)
            tf.summary.scalar("l2_gauss_" + out_name, l2_gauss)
            names[out_name + "_l2_gauss"] = l2_gauss.name

            l1_gauss = tf.losses.absolute_difference(blurred, output_it)
            loss_l1_gauss.append(l1_gauss)
            tf.summary.scalar("l1_gauss_" + out_name, l1_gauss)
            names[out_name + "_l1_gauss"] = l1_gauss.name

        l2_total = tf.add_n(loss_l2)
        tf.summary.scalar("l2_total", l2_total)

        l1_total = tf.add_n(loss_l1)
        tf.summary.scalar("l1_total", l1_total)

        l2_gauss_total = tf.add_n(loss_l2_gauss)
        tf.summary.scalar("l2_gauss_total", l2_gauss_total)

        l1_gauss_total = tf.add_n(loss_l1_gauss)
        tf.summary.scalar("l1_gauss_total", l1_gauss_total)

        if loss_name == "L2":
            loss_opt = l2_total
        elif loss_name == "L1":
            loss_opt = l1_total
        elif loss_name == "L2+L2gauss":
            loss_opt = l2_total + lamb * l2_gauss_total
        elif loss_name == "L2+L1gauss":
            loss_opt = l2_total + lamb * l1_gauss_total
        elif loss_name == "L1+L2gauss":
            loss_opt = l1_total + lamb * l2_gauss_total
        elif loss_name == "L1+L1gauss":
            loss_opt = l1_total + lamb * l1_gauss_total
        else:
            raise ValueError(loss_name + "not defined")
        names["loss"] = loss_opt.name
        opt = tf.train.AdamOptimizer(
            learning_rate=0.5e-4, beta1=0.95, beta2=0.999, epsilon=1e-8
        )
        optimizer = opt.minimize(loss_opt)
        names["optimizer"] = optimizer.name
        merged = tf.summary.merge_all()
        names["summary"] = merged.name

        with open("{0:}_io_names.json".format(net_name), "w") as f:
            json.dump(names, f)
    elif (
        mode.lower() == "inference"
        or mode.lower() == "prediction"
        or mode.lower() == "pred"
    ):
        pass
    else:
        raise ValueError("unknown mode for network construction {0:}".format(mode))

    tf.train.export_meta_graph(filename=net_name + ".meta")
    return net_name, input_size_actual, output_shape
