from CNNectome.networks import ops3d
import tensorflow as tf
import json
from CNNectome.utils.label import *
import logging
import numpy as np



def make_net(unet, labels, added_steps, loss_name="loss_total", mode="train"):
    names = dict()
    input_size = unet.min_input_shape
    input_size_actual = (input_size + added_steps * unet.step_valid_shape).astype(
        np.int
    )

    raw = tf.placeholder(tf.float32, shape=tuple(input_size_actual))
    names["raw"] = raw.name
    raw_bc = tf.reshape(raw, (1, 1) + tuple(input_size_actual))
    last_fmap, fov, anisotropy = unet.build(raw_bc)
    dist_bc, fov = ops3d.conv_pass(
        last_fmap,
        kernel_size=[[1, 1, 1]],
        num_fmaps=len(labels),
        activation=None,
        padding=unet.padding,
        fov=fov,
        voxel_size=anisotropy,
    )
    output_shape_bc = dist_bc.get_shape().as_list()
    output_shape_c = output_shape_bc[1:]  # strip the batch dimension
    output_shape = output_shape_c[1:]

    dist_c = tf.reshape(dist_bc, output_shape_c)
    names["dist"] = dist_c.name
    network_outputs = tf.unstack(dist_c, len(labels), axis=0)
    if mode.lower() == "train" or mode.lower() == "training":
        mask = tf.placeholder(tf.float32, shape=output_shape)
        names["mask"] = mask.name
        # ribo_mask = tf.placeholder(tf.float32, shape=output_shape)

        gt = []
        w = []
        # cw = []
        masks = []
        for l in labels:
            masks.append(tf.placeholder(tf.float32, shape=output_shape))
            gt.append(tf.placeholder(tf.float32, shape=output_shape))
            w.append(tf.placeholder(tf.float32, shape=output_shape))
            #cw.append(l.class_weight)

        lb = []
        lub = []
        for output_it, gt_it, w_it, m_it, label in zip(
            network_outputs, gt, w, masks, labels
        ):
            lb.append(tf.losses.mean_squared_error(gt_it, output_it, w_it * m_it * mask))
            lub.append(tf.losses.mean_squared_error(gt_it, output_it, m_it * mask))
            names[label.labelname] = output_it.name
            names["gt_" + label.labelname] = gt_it.name
            names["w_" + label.labelname] = w_it.name
            names["mask_" + label.labelname] = m_it.name
        for label, lb_it, lub_it in zip(labels, lb, lub):
            tf.summary.scalar("lb_" + label.labelname, lb_it)
            tf.summary.scalar("lub_" + label.labelname, lub_it)
            names["lb_" + label.labelname] = lb_it.name
            names["lb_" + label.labelname] = lub_it.name

        loss_total = tf.add_n(lb)
        loss_total_unbalanced = tf.add_n(lub)
        # loss_total_classweighted = tf.tensordot(lb, cw, axes=1)
        # loss_total_unbalanced_classweighted = tf.tensordot(lub, cw, axes=1)

        tf.summary.scalar("loss_total", loss_total)
        names["loss_total"] = loss_total.name
        tf.summary.scalar("loss_total_unbalanced", loss_total_unbalanced)
        names["loss_total_unbalanced"] = loss_total_unbalanced.name
        # tf.summary.scalar("loss_total_classweighted", loss_total_classweighted)
        # names["loss_total_classweighted"] = loss_total_classweighted.name
        # tf.summary.scalar(
        #     "loss_total_unbalanced_classweighted", loss_total_unbalanced_classweighted
        # )
        # names[
        #     "loss_total_unbalanced_classweighted"
        # ] = loss_total_unbalanced_classweighted.name
        #
        opt = tf.train.AdamOptimizer(
            learning_rate=0.5e-4, beta1=0.95, beta2=0.999, epsilon=1e-8
        )
        if loss_name == "loss_total":
            optimizer = opt.minimize(loss_total)
        elif loss_name == "loss_total_unbalanced":
            optimizer = opt.minimize(loss_total_unbalanced)
        # elif loss_name == "loss_total_unbalanced_classweighted":
        #     optimizer = opt.minimize(loss_total_unbalanced_classweighted)
        # elif loss_name == "loss_total_classweighted":
        #     optimizer = opt.minimize(loss_total_classweighted)
        else:
            raise ValueError(loss_name + " not defined")
        names["optimizer"] = optimizer.name
        merged = tf.summary.merge_all()
        names["summary"] = merged.name

        with open("net_io_names.json", "w") as f:
            json.dump(names, f)
    elif (
        mode.lower() == "inference"
        or mode.lower() == "prediction"
        or mode.lower == "pred"
    ):
        pass
    else:
        raise ValueError("unknown mode for network construction {0:}".format(mode))
    net_name = "unet_" + mode
    tf.train.export_meta_graph(filename=net_name + ".meta")
    return net_name, input_size_actual, output_shape


def make_net_upsample(unet, labels, added_steps, upsample_factor, final_kernel_size, final_feature_width,
    loss_name="loss_total", mode="train"):
    names = dict()
    input_size = unet.min_input_shape
    input_size_actual = (input_size + added_steps * unet.step_valid_shape).astype(
        np.int
    )

    raw = tf.placeholder(tf.float32, shape=tuple(input_size_actual))
    names["raw"] = raw.name
    raw_bc = tf.reshape(raw, (1, 1) + tuple(input_size_actual))
    last_fmap, fov, anisotropy = unet.build(raw_bc)
    last_fmap_up, anisotropy = ops3d.upsample(
        last_fmap,
        upsample_factor,
        final_feature_width,
        name="up_final",
        fov=fov,
        voxel_size=anisotropy,
        constant_upsample=unet.constant_upsample
    )
    conv_last_fmap_up, fov = ops3d.conv_pass(
        last_fmap_up,
        kernel_size=final_kernel_size,
        num_fmaps=final_feature_width,
        activation="relu",
        padding=unet.padding,
        fov=fov,
        voxel_size=anisotropy,
        name="final_conv",
    )

    dist_bc, fov = ops3d.conv_pass(
        conv_last_fmap_up,
        kernel_size=[[1, 1, 1]],
        num_fmaps=len(labels),
        activation=None,
        padding=unet.padding,
        fov=fov,
        voxel_size=anisotropy,
    )
    output_shape_bc = dist_bc.get_shape().as_list()
    output_shape_c = output_shape_bc[1:]  # strip the batch dimension
    output_shape = output_shape_c[1:]

    dist_c = tf.reshape(dist_bc, output_shape_c)
    names["dist"] = dist_c.name
    network_outputs = tf.unstack(dist_c, len(labels), axis=0)
    if mode.lower() == "train" or mode.lower() == "training":
        mask = tf.placeholder(tf.float32, shape=output_shape)
        names["mask"] = mask.name
        # ribo_mask = tf.placeholder(tf.float32, shape=output_shape)

        gt = []
        w = []
        # cw = []
        masks = []
        for l in labels:
            masks.append(tf.placeholder(tf.float32, shape=output_shape))
            gt.append(tf.placeholder(tf.float32, shape=output_shape))
            w.append(tf.placeholder(tf.float32, shape=output_shape))
            #cw.append(l.class_weight)

        lb = []
        lub = []
        for output_it, gt_it, w_it, m_it, label in zip(
            network_outputs, gt, w, masks, labels
        ):
            lb.append(tf.losses.mean_squared_error(gt_it, output_it, w_it * m_it * mask))
            lub.append(tf.losses.mean_squared_error(gt_it, output_it, m_it * mask))
            names[label.labelname] = output_it.name
            names["gt_" + label.labelname] = gt_it.name
            names["w_" + label.labelname] = w_it.name
            names["mask_" + label.labelname] = m_it.name
        for label, lb_it, lub_it in zip(labels, lb, lub):
            tf.summary.scalar("lb_" + label.labelname, lb_it)
            tf.summary.scalar("lub_" + label.labelname, lub_it)
            names["lb_" + label.labelname] = lb_it.name
            names["lb_" + label.labelname] = lub_it.name

        loss_total = tf.add_n(lb)
        loss_total_unbalanced = tf.add_n(lub)
        # loss_total_classweighted = tf.tensordot(lb, cw, axes=1)
        # loss_total_unbalanced_classweighted = tf.tensordot(lub, cw, axes=1)

        tf.summary.scalar("loss_total", loss_total)
        names["loss_total"] = loss_total.name
        tf.summary.scalar("loss_total_unbalanced", loss_total_unbalanced)
        names["loss_total_unbalanced"] = loss_total_unbalanced.name
        # tf.summary.scalar("loss_total_classweighted", loss_total_classweighted)
        # names["loss_total_classweighted"] = loss_total_classweighted.name
        # tf.summary.scalar(
        #     "loss_total_unbalanced_classweighted", loss_total_unbalanced_classweighted
        # )
        # names[
        #     "loss_total_unbalanced_classweighted"
        # ] = loss_total_unbalanced_classweighted.name
        #
        opt = tf.train.AdamOptimizer(
            learning_rate=0.5e-4, beta1=0.95, beta2=0.999, epsilon=1e-8
        )
        if loss_name == "loss_total":
            optimizer = opt.minimize(loss_total)
        elif loss_name == "loss_total_unbalanced":
            optimizer = opt.minimize(loss_total_unbalanced)
        # elif loss_name == "loss_total_unbalanced_classweighted":
        #     optimizer = opt.minimize(loss_total_unbalanced_classweighted)
        # elif loss_name == "loss_total_classweighted":
        #     optimizer = opt.minimize(loss_total_classweighted)
        else:
            raise ValueError(loss_name + " not defined")
        names["optimizer"] = optimizer.name
        merged = tf.summary.merge_all()
        names["summary"] = merged.name

        with open("net_io_names.json", "w") as f:
            json.dump(names, f)
    elif (
        mode.lower() == "inference"
        or mode.lower() == "prediction"
        or mode.lower == "pred"
    ):
        pass
    else:
        raise ValueError("unknown mode for network construction {0:}".format(mode))
    net_name = "unet_" + mode
    tf.train.export_meta_graph(filename=net_name + ".meta")
    return net_name, input_size_actual, output_shape