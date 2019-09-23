from CNNectome.networks import scale_net
from CNNectome.networks import ops3d
import tensorflow as tf
import numpy as np
import json


def make_net(labels, added_steps, mode="train", loss_name="loss_total"):
    unet0 = scale_net.SerialUNet(
        [12, 12 * 6, 12 * 6 ** 2],
        [48, 12 * 6, 12 * 6 ** 2],
        [(3, 3, 3), (3, 3, 3)],
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
    # input_voxel_size=(
    # 36,36,36))
    input_size = unet0.min_input_shape
    input_size_actual = input_size + added_steps * unet0.step_valid_shape
    scnet = scale_net.ScaleNet([unet0, unet1], input_size_actual, name="scnet_" + mode)
    inputs = []
    names = dict()
    for k, (inp, vs) in enumerate(zip(scnet.input_shapes, scnet.voxel_sizes)):
        raw = tf.placeholder(tf.float32, shape=inp)
        raw_bc = tf.reshape(raw, (1, 1) + tuple(inp.astype(np.int)))
        inputs.append(raw_bc)
        names["raw_{0:}".format(vs[0])] = raw.name

    last_fmap, fov, anisotropy = scnet.build(inputs)

    dist_bc, fov = ops3d.conv_pass(
        last_fmap,
        kernel_size=[(1, 1, 1)],
        num_fmaps=len(labels),
        activation=None,
        fov=fov,
        voxel_size=anisotropy,
    )
    output_shape_bc = dist_bc.get_shape().as_list()
    output_shape_c = output_shape_bc[1:]
    output_shape = output_shape_c[1:]

    dist_c = tf.reshape(dist_bc, output_shape_c)
    names["dist"] = dist_c.name
    network_outputs = tf.unstack(dist_c, len(labels), axis=0)
    if mode.lower() == "train" or mode.lower() == "training":
        # mask = tf.placeholder(tf.float32, shape=output_shape)
        # names['mask'] = mask.name
        # ribo_mask = tf.placeholder(tf.float32, shape=output_shape)
        # names['ribo_mask'] = ribo_mask.name
        gt = []
        w = []
        cw = []
        masks = []
        for l in labels:
            masks.append(tf.placeholder(tf.float32, shape=output_shape))
            gt.append(tf.placeholder(tf.float32, shape=output_shape))
            w.append(tf.placeholder(tf.float32, shape=output_shape))
            cw.append(l.class_weight)
        lb = []
        lub = []
        for output_it, gt_it, w_it, m_it, l in zip(
            network_outputs, gt, w, masks, labels
        ):
            lb.append(tf.losses.mean_squared_error(gt_it, output_it, w_it * m_it))
            lub.append(tf.losses.mean_squared_error(gt_it, output_it, m_it))
            # if l.labelname != 'ribosomes':
            #    lub.append(tf.losses.mean_squared_error(gt_it, output_it, mask))
            # else:
            #    lub.append(tf.losses.mean_squared_error(gt_it, output_it, ribo_mask))
            names[l.labelname] = output_it.name
            names["gt_" + l.labelname] = gt_it.name
            names["w_" + l.labelname] = w_it.name
            names["mask_" + l.labelname] = m_it.name
        for l, lb_it, lub_it in zip(labels, lb, lub):
            tf.summary.scalar("lb_" + l.labelname, lb_it)
            tf.summary.scalar("lub_" + l.labelname, lub_it)
            names["lb_" + l.labelname] = lb_it.name
            names["lub_" + l.labelname] = lub_it.name

        loss_total = tf.add_n(lb)
        loss_total_unbalanced = tf.add_n(lub)
        loss_total_classweighted = tf.tensordot(lb, cw, axes=1)
        loss_total_unbalanced_classweighted = tf.tensordot(lub, cw, axes=1)

        tf.summary.scalar("loss_total", loss_total)
        names["loss_total"] = loss_total.name
        tf.summary.scalar("loss_total_unbalanced", loss_total_unbalanced)
        names["loss_total_unbalanced"] = loss_total_unbalanced.name
        tf.summary.scalar("loss_total_classweighted", loss_total_classweighted)
        names["loss_total_classweighted"] = loss_total_classweighted.name
        tf.summary.scalar(
            "loss_total_unbalanced_classweighted", loss_total_unbalanced_classweighted
        )
        names[
            "loss_total_unbalanced_classweighted"
        ] = loss_total_unbalanced_classweighted.name

        opt = tf.train.AdamOptimizer(
            learning_rate=0.5e-4, beta1=0.95, beta2=0.999, epsilon=1e-8
        )
        if loss_name == "loss_total":
            optimizer = opt.minimize(loss_total)
        elif loss_name == "loss_total_unbalanced":
            optimizer = opt.minimize(loss_total_unbalanced)
        elif loss_name == "loss_total_unbalanced_classweighted":
            optimizer = opt.minimize(loss_total_unbalanced_classweighted)
        elif loss_name == "loss_total_classweighted":
            optimizer = opt.minimize(loss_total_classweighted)
        else:
            raise ValueError(loss_name + " not defined")
        names["optimizer"] = optimizer.name
        merged = tf.summary.merge_all()
        names["summary"] = merged.name
        with open("net_io_names.json", "w") as f:
            json.dump(names, f)
    elif mode.lower() == "inference" or mode.lower() == "prediction":
        pass
    else:
        raise ValueError("unknown mode for network construction: {0:}".format(mode))
    tf.train.export_meta_graph(filename=scnet.name + ".meta")
    return scnet


def make_any_scale_net(
    serial_unet_list, labels, added_steps, mode="train", loss_name="loss_total"
):
    # input_voxel_size=(
    # 36,36,36))
    input_size = serial_unet_list[0].min_input_shape
    input_size_actual = input_size + added_steps * serial_unet_list[0].step_valid_shape
    scnet = scale_net.ScaleNet(
        serial_unet_list, input_size_actual, name="scnet_" + mode
    )
    inputs = []
    names = dict()
    for k, (inp, vs) in enumerate(zip(scnet.input_shapes, scnet.voxel_sizes)):
        raw = tf.placeholder(tf.float32, shape=inp)
        raw_bc = tf.reshape(raw, (1, 1) + tuple(inp.astype(np.int)))
        inputs.append(raw_bc)
        names["raw_{0:}".format(vs[0])] = raw.name

    last_fmap, fov, anisotropy = scnet.build(inputs)

    dist_bc, fov = ops3d.conv_pass(
        last_fmap,
        kernel_size=[(1, 1, 1)],
        num_fmaps=len(labels),
        activation=None,
        fov=fov,
        voxel_size=anisotropy,
    )
    output_shape_bc = dist_bc.get_shape().as_list()
    output_shape_c = output_shape_bc[1:]
    output_shape = output_shape_c[1:]

    dist_c = tf.reshape(dist_bc, output_shape_c)
    names["dist"] = dist_c.name
    network_outputs = tf.unstack(dist_c, len(labels), axis=0)
    if mode.lower() == "train" or mode.lower() == "training":
        # mask = tf.placeholder(tf.float32, shape=output_shape)
        # names['mask'] = mask.name
        # ribo_mask = tf.placeholder(tf.float32, shape=output_shape)
        # names['ribo_mask'] = ribo_mask.name
        gt = []
        w = []
        cw = []
        masks = []
        for l in labels:
            masks.append(tf.placeholder(tf.float32, shape=output_shape))
            gt.append(tf.placeholder(tf.float32, shape=output_shape))
            w.append(tf.placeholder(tf.float32, shape=output_shape))
            cw.append(l.class_weight)
        lb = []
        lub = []
        for output_it, gt_it, w_it, m_it, l in zip(
            network_outputs, gt, w, masks, labels
        ):
            lb.append(tf.losses.mean_squared_error(gt_it, output_it, w_it * m_it))
            lub.append(tf.losses.mean_squared_error(gt_it, output_it, m_it))
            # if l.labelname != 'ribosomes':
            #    lub.append(tf.losses.mean_squared_error(gt_it, output_it, mask))
            # else:
            #    lub.append(tf.losses.mean_squared_error(gt_it, output_it, ribo_mask))
            names[l.labelname] = output_it.name
            names["gt_" + l.labelname] = gt_it.name
            names["w_" + l.labelname] = w_it.name
            names["mask_" + l.labelname] = m_it.name
        for l, lb_it, lub_it in zip(labels, lb, lub):
            tf.summary.scalar("lb_" + l.labelname, lb_it)
            tf.summary.scalar("lub_" + l.labelname, lub_it)
            names["lb_" + l.labelname] = lb_it.name
            names["lub_" + l.labelname] = lub_it.name

        loss_total = tf.add_n(lb)
        loss_total_unbalanced = tf.add_n(lub)
        loss_total_classweighted = tf.tensordot(lb, cw, axes=1)
        loss_total_unbalanced_classweighted = tf.tensordot(lub, cw, axes=1)

        tf.summary.scalar("loss_total", loss_total)
        names["loss_total"] = loss_total.name
        tf.summary.scalar("loss_total_unbalanced", loss_total_unbalanced)
        names["loss_total_unbalanced"] = loss_total_unbalanced.name
        tf.summary.scalar("loss_total_classweighted", loss_total_classweighted)
        names["loss_total_classweighted"] = loss_total_classweighted.name
        tf.summary.scalar(
            "loss_total_unbalanced_classweighted", loss_total_unbalanced_classweighted
        )
        names[
            "loss_total_unbalanced_classweighted"
        ] = loss_total_unbalanced_classweighted.name

        opt = tf.train.AdamOptimizer(
            learning_rate=0.5e-4, beta1=0.95, beta2=0.999, epsilon=1e-8
        )
        if loss_name == "loss_total":
            optimizer = opt.minimize(loss_total)
        elif loss_name == "loss_total_unbalanced":
            optimizer = opt.minimize(loss_total_unbalanced)
        elif loss_name == "loss_total_unbalanced_classweighted":
            optimizer = opt.minimize(loss_total_unbalanced_classweighted)
        elif loss_name == "loss_total_classweighted":
            optimizer = opt.minimize(loss_total_classweighted)
        else:
            raise ValueError(loss_name + " not defined")
        names["optimizer"] = optimizer.name
        merged = tf.summary.merge_all()
        names["summary"] = merged.name
        with open("net_io_names.json", "w") as f:
            json.dump(names, f)
    elif mode.lower() == "inference" or mode.lower() == "prediction":
        pass
    else:
        raise ValueError("unknown mode for network construction: {0:}".format(mode))
    tf.train.export_meta_graph(filename=scnet.name + ".meta")
    return scnet


def make_mini_net(labels, added_steps, mode="train", loss_name="loss_total"):
    unet0 = scale_net.SerialUNet(
        [12, 12 * 6],
        [48, 12 * 6],
        [(2, 2, 2)],
        [[(3, 3, 3)], [(3, 3, 3)]],
        [[(3, 3, 3)]],
        input_voxel_size=(1, 1, 1),
    )
    unet1 = scale_net.SerialUNet(
        [12, 12 * 6],
        [48, 12 * 6],
        [(2, 2, 2)],
        [[(3, 3, 3)], [(3, 3, 3)]],
        [[(3, 3, 3)]],
        input_voxel_size=(2, 2, 2),
    )

    # input_voxel_size=(
    # 36,36,36))
    input_size = unet0.min_input_shape
    input_size_actual = input_size + added_steps * unet0.step_valid_shape
    scnet = scale_net.ScaleNet([unet0, unet1], input_size_actual, name="scnet_" + mode)
    inputs = []
    names = dict()
    for k, (inp, vs) in enumerate(zip(scnet.input_shapes, scnet.voxel_sizes)):
        raw = tf.placeholder(tf.float32, shape=inp)
        raw_bc = tf.reshape(raw, (1, 1) + tuple(inp.astype(np.int)))
        inputs.append(raw_bc)
        names["raw_{0:}".format(vs[0])] = raw.name

    last_fmap, fov, anisotropy = scnet.build(inputs)

    dist_bc, fov = ops3d.conv_pass(
        last_fmap,
        kernel_size=[(1, 1, 1)],
        num_fmaps=len(labels),
        activation=None,
        fov=fov,
        voxel_size=anisotropy,
    )
    output_shape_bc = dist_bc.get_shape().as_list()
    output_shape_c = output_shape_bc[1:]
    output_shape = output_shape_c[1:]

    dist_c = tf.reshape(dist_bc, output_shape_c)
    names["dist"] = dist_c.name
    network_outputs = tf.unstack(dist_c, len(labels), axis=0)
    if mode.lower() == "train" or mode.lower() == "training":
        # mask = tf.placeholder(tf.float32, shape=output_shape)
        # names['mask'] = mask.name
        # ribo_mask = tf.placeholder(tf.float32, shape=output_shape)
        # names['ribo_mask'] = ribo_mask.name
        gt = []
        w = []
        cw = []
        masks = []
        for l in labels:
            masks.append(tf.placeholder(tf.float32, shape=output_shape))
            gt.append(tf.placeholder(tf.float32, shape=output_shape))
            w.append(tf.placeholder(tf.float32, shape=output_shape))
            cw.append(l.class_weight)
        lb = []
        lub = []
        for output_it, gt_it, w_it, m_it, l in zip(
            network_outputs, gt, w, masks, labels
        ):
            lb.append(tf.losses.mean_squared_error(gt_it, output_it, w_it * m_it))
            lub.append(tf.losses.mean_squared_error(gt_it, output_it, m_it))
            # if l.labelname != 'ribosomes':
            #    lub.append(tf.losses.mean_squared_error(gt_it, output_it, mask))
            # else:
            #    lub.append(tf.losses.mean_squared_error(gt_it, output_it, ribo_mask))
            names[l.labelname] = output_it.name
            names["gt_" + l.labelname] = gt_it.name
            names["w_" + l.labelname] = w_it.name
            names["mask_" + l.labelname] = m_it.name
        for l, lb_it, lub_it in zip(labels, lb, lub):
            tf.summary.scalar("lb_" + l.labelname, lb_it)
            tf.summary.scalar("lub_" + l.labelname, lub_it)
            names["lb_" + l.labelname] = lb_it.name
            names["lub_" + l.labelname] = lub_it.name

        loss_total = tf.add_n(lb)
        loss_total_unbalanced = tf.add_n(lub)
        loss_total_classweighted = tf.tensordot(lb, cw, axes=1)
        loss_total_unbalanced_classweighted = tf.tensordot(lub, cw, axes=1)

        tf.summary.scalar("loss_total", loss_total)
        names["loss_total"] = loss_total.name
        tf.summary.scalar("loss_total_unbalanced", loss_total_unbalanced)
        names["loss_total_unbalanced"] = loss_total_unbalanced.name
        tf.summary.scalar("loss_total_classweighted", loss_total_classweighted)
        names["loss_total_classweighted"] = loss_total_classweighted.name
        tf.summary.scalar(
            "loss_total_unbalanced_classweighted", loss_total_unbalanced_classweighted
        )
        names[
            "loss_total_unbalanced_classweighted"
        ] = loss_total_unbalanced_classweighted.name

        opt = tf.train.AdamOptimizer(
            learning_rate=0.5e-4, beta1=0.95, beta2=0.999, epsilon=1e-8
        )
        if loss_name == "loss_total":
            optimizer = opt.minimize(loss_total)
        elif loss_name == "loss_total_unbalanced":
            optimizer = opt.minimize(loss_total_unbalanced)
        elif loss_name == "loss_total_unbalanced_classweighted":
            optimizer = opt.minimize(loss_total_unbalanced_classweighted)
        elif loss_name == "loss_total_classweighted":
            optimizer = opt.minimize(loss_total_classweighted)
        else:
            raise ValueError(loss_name + " not defined")
        names["optimizer"] = optimizer.name
        merged = tf.summary.merge_all()
        names["summary"] = merged.name
        with open("net_io_names.json", "w") as f:
            json.dump(names, f)
    elif mode.lower() == "inference" or mode.lower() == "prediction":
        pass
    else:
        raise ValueError("unknown mode for network construction: {0:}".format(mode))
    tf.train.export_meta_graph(filename=scnet.name + ".meta")
    return scnet
