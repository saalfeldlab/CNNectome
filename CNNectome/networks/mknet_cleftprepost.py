from CNNectome.networks import ops3d
import numpy as np
import tensorflow.compat.v1 as tf
import json


def make_net(unet, added_steps, loss_name="loss_total", padding="valid", mode="train"):
    # input_shape = (43, 430, 430)
    names = dict()
    if padding == "valid":
        input_size = unet.min_input_shape
    else:
        input_size = np.array((0, 0, 0))
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
        num_fmaps=3,
        activation=None,
        padding=padding,
        fov=fov,
        voxel_size=anisotropy,
    )
    output_shape_bc = dist_bc.get_shape().as_list()
    output_shape_c = output_shape_bc[1:]
    output_shape = output_shape_c[1:]

    dist_c = tf.reshape(dist_bc, shape=output_shape_c)
    names["dist"] = dist_c.name
    cleft_dist, pre_dist, post_dist = tf.unstack(dist_c, 3, axis=0)
    names["cleft_dist"] = cleft_dist.name
    names["pre_dist"] = pre_dist.name
    names["post_dist"] = post_dist.name

    if mode.lower() == "train" or mode.lower() == "training":
        gt_cleft_dist = tf.placeholder(tf.float32, shape=output_shape)
        gt_pre_dist = tf.placeholder(tf.float32, shape=output_shape)
        gt_post_dist = tf.placeholder(tf.float32, shape=output_shape)

        names["gt_cleft_dist"] = gt_cleft_dist.name
        names["gt_pre_dist"] = gt_pre_dist.name
        names["gt_post_dist"] = gt_post_dist.name

        loss_weights_cleft = tf.placeholder(tf.float32, shape=output_shape)
        loss_weights_pre = tf.placeholder(tf.float32, shape=output_shape)
        loss_weights_post = tf.placeholder(tf.float32, shape=output_shape)

        names["loss_weights_cleft"] = loss_weights_cleft.name
        names["loss_weights_pre"] = loss_weights_pre.name
        names["loss_weights_post"] = loss_weights_post.name

        cleft_mask = tf.placeholder(tf.float32, shape=output_shape)
        pre_mask = tf.placeholder(tf.float32, shape=output_shape)
        post_mask = tf.placeholder(tf.float32, shape=output_shape)

        names["cleft_mask"] = cleft_mask.name
        names["pre_mask"] = pre_mask.name
        names["post_mask"] = post_mask.name

        loss_balanced_cleft = tf.losses.mean_squared_error(
            gt_cleft_dist, cleft_dist, loss_weights_cleft * cleft_mask
        )
        loss_balanced_pre = tf.losses.mean_squared_error(
            gt_pre_dist, pre_dist, loss_weights_pre * pre_mask
        )
        loss_balanced_post = tf.losses.mean_squared_error(
            gt_post_dist, post_dist, loss_weights_post * post_mask
        )

        names["loss_balanced_cleft"] = loss_balanced_cleft.name
        names["loss_balanced_pre"] = loss_balanced_pre.name
        names["loss_balanced_post"] = loss_balanced_post.name

        loss_unbalanced_cleft = tf.losses.mean_squared_error(
            gt_cleft_dist, cleft_dist, cleft_mask
        )
        loss_unbalanced_pre = tf.losses.mean_squared_error(
            gt_pre_dist, pre_dist, pre_mask
        )
        loss_unbalanced_post = tf.losses.mean_squared_error(
            gt_post_dist, post_dist, post_mask
        )

        names["loss_unbalanced_cleft"] = loss_unbalanced_cleft.name
        names["loss_unbalanced_pre"] = loss_unbalanced_pre.name
        names["loss_unbalanced_post"] = loss_unbalanced_post.name

        loss_total = loss_balanced_cleft + loss_balanced_pre + loss_balanced_post
        loss_total_unbalanced = (
            loss_unbalanced_cleft + loss_unbalanced_pre + loss_unbalanced_post
        )
        names["loss_total"] = loss_total.name
        names["loss_total_unbalanced"] = loss_total_unbalanced.name

        tf.summary.scalar("loss_balanced_cleft", loss_balanced_cleft)
        tf.summary.scalar("loss_balanced_pre", loss_balanced_pre)
        tf.summary.scalar("loss_balanced_post", loss_balanced_post)

        tf.summary.scalar("loss_unbalanced_cleft", loss_unbalanced_cleft)
        tf.summary.scalar("loss_unbalanced_pre", loss_unbalanced_pre)
        tf.summary.scalar("loss_unbalanced_post", loss_unbalanced_post)
        tf.summary.scalar("loss_total", loss_total)
        tf.summary.scalar("loss_total_unbalanced", loss_total_unbalanced)

        opt = tf.train.AdamOptimizer(
            learning_rate=0.5e-4, beta1=0.95, beta2=0.999, epsilon=1e-8
        )
        if loss_name == "loss_total":
            optimizer = opt.minimize(loss_total)
        elif loss_name == "loss_total_unbalanced":
            optimizer = opt.minimize(loss_total_unbalanced)
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
        or mode.lower() == "pred"
    ):
        pass
    else:
        raise ValueError("unknown mode for netowrk construction: {0:}".format(mode))
    net_name = "unet_" + mode
    tf.train.export_meta_graph(filename=net_name + ".meta")
    return net_name, input_size_actual, output_shape
