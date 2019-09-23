from CNNectome.networks import unet, custom_ops
from CNNectome.networks import ops3d
import tensorflow as tf
import json


def train_net():

    # z    [1, 1, 1]:  66 ->  38 -> 10
    # y, x [2, 2, 2]: 228 -> 140 -> 52
    shape_0 = (220,) * 3
    shape_1 = (132,) * 3
    shape_2 = (44,) * 3
    ignore = False

    affs_0_bc = tf.ones((1, 3) + shape_0) * 0.5

    with tf.variable_scope("autocontext") as scope:

        # phase 1
        raw_0 = tf.placeholder(tf.float32, shape=shape_0)
        raw_0_bc = tf.reshape(raw_0, (1, 1) + shape_0)

        input_0_bc = tf.concat([raw_0_bc, affs_0_bc], 1)
        if ignore:
            keep_raw_bc = tf.ones_like(raw_0_bc)
            ignore_aff_bc = tf.zeros_like(affs_0_bc)
            ignore_mask_bc = tf.concat([keep_raw_bc, ignore_aff_bc], 1)
            input_0_bc = custom_ops.ignore(input_0_bc, ignore_mask_bc)

        out_bc, fov, anisotropy = unet.unet(
            input_0_bc,
            24,
            3,
            [[2, 2, 2], [2, 2, 2], [2, 2, 2]],
            [
                [(3, 3, 3), (3, 3, 3)],
                [(3, 3, 3), (3, 3, 3)],
                [(3, 3, 3), (3, 3, 3)],
                [(3, 3, 3), (3, 3, 3)],
            ],
            [
                [(3, 3, 3), (3, 3, 3)],
                [(3, 3, 3), (3, 3, 3)],
                [(3, 3, 3), (3, 3, 3)],
                [(3, 3, 3), (3, 3, 3)],
            ],
        )

        affs_1_bc, fov = ops3d.conv_pass(
            out_bc,
            kernel_size=[[1, 1, 1]],
            num_fmaps=3,
            activation="sigmoid",
            fov=fov,
            voxel_size=anisotropy,
        )

        affs_1_c = tf.reshape(affs_1_bc, (3,) + shape_1)
        gt_affs_1_c = tf.placeholder(tf.float32, shape=(3,) + shape_1)
        loss_weights_1_c = tf.placeholder(tf.float32, shape=(3,) + shape_1)

        loss_1 = tf.losses.mean_squared_error(gt_affs_1_c, affs_1_c, loss_weights_1_c)

        # phase 2
        tf.summary.scalar("loss_pred0", loss_1)
        scope.reuse_variables()
        tf.stop_gradient(affs_1_bc)
        raw_1 = ops3d.center_crop(raw_0, shape_1)
        raw_1_bc = tf.reshape(raw_1, (1, 1) + shape_1)

        input_1_bc = tf.concat([raw_1_bc, affs_1_bc], 1)

        out_bc, fov, anisotropy = unet.unet(
            input_1_bc,
            24,
            3,
            [[2, 2, 2], [2, 2, 2], [2, 2, 2]],
            [
                [(3, 3, 3), (3, 3, 3)],
                [(3, 3, 3), (3, 3, 3)],
                [(3, 3, 3), (3, 3, 3)],
                [(3, 3, 3), (3, 3, 3)],
            ],
            [
                [(3, 3, 3), (3, 3, 3)],
                [(3, 3, 3), (3, 3, 3)],
                [(3, 3, 3), (3, 3, 3)],
                [(3, 3, 3), (3, 3, 3)],
            ],
            fov=fov,
            voxel_size=anisotropy,
        )

        affs_2_bc, fov = ops3d.conv_pass(
            out_bc,
            kernel_size=[[1, 1, 1]],
            num_fmaps=3,
            activation="sigmoid",
            fov=fov,
            voxel_size=anisotropy,
        )

        affs_2_c = tf.reshape(affs_2_bc, (3,) + shape_2)
        gt_affs_2_c = ops3d.center_crop(gt_affs_1_c, (3,) + shape_2)
        loss_weights_2_c = ops3d.center_crop(loss_weights_1_c, (3,) + shape_2)

        loss_2 = tf.losses.mean_squared_error(gt_affs_2_c, affs_2_c, loss_weights_2_c)
        tf.summary.scalar("loss_pred1", loss_2)
    loss = loss_1 + loss_2
    tf.summary.scalar("loss_total", loss)
    tf.summary.scalar("loss_diff", loss_1 - loss_2)
    for trainable in tf.trainable_variables():
        custom_ops.tf_var_summary(trainable)
    merged = tf.summary.merge_all()

    opt = tf.train.AdamOptimizer(
        learning_rate=0.5e-4, beta1=0.95, beta2=0.999, epsilon=1e-8
    )
    optimizer = opt.minimize(loss)

    tf.train.export_meta_graph(filename="wnet.meta")

    names = {
        "raw": raw_0.name,
        "affs_1": affs_1_c.name,
        "affs_2": affs_2_c.name,
        "gt_affs": gt_affs_1_c.name,
        "loss_weights": loss_weights_1_c.name,
        "loss": loss.name,
        "optimizer": optimizer.name,
        "summary": merged.name,
    }
    with open("net_io_names.json", "w") as f:
        json.dump(names, f)


if __name__ == "__main__":
    train_net()
