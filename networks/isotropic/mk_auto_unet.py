from networks import unet_auto, ops3d
import tensorflow as tf
import json

if __name__ == "__main__":

    raw = tf.placeholder(tf.float32, shape=(132,) * 3)
    raw_bc = tf.reshape(raw, (1, 1) + (132,) * 3)

    pred = tf.placeholder(tf.float32, shape=(132,) * 3)
    pred_bc = tf.reshape(pred, (1, 1) + (132,) * 3)

    last_fmap, fov, anisotropy = unet_auto.unet_auto(
        raw_bc,
        pred_bc,
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
        [
            [(3, 3, 3), (3, 3, 3)],
            [(3, 3, 3), (3, 3, 3)],
            [(3, 3, 3), (3, 3, 3)],
            [(3, 3, 3), (3, 3, 3)],
        ],
    )

    affs_bc, fov = ops3d.conv_pass(
        last_fmap,
        kernel_size=[[1, 1, 1]],
        num_fmaps=3,
        activation="sigmoid",
        fov=fov,
        voxel_size=anisotropy,
    )

    output_shape_bc = affs_bc.get_shape().as_list()
    output_shape_c = output_shape_bc[1:]  # strip the batch dimension

    affs_c = tf.reshape(affs_bc, output_shape_c)

    gt_affs_c = tf.placeholder(tf.float32, shape=output_shape_c)

    loss_weights_c = tf.placeholder(tf.float32, shape=output_shape_c)

    loss = tf.losses.mean_squared_error(gt_affs_c, affs_c, loss_weights_c)

    opt = tf.train.AdamOptimizer(
        learning_rate=0.5e-4, beta1=0.95, beta2=0.999, epsilon=1e-8
    )
    optimizer = opt.minimize(loss)

    tf.train.export_meta_graph(filename="unet_auto.meta")

    names = {
        "raw": raw.name,
        "pred": pred.name,
        "affs": affs_c.name,
        "gt_affs": gt_affs_c.name,
        "loss_weights": loss_weights_c.name,
        "loss": loss.name,
        "optimizer": optimizer.name,
    }
    with open("net_io_names.json", "w") as f:
        json.dump(names, f)
