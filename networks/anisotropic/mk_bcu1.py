from networks import unet, ops3d
import tensorflow as tf
import json


def train_net():
    input_shape = (84, 268, 268)
    raw = tf.placeholder(tf.float32, shape=input_shape)
    raw_bc = tf.reshape(raw, (1, 1) + input_shape)

    last_fmap, fov, anisotropy = unet.unet(
        raw_bc,
        12,
        6,
        [[1, 3, 3], [1, 3, 3], [1, 3, 3]],
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
        voxel_size=(10, 1, 1),
        fov=(10, 1, 1),
    )

    logits_bc, fov = ops3d.conv_pass(
        last_fmap,
        kernel_size=[[1, 1, 1]],
        num_fmaps=2,
        activation=None,
        fov=fov,
        voxel_size=anisotropy,
    )

    output_shape_bc = logits_bc.get_shape().as_list()

    output_shape_c = output_shape_bc[1:]  # strip the batch dimension
    output_shape = output_shape_c[1:]
    flat_logits = tf.transpose(tf.reshape(tensor=logits_bc, shape=(2, -1)))

    gt_labels = tf.placeholder(tf.float32, shape=output_shape)
    gt_labels_flat = tf.reshape(gt_labels, (-1,))

    gt_bg = tf.to_float(tf.not_equal(gt_labels_flat, 1))
    flat_ohe = tf.stack(values=[gt_labels_flat, gt_bg], axis=1)

    loss_weights = tf.placeholder(tf.float32, shape=output_shape)
    mask = tf.placeholder(tf.float32, shape=output_shape)
    loss_weights_flat = tf.reshape(loss_weights, (-1,))
    mask_flat = tf.reshape(mask, (-1,))

    probabilities = tf.reshape(tf.nn.softmax(logits_bc, dim=1)[0], output_shape_c)
    predictions = tf.argmax(probabilities, axis=0)

    ce_loss_balanced = tf.losses.softmax_cross_entropy(
        flat_ohe, flat_logits, weights=loss_weights_flat
    )
    ce_loss_unbalanced = tf.losses.softmax_cross_entropy(
        flat_ohe, flat_logits, weights=mask_flat
    )
    tf.summary.scalar("loss_balanced_syn", ce_loss_balanced)
    tf.summary.scalar("loss_unbalanced_syn", ce_loss_unbalanced)

    opt = tf.train.AdamOptimizer(
        learning_rate=0.5e-4, beta1=0.95, beta2=0.999, epsilon=1e-8
    )

    optimizer = opt.minimize(ce_loss_balanced)
    merged = tf.summary.merge_all()

    tf.train.export_meta_graph(filename="unet.meta")

    names = {
        "raw": raw.name,
        "probabilities": probabilities.name,
        "predictions": predictions.name,
        "gt_labels": gt_labels.name,
        "loss_balanced_syn": ce_loss_balanced.name,
        "loss_unbalanced_syn": ce_loss_unbalanced.name,
        "loss_weights": loss_weights.name,
        "mask": mask.name,
        "optimizer": optimizer.name,
        "summary": merged.name,
    }

    with open("net_io_names.json", "w") as f:
        json.dump(names, f)


def inference_net():
    input_shape = (88, 808, 808)
    raw = tf.placeholder(tf.float32, shape=input_shape)
    raw_bc = tf.reshape(raw, (1, 1) + input_shape)

    last_fmap, fov, anisotropy = unet.unet(
        raw_bc,
        12,
        6,
        [[1, 3, 3], [1, 3, 3], [1, 3, 3]],
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
        voxel_size=(10, 1, 1),
        fov=(10, 1, 1),
    )

    logits_bc, fov = ops3d.conv_pass(
        last_fmap,
        kernel_size=[[1, 1, 1]],
        num_fmaps=2,
        activation=None,
        fov=fov,
        voxel_size=anisotropy,
    )

    output_shape_bc = logits_bc.get_shape().as_list()

    output_shape_c = output_shape_bc[1:]  # strip the batch dimension
    output_shape = output_shape_c[1:]  # strip the channel dimension

    probabilities = tf.reshape(tf.nn.softmax(logits_bc, dim=1)[0], output_shape_c)
    predictions = tf.argmax(probabilities, axis=0)

    tf.train.export_meta_graph(filename="unet_inference.meta")


if __name__ == "__main__":
    train_net()
    tf.reset_default_graph()
    inference_net()
