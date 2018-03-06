import networks
import tensorflow as tf
import json

def center_crop(tensor, size):

    shape = tensor.get_shape().as_list()
    diff = tuple(sh - si for sh, si in zip(shape, size))

    for d in diff:
        assert d >= 0
        assert d%2 == 0

    slices = tuple(slice(d/2, -d/2) if d > 0 else slice(None) for d in diff)

    print("Cropping from %s to %s"%(shape, size))
    print("Diff: %s"%(diff,))
    print("Slices: %s"%(slices,))

    cropped = tensor[slices]

    print("Result size: %s"%cropped.get_shape().as_list())

    return cropped

if __name__ == "__main__":

    # z    [1, 1, 1]:  66 ->  38 -> 10
    # y, x [2, 2, 2]: 228 -> 140 -> 52
    shape_0 = (220,)*3
    shape_1 = (132,)*3
    shape_2 = (44,)*3
    ignore=False

    affs_0_batched = tf.ones((1, 3,) + shape_0)*0.5

    with tf.variable_scope('autocontext') as scope:

        # phase 1

        raw_0 = tf.placeholder(tf.float32, shape=shape_0)
        raw_0_batched = tf.reshape(raw_0, (1, 1) + shape_0)

        input_0 = tf.concat([raw_0_batched, affs_0_batched], 1)
        if ignore:
            keep_raw = tf.ones_like(raw_0_batched)
            ignore_aff = tf.zeros_like(affs_0_batched)
            ignore_mask = tf.concat([keep_raw, ignore_aff], 1)
            input_0 = networks.ignore(input_0, ignore_mask)

        unet = networks.unet(input_0, 24, 3, [[2,2,2],[2,2,2],[2,2,2]])

        affs_1_batched = networks.conv_pass(
            unet,
            kernel_size=1,
            num_fmaps=3,
            num_repetitions=1,
            activation='sigmoid')

        affs_1 = tf.reshape(affs_1_batched, (3,) + shape_1)
        gt_affs_1 = tf.placeholder(tf.float32, shape=(3,) + shape_1)
        loss_weights_1 = tf.placeholder(tf.float32, shape=(3,) + shape_1)

        loss_1 = tf.losses.mean_squared_error(
            gt_affs_1,
            affs_1,
            loss_weights_1)

        # phase 2
        tf.summary.scalar('loss_pred0', loss_1)
        scope.reuse_variables()
        tf.stop_gradient(affs_1_batched)
        raw_1 = center_crop(raw_0, shape_1)
        raw_1_batched = tf.reshape(raw_1, (1, 1) + shape_1)

        input_1 = tf.concat([raw_1_batched, affs_1_batched], 1)

        unet = networks.unet(input_1, 24, 3, [[2,2,2],[2,2,2],[2,2,2]])

        affs_2_batched = networks.conv_pass(
            unet,
            kernel_size=1,
            num_fmaps=3,
            num_repetitions=1,
            activation='sigmoid')

        affs_2 = tf.reshape(affs_2_batched, (3,) + shape_2)
        gt_affs_2 = center_crop(gt_affs_1, (3,) + shape_2)
        loss_weights_2 = center_crop(loss_weights_1, (3,) + shape_2)

        loss_2 = tf.losses.mean_squared_error(
            gt_affs_2,
            affs_2,
            loss_weights_2)
        tf.summary.scalar('loss_pred1', loss_2)
    loss = loss_1 + loss_2
    tf.summary.scalar('loss_total', loss)
    tf.summary.scalar('loss_diff', loss_1-loss_2)
    for trainable in tf.trainable_variables():
        networks.tf_var_summary(trainable)
    merged = tf.summary.merge_all()

    opt = tf.train.AdamOptimizer(
        learning_rate=0.5e-4,
        beta1=0.95,
        beta2=0.999,
        epsilon=1e-8)
    optimizer = opt.minimize(loss)

    tf.train.export_meta_graph(filename='wnet.meta')

    names = {
        'raw': raw_0.name,
        'affs_1': affs_1.name,
        'affs_2': affs_2.name,
        'gt_affs_1': gt_affs_1.name,
        'loss_weights_1': loss_weights_1.name,
        'loss': loss.name,
        'optimizer': optimizer.name,
        'summary': merged.name}
    with open('net_io_names.json', 'w') as f:
        json.dump(names, f)
