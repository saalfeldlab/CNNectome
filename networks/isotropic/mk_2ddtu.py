from networks import unet2d
import tensorflow as tf
import json


def train_net():
    input_shape = (256, 256)
    raw_r = tf.placeholder(tf.float32, shape=input_shape)
    raw_g = tf.placeholder(tf.float32, shape=input_shape)
    raw_b = tf.placeholder(tf.float32, shape=input_shape)
    raw = tf.stack([raw_r, raw_g, raw_b], axis=0)
    raw_batched = tf.reshape(raw, (1, 3,) + input_shape)

    last_fmap, fov, anisotropy = unet2d.unet(raw_batched, 12, 6, [[3, 3], [3, 3], [3, 3]],
                                           [[(3, 3), (3, 3)], [(3, 3), (3, 3)],
                                            [(3, 3), (3, 3)], [(3, 3), (3, 3)]],
                                           [[(3, 3), (3, 3)], [(3, 3), (3, 3)],
                                            [(3, 3), (3, 3)], [(3, 3), (3, 3)]],
                                           voxel_size=(1, 1), fov=(1, 1))

    res_batched, fov = unet2d.conv_pass(
            last_fmap,
            kernel_size=[[1, 1]],
            num_fmaps=7,
            activation=None,
            fov=fov,
            voxel_size=anisotropy
            )
    aff1_0, aff1_1, aff3_0, aff3_1, aff9_0, aff9_1, dist = tf.unstack(res_batched, 7, axis=1)
    aff1 = tf.stack([aff1_0, aff1_1], axis=1)
    aff3 = tf.stack([aff3_0, aff3_1], axis=1)
    aff9 = tf.stack([aff9_0, aff9_1], axis=1)
    output_shape = dist.get_shape().as_list()

    gt_dist = tf.placeholder(tf.float32, shape=output_shape)
    gt_aff = tf.placeholder(tf.float32, shape=[1, 6, ]+output_shape[1:])
    gt_aff1_0, gt_aff1_1, gt_aff3_0, gt_aff3_1, gt_aff9_0, gt_aff9_1 = tf.unstack(gt_aff, 6, axis=1)
    gt_aff1 = tf.stack([gt_aff1_0, gt_aff1_1], axis=1)
    gt_aff3 = tf.stack([gt_aff3_0, gt_aff3_1], axis=1)
    gt_aff9 = tf.stack([gt_aff9_0, gt_aff9_1], axis=1)

    loss_weights_dist = tf.placeholder(tf.float32, shape=output_shape[1:])
    loss_weights_aff = tf.placeholder(tf.float32, shape=output_shape[1:])
    loss_weights_batched_dist = tf.reshape(loss_weights_dist, shape=output_shape)
    loss_weights_batched_aff = tf.reshape(loss_weights_aff, shape=[1, ]+output_shape)

    loss_dist = tf.losses.mean_squared_error(
        gt_dist,
        dist,
        loss_weights_batched_dist
    )
    loss_aff1 = tf.losses.mean_squared_error(gt_aff1, aff1, loss_weights_batched_aff)
    loss_aff3 = tf.losses.mean_squared_error(gt_aff3, aff3, loss_weights_batched_aff)
    loss_aff9 = tf.losses.mean_squared_error(gt_aff9, aff9, loss_weights_batched_aff)
    loss_total = loss_dist * 3 + loss_aff1 + loss_aff3 + loss_aff9
    tf.summary.scalar('loss_dist', loss_dist)
    tf.summary.scalar('loss_aff1', loss_aff1)
    tf.summary.scalar('loss_aff3', loss_aff3)
    tf.summary.scalar('loss_aff9', loss_aff9)
    tf.summary.scalar('loss_total', loss_total)

    loss_unbalanced_dist = tf.losses.mean_squared_error(gt_dist, dist)
    loss_unbalanced_aff1 = tf.losses.mean_squared_error(gt_aff1, aff1)
    loss_unbalanced_aff3 = tf.losses.mean_squared_error(gt_aff3, aff3)
    loss_unbalanced_aff9 = tf.losses.mean_squared_error(gt_aff9, aff9)
    tf.summary.scalar('loss_unbalanced_dist', loss_unbalanced_dist)
    tf.summary.scalar('loss_unbalanced_aff1', loss_unbalanced_aff1)
    tf.summary.scalar('loss_unbalanced_aff3', loss_unbalanced_aff3)
    tf.summary.scalar('loss_unbalanced_aff9', loss_unbalanced_aff9)

    opt = tf.train.AdamOptimizer(
        learning_rate=0.5e-4,
        beta1=0.95,
        beta2=0.999,
        epsilon=1e-8)

    optimizer = opt.minimize(loss_total)
    merged = tf.summary.merge_all()

    tf.train.export_meta_graph(filename='unet.meta')

    names = {
        'raw': raw.name,
        'raw_r': raw_r.name,
        'raw_g': raw_g.name,
        'raw_b': raw_b.name,
        'dist': dist.name,
        'aff1': aff1.name,
        'aff3': aff3.name,
        'aff9': aff9.name,
        'gt_dist': gt_dist.name,
        'gt_aff': gt_aff.name,
        'loss_dist': loss_dist.name,
        'loss_aff1': loss_aff1.name,
        'loss_aff3': loss_aff3.name,
        'loss_aff9': loss_aff9.name,
        'loss_total': loss_total.name,
        'loss_unbalanced_dist': loss_unbalanced_dist.name,
        'loss_unbalanced_aff1': loss_unbalanced_aff1.name,
        'loss_unbalanced_aff3': loss_unbalanced_aff3.name,
        'loss_unbalanced_aff9': loss_unbalanced_aff9.name,
        'loss_weights_dist': loss_weights_dist.name,
        'loss_weights_aff': loss_weights_aff.name,
        'optimizer': optimizer.name,
        'summary': merged.name}

    with open('net_io_names.json', 'w') as f:
        json.dump(names, f)


def inference_net():
    input_shape = (3, 256, 256)
    raw = tf.placeholder(tf.float32, shape=input_shape)
    raw_batched = tf.reshape(raw, (1,) + input_shape)

    last_fmap, fov, anisotropy = unet2d.unet(raw_batched, 12, 6, [[3, 3], [3, 3], [3, 3]],
                                           [[(3, 3), (3, 3)], [(3, 3), (3, 3)],
                                            [(3, 3), (3, 3)], [(3, 3), (3, 3)]],
                                           [[(3, 3), (3, 3)], [(3, 3), (3, 3)],
                                            [(3, 3), (3, 3)], [(3, 3), (3, 3)]],
                                           voxel_size=(1, 1), fov=(1, 1))

    res_batched, fov = unet2d.conv_pass(
        last_fmap,
        kernel_size=[[1, 1]],
        num_fmaps=7,
        activation=None,
        fov=fov,
        voxel_size=anisotropy
    )

    aff1_0, aff1_1, aff3_0, aff3_1, aff9_0, aff9_1, dist = tf.unstack(res_batched, 7, axis=1)
    aff1 = tf.stack([aff1_0, aff1_1], axis=1)
    aff3 = tf.stack([aff3_0, aff3_1], axis=1)
    aff9 = tf.stack([aff9_0, aff9_1], axis=1)
    output_shape = dist.get_shape().as_list()

    tf.train.export_meta_graph(filename='unet_inference.meta')


if __name__ == '__main__':
    train_net()
    tf.reset_default_graph()
    inference_net()
